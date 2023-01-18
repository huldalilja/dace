# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, Dict, Optional
from dace import dtypes, memlet as mm, properties, data as dt
from dace.codegen.prettycode import CodeIOStream
from dace.libraries.standard.environments.cuda import CUDA as cudaenv
from dace.symbolic import symstr
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
# For optimizations
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import (to_blastype, get_gemm_opts, check_access, dtype_to_cudadatatype,
                                              to_cublas_computetype)
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from .. import environments
import numpy as np
from numbers import Number


def _is_complex(dtype):
    if hasattr(dtype, "is_complex") and callable(dtype.is_complex):
        return dtype.is_complex()
    else:
        return dtype in [np.complex64, np.complex128]


def _cast_to_dtype_str(value, dtype: dace.dtypes.typeclass) -> str:
    if _is_complex(dtype) and _is_complex(type(value)):
        raise ValueError("Cannot use complex beta with non-complex array")

    if _is_complex(dtype):
        cast_value = complex(value)

        return "dace.{type}({real}, {imag})".format(
            type=dace.DTYPE_TO_TYPECLASS[dtype].to_string(),
            real=cast_value.real,
            imag=cast_value.imag,
        )
    else:
        return "dace.{}({})".format(dace.DTYPE_TO_TYPECLASS[dtype].to_string(), value)


@dace.library.expansion
class ExpandGemmPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b, shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_b).type]

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        if (len(trans_shape_a) != 2 or len(trans_shape_b) != 2 or trans_shape_a[1] != trans_shape_b[0]):
            raise SyntaxError("Matrix sizes must match")
        M, K, N = trans_shape_a[0], trans_shape_a[1], trans_shape_b[1]
        shape_c = (M, N)

        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-1], storage=cdata[1].storage)

        if node.alpha == 1.0:
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))

        if node.beta == 1:
            state = sdfg.add_state(node.label + "_state")
        else:
            init_state = sdfg.add_state(node.label + "_initstate")
            state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta != 0:
            sdfg.add_array("_cin", shape_c, dtype_c, strides=cdata[-1], storage=cdata[1].storage)

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        # Initialization / beta map
        if node.beta == 0:
            init_state.add_mapped_tasklet(
                'gemm_init', {'_o%d' % i: '0:%s' % symstr(d)
                              for i, d in enumerate(shape_c)}, {},
                'out = 0', {'out': dace.Memlet.simple(mul_out, ','.join(['_o%d' % i for i in range(len(shape_c))]))},
                external_edges=True)
        elif node.beta == 1:
            # Do nothing for initialization, only update the values
            pass
        else:
            # Beta map
            add_program = "__y = ({} * __c)".format(_cast_to_dtype_str(node.beta, dtype_a))

            # manually broadcasting C to [M, N]
            if list(shape_c) == [M, N]:
                memlet_idx = '__i0, __i1'
            elif list(shape_c) == [1, N]:
                memlet_idx = '0, __i1'
            elif list(shape_c) == [M, 1]:
                memlet_idx = '__i0, 0'
            elif list(shape_c) == [N]:
                memlet_idx = '__i1'
            else:
                raise ValueError("Could not broadcast input _c to ({}, {})".format(M, N))

            init_state.add_mapped_tasklet("gemm_init", {"__i%d" % i: "0:%s" % s
                                                        for i, s in enumerate([M, N])}, {
                                                            "__c": dace.Memlet.simple("_cin", memlet_idx),
                                                        },
                                          add_program, {"__y": dace.Memlet.simple("_c", "__i0, __i1")},
                                          external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("gemm", {"__i%d" % i: "0:%s" % s
                                          for i, s in enumerate([M, N, K])},
                                 {
                                     "__a": dace.Memlet.simple("_a", "__i2, __i0" if node.transA else "__i0, __i2"),
                                     "__b": dace.Memlet.simple("_b", "__i1, __i2" if node.transB else "__i2, __i1")
                                 },
                                 mul_program,
                                 {"__out": dace.Memlet.simple(mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")},
                                 external_edges=True,
                                 output_nodes=output_nodes)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandGemmPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandGemmOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, astrides), (_, bdesc, bshape, bstrides), _ = _get_matmul_operands(node, state, sdfg)
        dtype = adesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + 'gemm'
        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'

        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)

        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, dtype.ctype, func)

        # Adaptations for BLAS API
        opt['ta'] = 'CblasNoTrans' if opt['ta'] == 'N' else 'CblasTrans'
        opt['tb'] = 'CblasNoTrans' if opt['tb'] == 'N' else 'CblasTrans'

        code = ''
        if dtype in (dace.complex64, dace.complex128):
            code = f'''
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            opt['alpha'] = '&alpha'
            opt['beta'] = '&beta'

        code += ("cblas_{func}(CblasColMajor, {ta}, {tb}, "
                 "{M}, {N}, {K}, {alpha}, {x}, {lda}, {y}, {ldb}, {beta}, "
                 "_c, {ldc});").format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandGemmMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGemmOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandGemmCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == '_a':
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        # If buffers are not on the GPU, copy them
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))

        dtype = adesc.dtype.base_type
        func = '%sgemm' % to_blastype(dtype.type)
        if dtype == dace.float16:
            cdtype = '__half'
            factort = 'Half'
        elif dtype == dace.float32:
            cdtype = 'float'
            factort = 'Float'
        elif dtype == dace.float64:
            cdtype = 'double'
            factort = 'Double'
        elif dtype == dace.complex64:
            cdtype = 'cuComplex'
            factort = 'Complex64'
        elif dtype == dace.complex128:
            cdtype = 'cuDoubleComplex'
            factort = 'Complex128'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        call_prefix = environments.cublas.cuBLAS.handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        constants = {
            1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Pone()",
            #-1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Mone()",
            0.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Zero()",
        }
        if node.alpha not in constants or node.beta not in constants:
            # Deal with complex input constants
            if isinstance(node.alpha, complex):
                alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
            else:
                alpha = f'{dtype.ctype}({node.alpha})'
            if isinstance(node.beta, complex):
                beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
            else:
                beta = f'{dtype.ctype}({node.beta})'

            # Set pointer mode to host
            call_prefix += f'''cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_HOST);
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            call_suffix += '''cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);'''
            alpha = f'({cdtype} *)&alpha'
            beta = f'({cdtype} *)&beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['arr_prefix'] = arr_prefix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'

        # Matrix multiplication
        if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
            call = '''cublas{func}(__dace_cublas_handle,
                CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
                {M}, {N}, {K},
                {alpha},
                ({dtype}*){arr_prefix}{x}, {lda},
                ({dtype}*){arr_prefix}{y}, {ldb},
                {beta},
                ({dtype}*){arr_prefix}_c, {ldc});'''.format_map(opt)
        else:
            if node.compute_type is not None:
                acctype = node.compute_type
            elif node.accumulator_type is not None:
                acc_dtype: dtypes.typeclass = node.accumulator_type
                acctype = f'CUBLAS_COMPUTE_{to_cublas_computetype(acc_dtype)}'
            else:
                acctype = f'CUBLAS_COMPUTE_{to_cublas_computetype(dtype)}'

            algorithm = 'CUBLAS_GEMM_DEFAULT_TENSOR_OP'
            if node.algorithm is not None:
                algorithm = node.algorithm

            call = f'''
            cublasGemmEx(__dace_cublas_handle,
                CUBLAS_OP_{opt['ta']}, CUBLAS_OP_{opt['tb']},
                {opt['M']}, {opt['N']}, {opt['K']},
                {alpha},
                {arr_prefix}{opt['x']},
                {dtype_to_cudadatatype(opt['xdtype'])},
                {opt['lda']},
                {arr_prefix}{opt['y']},
                {dtype_to_cudadatatype(opt['ydtype'])},
                {opt['ldb']},
                {beta},
                {arr_prefix}_c,
                {dtype_to_cudadatatype(opt['cdtype'])},
                {opt['ldc']},
                {acctype},
                {algorithm});
            '''

        code = (call_prefix + call + call_suffix)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        # If buffers are not on the GPU, copy them
        if needs_copy:
            nsdfg = dace.SDFG('nested_gemm')
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                dcopy.transient = False
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
            nstate = nsdfg.add_state()
            a = nstate.add_read('_a')
            ga = nstate.add_access('_a_gpu')
            b = nstate.add_read('_b')
            gb = nstate.add_access('_b_gpu')
            c = nstate.add_write('_c')
            gc = nstate.add_access('_c_gpu')

            # Reset code and connectors
            tasklet.in_connectors = {"_conn" + k: None for k in tasklet.in_connectors}
            tasklet.out_connectors = {"_conn" + k: None for k in tasklet.out_connectors}

            nstate.add_node(tasklet)
            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(ga, None, tasklet, '_conn_a', dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, tasklet, '_conn_b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '_conn_c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                rc = nstate.add_read('_c')
                rgc = nstate.add_access('_c_gpu')
                tasklet.add_in_connector('_conn_cin')
                nstate.add_nedge(rc, rgc, dace.Memlet('_c'))
                nstate.add_edge(rgc, None, tasklet, '_conn_cin', dace.Memlet('_c_gpu'))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.expansion
class ExpandGemmTensorCore2(ExpandTransformation):
    # TODO hhannesdo, need to check somewhere if device has Tensor Cores

    environments = [cudaenv]

    @staticmethod
    def expansion(node, state, sdfg):
        print("Fast Expand Tensor Core Version")
        node.validate(sdfg, state)

        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == '_a':
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        # If buffers are not on the GPU, copy them
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))

        adtype = adesc.dtype.base_type
        bdtype = bdesc.dtype.base_type
        cdtype = cdesc.dtype.base_type
        # TODO, for later when using A100, can support higher precision
        # Will need to change things in the codegen too
        if adtype != dace.float16 or bdtype != dace.float16 or (cdtype != dace.float16 and cdtype != dace.float32):
            # TODO convert, if possible or use different architecture like Nvidia A100
            raise ValueError("Unsupported type: " + str(adtype))

        alpha = node.alpha
        beta = node.beta

        # Set up options for code formatting
        dtype = adesc.dtype.base_type
        func = '%sgemm' % to_blastype(dtype.type)
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['arr_prefix'] = arr_prefix = ''
        opt['WMMA_M'] = node.WMMA_M
        opt['WMMA_N'] = node.WMMA_N
        opt['WMMA_K'] = node.WMMA_K
        opt['WARP_SIZE'] = node.WARP_SIZE
        arr_suffix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'
            arr_suffix = '_gpu'
        
        if opt['swap']:
            opt_M = opt['M']
            opt['M'] = opt['N']
            opt['N'] = opt_M
            opt_lda = opt['lda']
            opt['lda'] = opt['ldb']
            opt['ldb'] = opt_lda
            opt['major_order'] = 'row'
        else:
            # TODO hhannesdo, haven't tested. Might need to make changes in maps too.
            opt['lda'] = 'M'
            opt['ldb'] = 'K'
            opt['ldc'] = 'M'
            opt['major_order'] = 'col'

        #######################################################################
        code_io_stream = CodeIOStream()
        host_code_io_stream = CodeIOStream()
        host_code_io_stream.write("""DACE_EXPORTED void calling_gemm(dace::float16* a, dace::float16* b, float* c, float alpha, float beta, int M, int N, int K);
        """)
        sdfg.append_global_code(host_code_io_stream.getvalue())

        code_io_stream.write(
            """#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * WMMA_K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define SHMEM_STRIDE (WMMA_N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (WMMA_N * WARP_ROW_TILES)

#define SKEW_HALF 8

#define checkKernelErrors(expr)                             \\
  do {                                                      \\
    expr;                                                   \\
                                                            \\
    cudaError_t __err = cudaGetLastError();                 \\
    if (__err != cudaSuccess) {                             \\
      printf("Line %d: '%s' failed: %s\\n", __LINE__, #expr, \\
             cudaGetErrorString(__err));                    \\
      abort();                                              \\
    }                                                       \\
  } while (0)

using namespace nvcuda;


__global__ void compute_gemm(const half *A, const half *B, float *C,
                             float alpha, float beta, int M, int N, int K) {
  extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

  const unsigned int M_TILES = M / 16;
  const unsigned int N_TILES = N / 16;
  const unsigned int K_TILES = K / 16;

  const unsigned int GLOBAL_MEM_STRIDE = N;

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * WMMA_K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < WMMA_K; i++) {
      typedef int4 copy_t;

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
            laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K dimension.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES]
                                                       [WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * WMMA_M * K] +
                                           WMMA_M * K * (warpId % 4) * 2)
                                        : (&B[block_tile_j * WMMA_N * K] +
                                           WMMA_N * K * (warpId % 4) * 2);

    // Go through the global WMMA_K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr =
            (int4 *)((half *)lane_ptr + K * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];

          wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * WMMA_N) * (warpId % 2) +
                                   (j * WMMA_N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];

              wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

#pragma unroll
    for (int i = 0; i < WMMA_K; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}

DACE_EXPORTED void calling_gemm(dace::float16* a, dace::float16* b, float* c, float alpha, float beta, int M, int N, int K);
void calling_gemm(dace::float16* a, dace::float16* b, float* c, float alpha, float beta, int M, int N, int K){
    // int dev = findCudaDevice(0, NULL);

    // cudaDeviceProp deviceProp;
    // checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(
        sizeof(half) * (BLOCK_COL_TILES * WMMA_M) * (CHUNK_K * WMMA_K + SKEW_HALF) * 2,
        WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };

  // printf("Required shared memory size: %lu Kb\\n", SHMEM_SZ / 1024UL);

  // If enough shared memory available on the GPU use high performant kernel
  //if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
    // printf("Computing... using high performance kernel compute_gemm \\n");
    // printf("%d\\n", deviceProp.multiProcessorCount);
    checkCudaErrors(cudaFuncSetAttribute(
        compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
    checkKernelErrors(
        (compute_gemm<<<68, THREADS_PER_BLOCK,
                        SHMEM_SZ>>>(a, b, c, alpha, beta, M, N, K)));
    //}
}
"""
        )
        ###############################################################

        sdfg.append_global_code(code_io_stream.getvalue(), 'cuda')
        code = """calling_gemm(_a, _b, _c, {alpha}, {beta}, {M}, {N}, {K});""".format_map(opt)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        
        return tasklet

        # Creating nested SDFG
        nsdfg = dace.SDFG('nested_gemm')
        global_code = '''
    #ifdef __CUDACC__
    #include <mma.h>
    using namespace nvcuda;
    #endif

    const int WMMA_M = {WMMA_M};
    const int WMMA_N = {WMMA_N};
    const int WMMA_K = {WMMA_K};
'''.format_map(opt)
        # Appending the global code to the CUDA-generated file.
        if ('cuda' not in nsdfg.global_code or 'mma.h' not in nsdfg.global_code['cuda'].code):
            nsdfg.append_global_code(global_code, 'cuda')
        
        nstate = nsdfg.add_state()

        # Creating maps
        map_entry, map_exit = nstate.add_map(
            node.name,
            dict(i='0:{M}:{WMMA_M}'.format_map(opt), j='0:{N}:{WMMA_N}'.format_map(opt)),
            dace.dtypes.ScheduleType.GPU_Device
        )

        map_entry.add_in_connector('IN_a')
        map_entry.add_in_connector('IN_b')
        map_entry.add_out_connector('OUT_a')
        map_entry.add_out_connector('OUT_b')
        map_exit.add_in_connector('IN_c')
        map_exit.add_out_connector('OUT_c')

        warp_map_entry, warp_map_exit = nstate.add_map(
            'warp_map',
            dict(_='0:{WARP_SIZE}'.format_map(opt)),
            dace.dtypes.ScheduleType.GPU_ThreadBlock
        )

        warp_map_entry.add_in_connector('IN' + arr_prefix + '_a')
        warp_map_entry.add_in_connector('IN' + arr_prefix + '_b')
        warp_map_entry.add_out_connector('OUT' + arr_prefix + '_a')
        warp_map_entry.add_out_connector('OUT' + arr_prefix + '_b')
        warp_map_exit.add_in_connector('IN' + arr_prefix + '_c')
        warp_map_exit.add_out_connector('OUT' + arr_prefix + '_c')

        # Creating second nested SDFG for code inside kernel
        ksdfg = dace.SDFG('kernel_gemm')

        wmma_state = ksdfg.add_state('wmma_state')
        k_loop_after_state = None
        ksdfg.add_array('acctile', (opt['WMMA_M'], opt['WMMA_N']), cdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_Accumulator, transient = True)

        # Cleanup code for rest of gemm
        # C = alpha * (A @ B) + beta * C 
        if(alpha != 1 or beta != 0):
            comp_state = ksdfg.add_state('comp_state')
            acctile = comp_state.add_access('acctile')
            k_loop_after_state = comp_state
            if(beta != 0):
                cslice = comp_state.add_read('_c')
                ksdfg.add_array('ctile', (opt['WMMA_M'], opt['WMMA_N']), cdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_Accumulator, transient = True)
                ctile = comp_state.add_access('ctile')
                comp_state.add_edge(cslice, None, ctile, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
                opt["beta_code"] = " + {beta} * cfragin.x[l]".format_map(opt)
            code = '''#pragma unroll
for(int l = 0; l < {WMMA_M}*{WMMA_N}; l++){{
    cfrag.x[l] = {alpha} * accfrag.x[l]{beta_code};
}}'''.format_map(opt)
            comp_tasklet = comp_state.add_tasklet('comp', None, None, code, language=dace.dtypes.Language.CPP)
            comp_tasklet.add_in_connector('accfrag')
            comp_state.add_edge(acctile, None, comp_tasklet, 'accfrag', dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            if beta != 0:
                comp_tasklet.add_in_connector('cfragin')
                comp_state.add_edge(ctile, None, comp_tasklet, 'cfragin', dace.Memlet(data="ctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            comp_tasklet.add_out_connector('cfrag')
            ctile_out = comp_state.add_access('ctile')
            comp_state.add_edge(comp_tasklet, 'cfrag',  ctile_out, None, dace.Memlet(data="ctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            _c_out = comp_state.add_write('_c')
            comp_state.add_edge(ctile_out, 'cfrag',  _c_out, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
        else:
            write_state = ksdfg.add_state('write_state')
            c_out = write_state.add_write('_c')
            acctile = write_state.add_access('acctile')
            write_state.add_edge(acctile, None, c_out, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            k_loop_after_state = write_state
        
        # Adding k-loop inside kernel (seq. map)
        ksdfg.add_loop(None, wmma_state, k_loop_after_state, 'k', '0', '(k < {K})'.format_map(opt), '(k + {WMMA_K})'.format_map(opt), wmma_state)

        aslice = wmma_state.add_read('_a')
        bslice = wmma_state.add_read('_b')
        atile = wmma_state.add_array('atile', (opt['WMMA_M'], opt['WMMA_K']), adesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_A, transient = True)
        btile = wmma_state.add_array('btile', (opt['WMMA_K'], opt['WMMA_N']), bdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_B, transient = True)
        wmma_state.add_edge(aslice, None, atile, None, dace.Memlet(data="_a", subset='0:{WMMA_M}, k:k + {WMMA_K}'.format_map(opt)))
        wmma_state.add_edge(bslice, None, btile, None, dace.Memlet(data="_b", subset='k:k + {WMMA_K}, 0:{WMMA_N}'.format_map(opt)))
        wmma_tasklet = wmma_state.add_tasklet('wmma', None, None, "wmma::mma_sync(accfrag, afrag, bfrag, accfrag);", language=dace.dtypes.Language.CPP)
        wmma_tasklet.add_in_connector('afrag')
        wmma_tasklet.add_in_connector('bfrag')
        wmma_tasklet.add_out_connector('accfrag')
        wmma_state.add_edge(atile, None, wmma_tasklet, 'afrag', dace.Memlet(data="atile", subset='0:{WMMA_M}, 0:{WMMA_K}'.format_map(opt)))
        wmma_state.add_edge(btile, None, wmma_tasklet, 'bfrag', dace.Memlet(data="btile", subset='0:{WMMA_K}, 0:{WMMA_N}'.format_map(opt)))
        acctile = wmma_state.add_access('acctile')
        acctile.setzero = True
        wmma_state.add_edge(wmma_tasklet, 'accfrag', acctile, None, dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
        
        # Adding data descs to innermost nested SDFG
        for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
            if isinstance(desc, dt.View):
                dcopy = desc.as_array()
            else:
                dcopy = dc(desc)
            dcopy.transient = False
            dcopy.storage = dace.StorageType.GPU_Global
            ksdfg.add_datadesc(name, dcopy)

        nksdfg_in_conn = {'_a', '_b', '_c'} if beta != 0 else {'_a', '_b'} # TODO change to _cin, use node.in_connectors
        nksdfg = nstate.add_nested_sdfg(ksdfg, nstate, nksdfg_in_conn, node.out_connectors, name='kernel_gemm_nsdfg')
        
        # Add connectors between GPU map and warp map
        nstate.add_edge(map_entry, 'OUT_a', warp_map_entry, 'IN' + arr_prefix + '_a', dace.Memlet(data="_a" + arr_suffix, subset='i:i+{WMMA_M}, 0:{K}'.format_map(opt)))
        nstate.add_edge(map_entry, 'OUT_b', warp_map_entry, 'IN' + arr_prefix + '_b', dace.Memlet(data="_b" + arr_suffix, subset='0:{K}, j:j+{WMMA_N}'.format_map(opt)))
        nstate.add_edge(warp_map_exit, 'OUT' + arr_prefix + '_c', map_exit, 'IN_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))

        # Add connectors between warp (Block) map and tasklet
        nstate.add_edge(warp_map_entry, 'OUT' + arr_prefix + '_a', nksdfg, '_a', dace.Memlet(data="_a" + arr_suffix, subset='i:i+{WMMA_M}, 0:{K}'.format_map(opt)))
        nstate.add_edge(warp_map_entry, 'OUT' + arr_prefix + '_b', nksdfg, '_b', dace.Memlet(data="_b" + arr_suffix, subset='0:{K}, j:j+{WMMA_N}'.format_map(opt)))
        nstate.add_edge(nksdfg, '_c', warp_map_exit, 'IN' + arr_prefix + '_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))
        
        if node.beta != 0.0:
            rc = nstate.add_read('_c')
            map_entry.add_in_connector('IN_c')
            map_entry.add_out_connector('OUT_c')
            warp_map_entry.add_in_connector('IN' + arr_prefix + '_c')
            warp_map_entry.add_out_connector('OUT' + arr_prefix + '_c')
            nstate.add_edge(map_entry, 'OUT_c', warp_map_entry, 'IN' + arr_prefix + '_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))
            nstate.add_edge(warp_map_entry, 'OUT' + arr_prefix + '_c', nksdfg, '_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))

        if needs_copy:
            # If buffers are not on the GPU, copy them
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                dcopy.transient = False
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
            a = nstate.add_read('_a')
            b = nstate.add_read('_b')
            c = nstate.add_write('_c')
            ga = nstate.add_access('_a_gpu')
            gb = nstate.add_access('_b_gpu')
            gc = nstate.add_access('_c_gpu')

            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(ga, None, map_entry, 'IN_a', dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, map_entry, 'IN_b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(map_exit, 'OUT_c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                rgc = nstate.add_access('_c_gpu')
                nstate.add_nedge(rc, rgc, dace.Memlet('_c'))
                nstate.add_edge(rgc, None, map_entry, 'IN_c', dace.Memlet.from_array('_c_gpu', cdesc))
        else:
            # Arrays already on GPU
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.transient = False
                dcopy.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name, dcopy)
            a = nstate.add_read('_a')
            b = nstate.add_read('_b')
            c = nstate.add_write('_c')

            nstate.add_edge(a, None, map_entry, 'IN_a', dace.Memlet.from_array('_a', adesc))
            nstate.add_edge(b, None, map_entry, 'IN_b', dace.Memlet.from_array('_b', bdesc))
            nstate.add_edge(map_exit, 'OUT_c', c, None, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                nstate.add_edge(rc, None, map_entry, 'IN_c', dace.Memlet.from_array('_c', cdesc))
        ##### Optimization tests #####
        # xfutil.tile(nsdfg, map_entry, True, False, i=256, j=256)
        
        return nsdfg


# Version that is based on matmul gpu_optimization sample, tiled for shared memory
@dace.library.expansion
class ExpandGemmTensorCore3(ExpandTransformation):
    # TODO hhannesdo, need to check somewhere if device has Tensor Cores

    environments = [cudaenv]

    @staticmethod
    def expansion(node, state, sdfg):
        print("Optimized Expand Tensor Core Version")

        # Import here to avoid circular dependency
        from dace.transformation.dataflow import DoubleBuffering

        node.validate(sdfg, state)

        ##############################
        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == '_a':
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
            elif e.dst_conn == '_cin':
                cinnode = state.memlet_path(e)[0].src
                if isinstance(cinnode, dace.sdfg.nodes.AccessNode):
                    cindesc: dt.Array = sdfg.arrays[cinnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        # If buffers are not on the GPU, copy them
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))

        adtype = adesc.dtype.base_type
        bdtype = bdesc.dtype.base_type
        cdtype = cdesc.dtype.base_type
        # TODO, for later when using A100, can support higher precision
        # Will need to change things in the codegen too
        if adtype != dace.float16 or bdtype != dace.float16 or (cdtype != dace.float16 and cdtype != dace.float32):
            # TODO convert, if possible or use different architecture like Nvidia A100
            raise ValueError("Unsupported type: " + str(adtype))

        alpha = node.alpha
        beta = node.beta

        ##############################
        # Set up options for code formatting
        dtype = adesc.dtype.base_type
        func = '%sgemm' % to_blastype(dtype.type)
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['arr_prefix'] = arr_prefix = ''
        opt['WMMA_M'] = node.WMMA_M
        opt['WMMA_N'] = node.WMMA_N
        opt['WMMA_K'] = node.WMMA_K
        opt['WARP_SIZE'] = node.WARP_SIZE
        arr_suffix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'
            arr_suffix = '_gpu'
        
        if opt['swap']:
            opt_M = opt['M']
            opt['M'] = opt['N']
            opt['N'] = opt_M
            opt_lda = opt['lda']
            opt['lda'] = opt['ldb']
            opt['ldb'] = opt_lda
            opt['major_order'] = 'row'
        else:
            # TODO hhannesdo, haven't tested. Might need to make changes in maps too.
            opt['lda'] = 'M'
            opt['ldb'] = 'K'
            opt['ldc'] = 'M'
            opt['major_order'] = 'col'
        
        # Setting tiling and skewing parameters
        # TODO hhannesdo find best values
        opt['WM'] = 'WM'
        opt['WN'] = 'WN'
        
        opt['SM'] =  f"({opt['WMMA_M']} * {opt['WM']})"
        opt['SN'] =  f"({opt['WMMA_N']} * {opt['WN']})" 
        opt['SK'] =  opt['WMMA_K']

        opt['SSKEW'] = 'SSKEW'

        ##############################
        # Creating nested SDFG
        sdfg.add_constant('WM', 4)
        sdfg.add_constant('WN', 4)
        sdfg.add_constant('SSKEW', 8)
        nsdfg = dace.SDFG('nested_gemm')
        global_code = '''
    #ifdef __CUDACC__
    #include <mma.h>
    using namespace nvcuda;
    #endif

    #define WMMA_M {WMMA_M}
    #define WMMA_N {WMMA_N}
    #define WMMA_K {WMMA_K}
'''.format_map(opt)

        # Appending the global code to the CUDA-generated file.
        if ('cuda' not in nsdfg.global_code or 'mma.h' not in nsdfg.global_code['cuda'].code):
            nsdfg.append_global_code(global_code, 'cuda')
        
        nstate = nsdfg.add_state()

        ##############################
        # Creating first map, iterating slices
        map_entry, map_exit = nstate.add_map(
            node.name,
            dict(j='0:{N}:{SN}'.format_map(opt), i='0:{M}:{SM}'.format_map(opt)),
            dace.dtypes.ScheduleType.GPU_Device
        )
        
        ##############################
        # Second map for thread block
        warp_map_entry, warp_map_exit = nstate.add_map(
            'warp_map',
            dict(tIdz ='0:{WN}'.format_map(opt), tIdy ='0:{WM}'.format_map(opt), tIdx ='0:{WARP_SIZE}'.format_map(opt)),
            dace.dtypes.ScheduleType.GPU_ThreadBlock
        )
        # warp_i = tIdy * WMMA_M
        # warp_j = tIdz * WMMA_N

        ##############################
        # Innermost k map, for iterating through k dimension
        k_map_entry, k_map_exit = nstate.add_map(
            'k_map',
            dict(k ='0:{K}:{SK}'.format_map(opt)),
            dace.dtypes.ScheduleType.Sequential
        )
        
        ##############################
        # Adding shared memory of sizes SM*SK for a and SK*SN for b
        # Each warp(WM*WN warps) then computes a WMMA_M * WMMA_N of the output, in accumulation, while readong from shared memory accordingly  
        nsdfg.add_array('shared_a', (opt['SM'], f"({opt['SK']} + {opt['SSKEW']})"), adesc.dtype, storage=dtypes.StorageType.GPU_Shared, transient = True)
        nsdfg.add_array('shared_b', (opt['SK'], f"({opt['SN']} + {opt['SSKEW']})"), bdesc.dtype, storage=dtypes.StorageType.GPU_Shared, transient = True)
        # nsdfg.add_array('shared_c', (opt['SM'], opt['SN']), cdesc.dtype, storage=dtypes.StorageType.GPU_Shared, transient = True)
        ashared = nstate.add_access('shared_a')
        bshared = nstate.add_access('shared_b')
        # c_shared_out = nstate.add_access('shared_c')

        ##############################
        # Adding Tensor Core arrays
        nsdfg.add_array('atile', (opt['WMMA_M'], opt['WMMA_K']), adesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_A, transient = True)
        nsdfg.add_array('btile', (opt['WMMA_K'], opt['WMMA_N']), bdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_B, transient = True)
        nsdfg.add_array('acctile', (opt['WMMA_M'], opt['WMMA_N']), cdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_Accumulator, transient = True)
        atile = nstate.add_access('atile')
        btile = nstate.add_access('btile')
        acctile = nstate.add_access('acctile')
        acctile.setzero = True

        ##############################
        # Moving data from shared memory to Tensor Core tiles
        nstate.add_edge(ashared, None, atile, None, dace.Memlet(data="shared_a", subset='tIdy*{WMMA_M}:tIdy*{WMMA_M}+{WMMA_M}, 0:{SK}'.format_map(opt)))
        nstate.add_edge(bshared, None, btile, None, dace.Memlet(data="shared_b", subset='0:{SK},tIdz*{WMMA_N}:tIdz*{WMMA_N}+{WMMA_N}'.format_map(opt)))

        ##############################
        # Creating tasklet for computing the Tensor Core accumulated matrix multiplication
        wmma_tasklet = nstate.add_tasklet('wmma', None, None, "wmma::mma_sync(accfrag, afrag, bfrag, accfrag);", language=dace.dtypes.Language.CPP)
        wmma_tasklet.add_in_connector('afrag')
        wmma_tasklet.add_in_connector('bfrag')
        wmma_tasklet.add_out_connector('accfrag')
        nstate.add_edge(atile, None, wmma_tasklet, 'afrag', dace.Memlet(data="atile", subset='0:{WMMA_M}, 0:{WMMA_K}'.format_map(opt)))
        nstate.add_edge(btile, None, wmma_tasklet, 'bfrag', dace.Memlet(data="btile", subset='0:{WMMA_K}, 0:{WMMA_N}'.format_map(opt)))
        nstate.add_edge(wmma_tasklet, 'accfrag', acctile, None, dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
        
        ##############################
        # If buffers are not on the GPU, copy them
        copy_descs = [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]
        if beta != 0.0:
            copy_descs.append(('_cin', cindesc))
        if needs_copy:
            for name, desc in copy_descs:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                dcopy.transient = False
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
            a = nstate.add_access('_a')
            b = nstate.add_access('_b')
            c = nstate.add_access('_c')
            ga = nstate.add_access('_a_gpu')
            gb = nstate.add_access('_b_gpu')
            gc = nstate.add_access('_c_gpu')

            # Adding memlets, connecting everything together
            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))
            nstate.add_memlet_path(ga, map_entry, warp_map_entry, k_map_entry, ashared, memlet=dace.Memlet(data="_a_gpu", subset='i:i + {SM}, k:k + {SK}'.format_map(opt), other_subset='0:{SM}, 0:{SK}'.format_map(opt)))
            nstate.add_memlet_path(gb, map_entry, warp_map_entry, k_map_entry, bshared, memlet=dace.Memlet(data="_b_gpu", subset='k:k + {SK}, j:j + {SN}'.format_map(opt), other_subset='0:{SK}, 0:{SN}'.format_map(opt)))
            nstate.add_memlet_path(acctile, k_map_exit, warp_map_exit, map_exit, gc, memlet=dace.Memlet(data="_c_gpu", subset='i + tIdy*{WMMA_M}:i + tIdy*{WMMA_M} + {WMMA_M}, '
                                                                                                                              'j + tIdz*{WMMA_N}:j + tIdz*{WMMA_N} + {WMMA_N}'.format_map(opt)))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

        else:
            # Arrays already on GPU
            for name, desc in copy_descs:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.transient = False
                dcopy.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name, dcopy)
            a = nstate.add_access('_a')
            b = nstate.add_access('_b')
            c = nstate.add_access('_c')

            # Adding memlets, connecting everything together
            nstate.add_memlet_path(a, map_entry, warp_map_entry, k_map_entry, ashared, memlet=dace.Memlet(data="_a", subset='i:i + {SM}, k:k + {SK}'.format_map(opt), other_subset='0:{SM}, 0:{SK}'.format_map(opt)))
            nstate.add_memlet_path(b, map_entry, warp_map_entry, k_map_entry, bshared, memlet=dace.Memlet(data="_b", subset='k:k + {SK}, j:j + {SN}'.format_map(opt), other_subset='0:{SK}, 0:{SN}'.format_map(opt)))
            nstate.add_memlet_path(acctile, k_map_exit, warp_map_exit, map_exit, c, memlet=dace.Memlet(data="_c", subset='i + tIdy*{WMMA_M}:i + tIdy*{WMMA_M} + {WMMA_M}, '
                                                                                                                         'j + tIdz*{WMMA_N}:j + tIdz*{WMMA_N} + {WMMA_N}'.format_map(opt)))

        DoubleBuffering.apply_to(nsdfg, map_entry=k_map_entry, transient=ashared)
        for st, _ in nsdfg.all_nodes_recursive():
            if isinstance(st, SDFGState) and st.label == f'{k_map_entry.label}_double_buffered':
                for n in st.nodes():
                    if isinstance(n, dace.sdfg.nodes.AccessNode) and n.data == 'acctile':
                        for e in st.out_edges(n):
                            st.remove_node(e.dst)
                            # st.remove_edge(e)
                        break
                break
        
        ##############################
        # Cleanup code for rest of gemm
        # C = alpha * (A @ B) + beta * C
        cleanup_code = True if alpha != 1.0 or beta != 0.0 else False
        if(cleanup_code):
            # Computing the rest of the gemm definition
            opt["beta_code"] = " + {beta} * cfrag.x[l]".format_map(opt) if beta != 0.0 else ""
            code = '''#pragma unroll
for(int l = 0; l < {WMMA_M}*{WMMA_N}; l++){{
    accfrag.x[l] = {alpha} * accfragin.x[l]{beta_code};
}}'''.format_map(opt)

            # Finding nodes in SDFG to use for adding the final computation
            for st, _ in nsdfg.all_nodes_recursive():
                if isinstance(st, dace.sdfg.nodes.NestedSDFG):
                    nested_sdfg = st
                    break
            
            for st, _ in nsdfg.all_nodes_recursive():
                if isinstance(st, SDFGState) and st.label == f'{k_map_entry.label}_final_computation':
                    final_state = st
                    for n in st.nodes():
                        if isinstance(n, dace.sdfg.nodes.AccessNode) and n.data == 'acctile':
                            for e in st.out_edges(n):
                                acctile_wmma = n
                                cwrite = e.dst
                                st.remove_edge(e)
                            break
                    break

            # Creating final computation tasklet
            comp_tasklet = final_state.add_tasklet('comp', None, None, code, language=dace.dtypes.Language.CPP)
            comp_tasklet.add_in_connector('accfragin')
            comp_tasklet.add_out_connector('accfrag')

            # Connect with wmma tasklet in final state in k-loop
            acctile_res = final_state.add_access('acctile')
            final_state.add_edge(acctile_wmma, None, comp_tasklet, 'accfragin', dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            final_state.add_edge(comp_tasklet, 'accfrag', acctile_res, None, dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            final_state.add_edge(acctile_res, None, cwrite, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))

            if beta != 0.0:
                # Adding read of C and feeding into final computation tasklet
                nested_sdfg.add_in_connector('_cin')
                rc = nstate.add_access('_cin')
                if needs_copy:
                    grc = nstate.add_access('_cin_gpu')
                    nstate.add_nedge(rc, grc, dace.Memlet.from_array('_cin', cdesc))
                    nstate.add_memlet_path(grc, map_entry, warp_map_entry, nested_sdfg, memlet=dace.Memlet(data="_cin_gpu", subset='i:i+{SM}, j:j+{SN}'.format_map(opt),other_subset='0:{SM}, 0:{SN}'.format_map(opt)), dst_conn='_cin')
                else:
                    nstate.add_memlet_path(rc, map_entry, warp_map_entry, nested_sdfg, memlet=dace.Memlet(data="_cin", subset='i:i+{SM}, j:j+{SN}'.format_map(opt),other_subset='0:{SM}, 0:{SN}'.format_map(opt)), dst_conn='_cin')

                # Note state.add_array is deprecated, try NestedSDFG.sdfg.add_array and then state.add_access
                nested_sdfg.sdfg.add_array('_cin', (opt['SM'], opt['SN']), cindesc.dtype, storage=dtypes.StorageType.GPU_Global, transient = False, strides = cindesc.strides, total_size=cindesc.total_size)
                cin = final_state.add_access('_cin')

                # TODO try adding array to sdfg and add_access after
                ctile = final_state.add_array('ctile', (opt['WMMA_M'], opt['WMMA_N']), cindesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_Accumulator, transient = True)
                final_state.add_edge(cin, None, ctile, None, dace.Memlet(data="_cin", subset='tIdy*{WMMA_M}:tIdy*{WMMA_M}+{WMMA_M}, tIdz*{WMMA_N}:tIdz*{WMMA_N}+{WMMA_N}'.format_map(opt)))
                comp_tasklet.add_in_connector('cfrag')
                final_state.add_edge(ctile, None, comp_tasklet, 'cfrag', dace.Memlet(data="ctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))       
        ##############################
        
        return nsdfg


@dace.library.expansion
class ExpandGemmTensorCore(ExpandTransformation):
    # TODO hhannesdo, need to check somewhere if device has Tensor Cores

    environments = [cudaenv]

    @staticmethod
    def expansion(node, state, sdfg):
        print("Simple Expand Tensor Core Version")
        node.validate(sdfg, state)

        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == '_a':
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        # If buffers are not on the GPU, copy them
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))

        adtype = adesc.dtype.base_type
        bdtype = bdesc.dtype.base_type
        cdtype = cdesc.dtype.base_type
        # TODO, for later when using A100, can support higher precision
        # Will need to change things in the codegen too
        if adtype != dace.float16 or bdtype != dace.float16 or (cdtype != dace.float16 and cdtype != dace.float32):
            # TODO convert, if possible or use different architecture like Nvidia A100
            raise ValueError("Unsupported type: " + str(adtype))

        alpha = node.alpha
        beta = node.beta

        # Set up options for code formatting
        dtype = adesc.dtype.base_type
        func = '%sgemm' % to_blastype(dtype.type)
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['arr_prefix'] = arr_prefix = ''
        opt['WMMA_M'] = node.WMMA_M
        opt['WMMA_N'] = node.WMMA_N
        opt['WMMA_K'] = node.WMMA_K
        opt['WARP_SIZE'] = node.WARP_SIZE
        arr_suffix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'
            arr_suffix = '_gpu'
        
        if opt['swap']:
            opt_M = opt['M']
            opt['M'] = opt['N']
            opt['N'] = opt_M
            opt_lda = opt['lda']
            opt['lda'] = opt['ldb']
            opt['ldb'] = opt_lda
            opt['major_order'] = 'row'
        else:
            # TODO hhannesdo, haven't tested. Might need to make changes in maps too.
            opt['lda'] = 'M'
            opt['ldb'] = 'K'
            opt['ldc'] = 'M'
            opt['major_order'] = 'col'

        # Creating nested SDFG
        nsdfg = dace.SDFG('nested_gemm')
        global_code = '''
    #ifdef __CUDACC__
    #include <mma.h>
    using namespace nvcuda;
    #endif

    const int WMMA_M = {WMMA_M};
    const int WMMA_N = {WMMA_N};
    const int WMMA_K = {WMMA_K};
'''.format_map(opt)
        # Appending the global code to the CUDA-generated file.
        if ('cuda' not in nsdfg.global_code or 'mma.h' not in nsdfg.global_code['cuda'].code):
            nsdfg.append_global_code(global_code, 'cuda')
        
        nstate = nsdfg.add_state()

        # Creating maps
        map_entry, map_exit = nstate.add_map(
            node.name,
            dict(i='0:{M}:{WMMA_M}'.format_map(opt), j='0:{N}:{WMMA_N}'.format_map(opt)),
            dace.dtypes.ScheduleType.GPU_Device
        )

        map_entry.add_in_connector('IN_a')
        map_entry.add_in_connector('IN_b')
        map_entry.add_out_connector('OUT_a')
        map_entry.add_out_connector('OUT_b')
        map_exit.add_in_connector('IN_c')
        map_exit.add_out_connector('OUT_c')

        warp_map_entry, warp_map_exit = nstate.add_map(
            'warp_map',
            dict(_='0:{WARP_SIZE}'.format_map(opt)),
            dace.dtypes.ScheduleType.GPU_ThreadBlock
        )

        warp_map_entry.add_in_connector('IN' + arr_prefix + '_a')
        warp_map_entry.add_in_connector('IN' + arr_prefix + '_b')
        warp_map_entry.add_out_connector('OUT' + arr_prefix + '_a')
        warp_map_entry.add_out_connector('OUT' + arr_prefix + '_b')
        warp_map_exit.add_in_connector('IN' + arr_prefix + '_c')
        warp_map_exit.add_out_connector('OUT' + arr_prefix + '_c')

        # Creating second nested SDFG for code inside kernel
        ksdfg = dace.SDFG('kernel_gemm')

        wmma_state = ksdfg.add_state('wmma_state')
        k_loop_after_state = None
        ksdfg.add_array('acctile', (opt['WMMA_M'], opt['WMMA_N']), cdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_Accumulator, transient = True)

        # Cleanup code for rest of gemm
        # C = alpha * (A @ B) + beta * C 
        if(alpha != 1 or beta != 0):
            comp_state = ksdfg.add_state('comp_state')
            acctile = comp_state.add_access('acctile')
            k_loop_after_state = comp_state
            if(beta != 0):
                cslice = comp_state.add_read('_c')
                ksdfg.add_array('ctile', (opt['WMMA_M'], opt['WMMA_N']), cdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_Accumulator, transient = True)
                ctile = comp_state.add_access('ctile')
                comp_state.add_edge(cslice, None, ctile, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
                opt["beta_code"] = " + {beta} * cfragin.x[l]".format_map(opt)
            code = '''#pragma unroll
for(int l = 0; l < {WMMA_M}*{WMMA_N}; l++){{
    cfrag.x[l] = {alpha} * accfrag.x[l]{beta_code};
}}'''.format_map(opt)
            comp_tasklet = comp_state.add_tasklet('comp', None, None, code, language=dace.dtypes.Language.CPP)
            comp_tasklet.add_in_connector('accfrag')
            comp_state.add_edge(acctile, None, comp_tasklet, 'accfrag', dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            if beta != 0:
                comp_tasklet.add_in_connector('cfragin')
                comp_state.add_edge(ctile, None, comp_tasklet, 'cfragin', dace.Memlet(data="ctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            comp_tasklet.add_out_connector('cfrag')
            ctile_out = comp_state.add_access('ctile')
            comp_state.add_edge(comp_tasklet, 'cfrag',  ctile_out, None, dace.Memlet(data="ctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            _c_out = comp_state.add_write('_c')
            comp_state.add_edge(ctile_out, 'cfrag',  _c_out, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
        else:
            write_state = ksdfg.add_state('write_state')
            c_out = write_state.add_write('_c')
            acctile = write_state.add_access('acctile')
            write_state.add_edge(acctile, None, c_out, None, dace.Memlet(data="_c", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
            k_loop_after_state = write_state
        
        # Adding k-loop inside kernel (seq. map)
        ksdfg.add_loop(None, wmma_state, k_loop_after_state, 'k', '0', '(k < {K})'.format_map(opt), '(k + {WMMA_K})'.format_map(opt), wmma_state)

        aslice = wmma_state.add_read('_a')
        bslice = wmma_state.add_read('_b')
        atile = wmma_state.add_array('atile', (opt['WMMA_M'], opt['WMMA_K']), adesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_A, transient = True)
        btile = wmma_state.add_array('btile', (opt['WMMA_K'], opt['WMMA_N']), bdesc.dtype, storage=dtypes.StorageType.GPU_TensorCore_B, transient = True)
        wmma_state.add_edge(aslice, None, atile, None, dace.Memlet(data="_a", subset='0:{WMMA_M}, k:k + {WMMA_K}'.format_map(opt)))
        wmma_state.add_edge(bslice, None, btile, None, dace.Memlet(data="_b", subset='k:k + {WMMA_K}, 0:{WMMA_N}'.format_map(opt)))
        wmma_tasklet = wmma_state.add_tasklet('wmma', None, None, "wmma::mma_sync(accfrag, afrag, bfrag, accfrag);", language=dace.dtypes.Language.CPP)
        wmma_tasklet.add_in_connector('afrag')
        wmma_tasklet.add_in_connector('bfrag')
        wmma_tasklet.add_out_connector('accfrag')
        wmma_state.add_edge(atile, None, wmma_tasklet, 'afrag', dace.Memlet(data="atile", subset='0:{WMMA_M}, 0:{WMMA_K}'.format_map(opt)))
        wmma_state.add_edge(btile, None, wmma_tasklet, 'bfrag', dace.Memlet(data="btile", subset='0:{WMMA_K}, 0:{WMMA_N}'.format_map(opt)))
        acctile = wmma_state.add_access('acctile')
        acctile.setzero = True
        wmma_state.add_edge(wmma_tasklet, 'accfrag', acctile, None, dace.Memlet(data="acctile", subset='0:{WMMA_M}, 0:{WMMA_N}'.format_map(opt)))
        
        # Adding data descs to innermost nested SDFG
        for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
            if isinstance(desc, dt.View):
                dcopy = desc.as_array()
            else:
                dcopy = dc(desc)
            dcopy.transient = False
            dcopy.storage = dace.StorageType.GPU_Global
            ksdfg.add_datadesc(name, dcopy)

        nksdfg_in_conn = {'_a', '_b', '_c'} if beta != 0 else {'_a', '_b'} # TODO change to _cin, use node.in_connectors
        nksdfg = nstate.add_nested_sdfg(ksdfg, nstate, nksdfg_in_conn, node.out_connectors, name='kernel_gemm_nsdfg')
        
        # Add connectors between GPU map and warp map
        nstate.add_edge(map_entry, 'OUT_a', warp_map_entry, 'IN' + arr_prefix + '_a', dace.Memlet(data="_a" + arr_suffix, subset='i:i+{WMMA_M}, 0:{K}'.format_map(opt)))
        nstate.add_edge(map_entry, 'OUT_b', warp_map_entry, 'IN' + arr_prefix + '_b', dace.Memlet(data="_b" + arr_suffix, subset='0:{K}, j:j+{WMMA_N}'.format_map(opt)))
        nstate.add_edge(warp_map_exit, 'OUT' + arr_prefix + '_c', map_exit, 'IN_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))

        # Add connectors between warp (Block) map and tasklet
        nstate.add_edge(warp_map_entry, 'OUT' + arr_prefix + '_a', nksdfg, '_a', dace.Memlet(data="_a" + arr_suffix, subset='i:i+{WMMA_M}, 0:{K}'.format_map(opt)))
        nstate.add_edge(warp_map_entry, 'OUT' + arr_prefix + '_b', nksdfg, '_b', dace.Memlet(data="_b" + arr_suffix, subset='0:{K}, j:j+{WMMA_N}'.format_map(opt)))
        nstate.add_edge(nksdfg, '_c', warp_map_exit, 'IN' + arr_prefix + '_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))
        
        if node.beta != 0.0:
            rc = nstate.add_read('_c')
            map_entry.add_in_connector('IN_c')
            map_entry.add_out_connector('OUT_c')
            warp_map_entry.add_in_connector('IN' + arr_prefix + '_c')
            warp_map_entry.add_out_connector('OUT' + arr_prefix + '_c')
            nstate.add_edge(map_entry, 'OUT_c', warp_map_entry, 'IN' + arr_prefix + '_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))
            nstate.add_edge(warp_map_entry, 'OUT' + arr_prefix + '_c', nksdfg, '_c', dace.Memlet(data="_c" + arr_suffix, subset='i:i+{WMMA_M}, j:j+{WMMA_N}'.format_map(opt)))

        if needs_copy:
            # If buffers are not on the GPU, copy them
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                dcopy.transient = False
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
            a = nstate.add_read('_a')
            b = nstate.add_read('_b')
            c = nstate.add_write('_c')
            ga = nstate.add_access('_a_gpu')
            gb = nstate.add_access('_b_gpu')
            gc = nstate.add_access('_c_gpu')

            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(ga, None, map_entry, 'IN_a', dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, map_entry, 'IN_b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(map_exit, 'OUT_c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                rgc = nstate.add_access('_c_gpu')
                nstate.add_nedge(rc, rgc, dace.Memlet('_c'))
                nstate.add_edge(rgc, None, map_entry, 'IN_c', dace.Memlet.from_array('_c_gpu', cdesc))
        else:
            # Arrays already on GPU
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.transient = False
                dcopy.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name, dcopy)
            a = nstate.add_read('_a')
            b = nstate.add_read('_b')
            c = nstate.add_write('_c')

            nstate.add_edge(a, None, map_entry, 'IN_a', dace.Memlet.from_array('_a', adesc))
            nstate.add_edge(b, None, map_entry, 'IN_b', dace.Memlet.from_array('_b', bdesc))
            nstate.add_edge(map_exit, 'OUT_c', c, None, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                nstate.add_edge(rc, None, map_entry, 'IN_c', dace.Memlet.from_array('_c', cdesc))
        ##### Optimization tests #####
        # xfutil.tile(nsdfg, map_entry, True, False, i=256, j=256)
        
        return nsdfg


@dace.library.expansion
class ExpandGemmPBLAS(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, astrides), (_, bdesc, bshape, bstrides), _ = _get_matmul_operands(node, state, sdfg)
        dtype = adesc.dtype.base_type

        if node.beta != 0:
            raise NotImplementedError

        M = ashape[0]
        K = ashape[1]
        N = bshape[1]
        Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
        Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
        try:
            sdfg.add_symbol('Px', dace.int32)
            sdfg.add_symbol('Py', dace.int32)
        except FileExistsError:
            pass

        @dace.program
        def _gemm_pblas(_a: dtype[M, K], _b: dtype[K, N], _c: dtype[M, N]):
            lA = np.empty((M // Px, K // Py), dtype=_a.dtype)
            lB = np.empty((K // Px, N // Py), dtype=_b.dtype)
            dace.comm.BCScatter(_a, lA, (M // Px, K // Py))
            dace.comm.BCScatter(_b, lB, (K // Px, N // Py))
            lC = distr.MatMult(lA, lB, (M, N, K))
            dace.comm.BCGather(lC, _c, (M // Px, N // Py))

        return _gemm_pblas.to_sdfg()


class ExpandGemmFPGA1DSystolic(ExpandTransformation):
    """
    FPGA based implementation of GEMM, using a 1D systolic array.

    Currently it supports non-transposed input matrices, and non-vectorized input array A.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, num_pes=32, tile_size_m=None):
        """
        GEMM node expansion.

        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param num_pes: Number of Processing Elements of the systolic array. By default it is set to 32.

        :param tile_size_m: tiling size considering columns of the input matrix B and resulting matrix C.
                            If B/C are vectorized, the tile size refers to the vectorized container.
                            If set to None, no tiling is used, corresponding to setting the tile size
                            equal to the number of columns of B/C.
        :return:
        """

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b, shape_b, strides_b),
         (edge_c, outer_array_c, shape_c, strides_c)) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_b).type]
        shape_c = (shape_a[0], shape_b[1])
        if node.transA:
            raise NotImplementedError("GEMM FPGA expansion not implemented for transposed A.")
        if node.transB:
            raise NotImplementedError("GEMM FPGA expansion not implemented for transposed B.")

        if outer_array_a.veclen > 1:
            raise NotImplementedError("Vectorization not support for input array A.")

        if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[1] != shape_b[0]:
            raise SyntaxError("Matrix sizes must match")

        if outer_array_b.dtype.veclen != outer_array_c.dtype.veclen:
            raise SyntaxError("Vectorization lengths of B and C must match")

        ######################################################################
        # GEMM Parameters and checks

        # Note: the following sizes consider also vectorization
        vec_width = outer_array_b.dtype.veclen
        vec_type = dace.vector(dtype_c, vec_width)
        N, K, M = shape_a[0], shape_a[1], shape_b[1]

        P = num_pes
        T = tile_size_m
        if T is None:
            T = M

        # we will perform sanity check using T and M. But at this stage, we still
        # don't know to what outer symbol they will map.
        # We try to resolve them to constant if they are symbolic, otherwise we skip the checks
        T_constant = dace.symbolic.resolve_symbol_to_constant(T, parent_sdfg)
        K_constant = dace.symbolic.resolve_symbol_to_constant(K, parent_sdfg)

        # Safe delay: this will be used in the compute state, pipeline scope, to insert
        # a delay between accumulation on the same result if needed.
        # Further explanations are provided in the compute state.

        # Note: this is a platform and type dependent parameter.
        if T_constant is not None:
            L = max(16 - T_constant, 0)
        else:
            L = 0

        # This implementation uses a flattened nested loop, that overlaps feeding,
        # computing and draining phases. Each PE is responsible for computing one
        # tile of one row of the final result C. With the current implementation,
        # A PE needs K*T cycles to compute the results and then P*T clock cycles
        # to fully drain them (draining is distributed across PEs).
        # Therefore, in order to guarantee correctness and deadlock free we have
        # to ensure that the number of cycles needed to drain the results is less
        # or equal to the number of cycles needed to compute them.
        # That is PT <= KT.

        if K_constant is not None and P > K_constant:
            raise ValueError(f"GEMM-FPGA: Number of processing elements {P} must be smaller than the K-dimension {K}.")

        ######################################################################
        # Build the SDFG

        new_sdfg = dace.SDFG(node.label + "_sdfg")
        new_state = new_sdfg.add_state("compute")

        # Add data descriptors

        new_sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        new_sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        new_sdfg.add_array("_c", shape_c, dtype_c, strides=strides_c, storage=outer_array_c.storage)

        if node.beta != 0:
            new_sdfg.add_array("_cin", shape_c, dtype_c, strides=strides_c, storage=outer_array_c.storage)

        def make_read_A(state):

            # A given row of A must be repeated according to B number of tiles
            # Both N and M can be not a multiple of P and T respectively
            entry, exit = state.add_map("read_A", {
                "n0": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "k": f"0:{K}",
                "n1": f"0:{P}"
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            # The reader of A reads one element per clock cycle.
            # Note that if P > T+L, then this will be the bottleneck

            mem = state.add_read("_a")
            pipe = state.add_write("A_pipe")

            # Read data from memory: if we are out-of-bound do not read from memory
            # but inject dummy data
            tasklet = state.add_tasklet("read_A", {"from_memory"}, {"to_kernel"}, f"""\
data = from_memory if n0 * {P} + n1 < {N} else 0
to_kernel = data""")

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(f"_a[n0 * {P} + n1, k]", dynamic=True, allow_oob=True))
            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(f"A_pipe[{P} - n1 - 1]"))

        def make_read_B(state):

            # Also while reading B, we have to consider that T and P could not divide
            # M and N

            entry, exit = state.add_map("read_B", {
                "n": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "k": f"0:{K}",
                "m": f"0:{T}"
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            # If we are out-of bound, use a dummy value
            new_sdfg.add_array("B_dummy",
                               dtype=vec_type,
                               shape=[1],
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
            b_dummy = state.add_access("B_dummy")
            init_tasklet = state.add_tasklet("init_dummy_B", {}, {"init_data"}, "init_data = 0")

            state.add_memlet_path(init_tasklet, b_dummy, src_conn="init_data", memlet=dace.Memlet("B_dummy[0]"))

            mem = state.add_read("_b")
            pipe = state.add_write("B_pipe")
            tasklet = state.add_tasklet(
                "read_B", {"from_memory", "dummy_data"}, {"to_kernel"}, f"""\
data = from_memory if tm*{T} + m < {M} else dummy_data
to_kernel = data""")

            state.add_memlet_path(b_dummy, entry, tasklet, dst_conn="dummy_data", memlet=dace.Memlet("B_dummy[0]"))

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(f"_b[k, tm*{T} + m]", dynamic=True, allow_oob=True))

            state.add_memlet_path(tasklet, exit, pipe, src_conn="to_kernel", memlet=dace.Memlet("B_pipe[0]"))

        def make_write_C(state):

            # Receives the results and adds it to C

            pipe = state.add_read("C_pipe")
            if node.beta != 0:
                mem_read = state.add_read("_cin")
            mem = state.add_write("_c")

            entry_map, exit_map = state.add_map("write_C", {
                "n0": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "n1": f"0:{P}",
                "m": f"0:{T}"
            },
                                                schedule=dace.ScheduleType.FPGA_Device)

            # write in memory by adding C when we copy that to memory

            # deal with out-of-bound accesses

            mul_accumulated = f"{node.alpha} * from_kernel" if node.alpha != 1.0 else "from_kernel"
            if node.beta != 0:
                if node.beta != 1.0:
                    add_prev_c = f" + {node.beta} * prev_c"
                else:
                    add_prev_c = " + prev_c"
            else:
                add_prev_c = ""
            tasklet_inputs = {"from_kernel", "prev_c"} if node.beta != 0 else {"from_kernel"}
            tasklet = state.add_tasklet(
                "write_C", tasklet_inputs, {"to_memory"}, f"""\
if tm * {T} + m  < {M}  and  n0 * {P} + n1 < {N} :                                               
    to_memory = {mul_accumulated}{add_prev_c}
""")
            state.add_memlet_path(pipe,
                                  entry_map,
                                  tasklet,
                                  dst_conn="from_kernel",
                                  memlet=dace.Memlet(f"C_pipe[{P}-1]"))
            if node.beta != 0:
                state.add_memlet_path(mem_read,
                                      entry_map,
                                      tasklet,
                                      dst_conn="prev_c",
                                      memlet=dace.Memlet(f"_cin[n0 * {P} + n1, tm * {T} + m]",
                                                         dynamic=True,
                                                         allow_oob=True))

            state.add_memlet_path(tasklet,
                                  exit_map,
                                  mem,
                                  src_conn="to_memory",
                                  memlet=dace.Memlet(f"_c[n0 * {P} + n1, tm * {T} + m]", dynamic=True, allow_oob=True))

        def make_compute(sdfg, state):

            A_pipe_in = state.add_read("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            C_pipe_in = state.add_read("C_pipe")
            C_pipe_out = state.add_write("C_pipe")

            # The computation is expressed a single, flattened loop, which is generated by the following
            # pipeline scope. Each PE accumulates over T partial results. The drain phase last P*T clock cycles.
            # Draining and compute are overlapped.
            # We are generating the loop by explicitly ignoring loop carried dependencies. Therefore, we have
            # to guarantee that the PE will accumulate on the same partial result only when its value is consolidated.
            # The + L is a safe delay between accumulation between the same partial result.
            # It must be computed by considering T and the latency needed to consolidate a partial result
            # (which is the latency of the add + latency for reading and writing to BRAM).

            entry_pipeline, exit_pipeline = state.add_pipeline("compute_and_drain", {
                "n0": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "k": f"0:{K}",
                "m": f"0:{T} + {L}"
            },
                                                               drain_size=P * T,
                                                               drain_overlap=False,
                                                               additional_iterators={
                                                                   'm_drain': 0,
                                                                   'k_drain': 0
                                                               },
                                                               schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg", dtype=dtype_a, transient=True, storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")
            A_reg_init = state.add_access("A_reg")

            # For C result we are going to use vectorized data type

            # Note: for some of the Sacred Mysteries of Intel OpenCL Compiler (TM), if this buffer is smaller
            # than 24 floats, the II of the pipeline will be 5. Therefore we check this and in case we enlarge it
            buffer_size = T if T_constant is None else max(T_constant, 24)
            sdfg.add_array("C_buffer", [buffer_size],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            C_buffer_in = state.add_read("C_buffer")
            C_buffer_out = state.add_write("C_buffer")

            # Init data to reset partial results
            new_sdfg.add_array("C_init",
                               dtype=vec_type,
                               shape=[1],
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
            C_init = state.add_access("C_init")
            C_init_tasklet = state.add_tasklet("C_data_init", {}, {"init_data"}, "init_data = 0")

            state.add_memlet_path(C_init_tasklet, C_init, src_conn="init_data", memlet=dace.Memlet("C_init[0]"))
            state.add_memlet_path(entry_pipeline, C_init_tasklet, memlet=dace.Memlet())

            # Feed A
            # every PE: reads input data, buffer the data assigned to it
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {
                    "a_reg",
                }, f"""\
if m == 0 and not {entry_pipeline.pipeline.drain_condition()}:
    a_reg = a_in""")

            state.add_memlet_path(A_pipe_in,
                                  entry_pipeline,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]", dynamic=True),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")

            # Feed B
            sdfg.add_array("B_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            B_reg = state.add_access("B_reg")
            buffer_b_tasklet = state.add_tasklet(
                "buffer_b", {"b_in"}, {"b_reg_out"}, f"""\
if  m>={L} and not {entry_pipeline.pipeline.drain_condition()}:
    b_reg_out = b_in""")

            state.add_memlet_path(B_pipe_in,
                                  entry_pipeline,
                                  buffer_b_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]", dynamic=True),
                                  dst_conn="b_in")
            state.add_memlet_path(buffer_b_tasklet,
                                  B_reg,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=True),
                                  src_conn="b_reg_out")

            # Compute, Forward B, and Drain
            compute_tasklet = state.add_tasklet(
                "compute_and_drain", {"a_in", "b_in", "c_in", "forward_in", "c_init_data"},
                {"b_out", "c_out", "c_pipe_out"}, f"""\
result = c_in
if m >= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    c_prev = c_init_data if k == 0 else c_in
    result =  c_prev + a_in * b_in
    c_out = result
    if p < {P} - 1:
        b_out = b_in
# Drain
# when we have to drain:
# - if we are working on second assigned row or second tile and we have something to drain
# - if k = K-1 and m>=L: each PE has just finished to compute something
# - if we are in the draining phase
# How: 
# - if k = K-1 and m>=L: then the PE drains its own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((n0 > 0 or tm > 0)  and k_drain <p and m_drain <{T}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
    c_pipe_out = result if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
    """)

            state.add_memlet_path(A_reg, compute_tasklet, dst_conn="a_in", memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=False),
                                  dst_conn="b_in")
            state.add_memlet_path(C_init, compute_tasklet, memlet=dace.Memlet("C_init[0]"), dst_conn="c_init_data")

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]", dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(C_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="c_in",
                                  memlet=dace.Memlet(f"C_buffer[m-{L}]", allow_oob=True))

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_buffer_out,
                                  memlet=dace.Memlet(f"C_buffer[m-{L}]", allow_oob=True, dynamic=True),
                                  src_conn="c_out")

            state.add_memlet_path(C_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("C_pipe[p-1]", dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_pipe_out,
                                  memlet=dace.Memlet("C_pipe[p]", dynamic=True),
                                  src_conn="c_pipe_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map("unroll_compute", {"p": "0:{}".format(P)},
                                                        schedule=dace.ScheduleType.FPGA_Device,
                                                        unroll=True)

            # Bring data nodes into scope
            state.add_memlet_path(compute_entry, A_pipe_in, memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry, B_pipe_in, memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry, C_pipe_in, memlet=dace.memlet.Memlet())

            state.add_memlet_path(B_pipe_out, compute_exit, memlet=dace.memlet.Memlet())

            state.add_memlet_path(C_pipe_out, compute_exit, memlet=dace.memlet.Memlet())

            state.add_memlet_path(compute_entry, A_reg_init, memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_reg_init, entry_pipeline, memlet=dace.memlet.Memlet())
            b_init = state.add_access("B_reg")
            state.add_memlet_path(compute_entry, b_init, memlet=dace.Memlet())
            state.add_memlet_path(b_init, entry_pipeline, memlet=dace.Memlet())
            state.add_memlet_path(compute_entry, C_buffer_in, memlet=dace.Memlet())
            state.add_memlet_path(C_buffer_out, compute_exit, memlet=dace.Memlet())

        # build the compute State

        new_sdfg.add_stream("A_pipe",
                            dtype_a,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=1,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("C_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=T,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state)
        make_compute(new_sdfg, new_state)
        make_write_C(new_state)
        return new_sdfg


@dace.library.node
class Gemm(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ B) + beta * C. C should be unidirectionally
       broadcastable (ONNX terminology) to A @ B.
    """

    # Global properties
    implementations = {
        "pure": ExpandGemmPure,
        "MKL": ExpandGemmMKL,
        "OpenBLAS": ExpandGemmOpenBLAS,
        "cuBLAS": ExpandGemmCuBLAS,
        "TensorCore": ExpandGemmTensorCore3,
        "PBLAS": ExpandGemmPBLAS,
        "FPGA1DSystolic": ExpandGemmFPGA1DSystolic
    }
    default_implementation = None

    # Constants used in Tensor Core expansion
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WARP_SIZE = 32

    # Object fields
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")
    cin = properties.Property(dtype=bool, default=True, desc="Whether to have a _cin connector when beta != 0")
    algorithm = properties.Property(dtype=str,
                                    allow_none=True,
                                    default=None,
                                    desc="If applicable, chooses the vendor-provided implementation "
                                    "(algorithm) for the multiplication")
    accumulator_type = properties.TypeClassProperty(
        default=None,
        choices=dtypes.Typeclasses,
        allow_none=True,
        desc="Accumulator or intermediate storage type used in multiplication")
    compute_type = properties.Property(default=None,
                                       dtype=str,
                                       allow_none=True,
                                       desc="If applicable, overrides computation type (CUBLAS-specific, see "
                                       "``cublasComputeType_t``)")

    def __init__(self, name, location=None, transA=False, transB=False, alpha=1, beta=0, cin=True):
        super().__init__(name,
                         location=location,
                         inputs=({"_a", "_b", "_cin"} if beta != 0 and cin else {"_a", "_b"}),
                         outputs={"_c"})
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta
        self.cin = cin

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to gemm")
        size2 = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
            if dst_conn == '_c':
                subset = dc(memlet.subset)
                subset.squeeze()
                size2 = subset.size()

        if self.transA:
            size0 = list(reversed(size0))
        if self.transB:
            size1 = list(reversed(size1))

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if len(size0) != 2 or len(size1) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if size0[1] != size1[0]:
            raise ValueError("Inputs to matrix-matrix product " "must agree in the k-dimension")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size3 = out_subset.size()
        if size2 is not None and size2 != size3:
            raise ValueError("Input C matrix must match output matrix.")
        if len(size3) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if len(size3) == 2 and list(size3) != [size0[-2], size1[-1]]:
            raise ValueError("Output to matrix-matrix product must agree in the m and n " "dimensions")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemm')
@oprepo.replaces('dace.libraries.blas.Gemm')
def gemv_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 B,
                 C,
                 alpha,
                 beta,
                 trans_a=False,
                 trans_b=False):
    # Add nodes
    A_in, B_in = (state.add_read(name) for name in (A, B))
    C_out = state.add_write(C)

    libnode = Gemm('gemm', transA=trans_a, transB=trans_b, alpha=alpha, beta=beta)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_a', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_b', mm.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, mm.Memlet(C))

    if beta != 0:
        C_in = state.add_read(C)
        state.add_edge(C_in, None, libnode, '_cin', mm.Memlet(C))

    return []