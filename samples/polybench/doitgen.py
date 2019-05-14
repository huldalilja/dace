import math
import dace
import polybench

NQ = dace.symbol('NQ')
NR = dace.symbol('NR')
NP = dace.symbol('NP')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    NQ: 8,
    NR: 10,
    NP: 12
}, {
    NQ: 20,
    NR: 25,
    NP: 30
}, {
    NQ: 40,
    NR: 50,
    NP: 60
}, {
    NQ: 140,
    NR: 150,
    NP: 160
}, {
    NQ: 220,
    NR: 250,
    NP: 270
}]

args = [dace.ndarray([NR, NQ, NP], datatype), dace.ndarray([NP, NP], datatype)]


def init_array(A, C4):
    nr = NR.get()
    nq = NQ.get()
    np = NP.get()

    for i in range(nr):
        for j in range(nq):
            for k in range(np):
                A[i, j, k] = datatype((i * j + k) % np) / np
    for i in range(np):
        for j in range(np):
            C4[i, j] = datatype((i * j) % np) / np


@dace.program(datatype[NR, NQ, NP], datatype[NP, NP])
def doitgen(A, C4):
    sum = dace.define_local([NR, NQ, NP], dtype=datatype)

    @dace.mapscope
    def doit(r: _[0:NR], q: _[0:NQ]):
        # nA << A[r, q, :]
        # nC4 << C4[:, :]
        # nSum << sum[r, q, :]
        # nAout >> A[r, q, :]

        @dace.map
        def compute_sum(p: _[0:NP], s: _[0:NP]):
            inA << A[r, q, s]
            inC4 << C4[s, p]
            s >> sum(1, lambda a, b: a + b, 0)[r, q, p]
            s = inA * inC4

        @dace.map
        def compute_A(p: _[0:NP]):
            insum << sum[r, q, p]
            out >> A[r, q, p]
            out = insum

        # @dace.program(datatype[NP], datatype[NP, NP], datatype[NP],
        #               datatype[NP])
        # def internal(nA, nC4, nSum, nAout):
        #     @dace.map
        #     def compute_sum(p: _[0:NP], s: _[0:NP]):
        #         inA << nA[s]
        #         inC4 << nC4[s, p]
        #         s >> nSum(1, lambda a, b: a + b, 0)[p]
        #         s = inA * inC4

        #     @dace.map
        #     def compute_A(p: _[0:NP]):
        #         insum << nSum[p]
        #         out >> nAout[p]
        #         out = insum


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, doitgen)
