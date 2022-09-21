# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class TensorCore:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_tensor_core.h"], 'tensor_core': ["../include/dace_tensor_core.h"]}
    init_code = ""
    finalize_code = ""
    dependencies = []
    state_fields = []

    @staticmethod
    def handle_setup_code(node):
        location = node.location
        if not location or "gpu" not in node.location:
            location = 0
        else:
            try:
                location = int(location["gpu"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
const int __dace_cuda_device = {location};\n"""

        return code.format(location=location)
