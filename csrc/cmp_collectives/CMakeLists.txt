cmake_minimum_required(VERSION 3.26)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(vllm_cmp_collectives LANGUAGES CUDA)
endif()

# A C ABI loaded through ctypes; independent of the Python and Torch ABIs.
add_library(_cmp_pair_tree MODULE pair_tree.cu)
set_target_properties(_cmp_pair_tree PROPERTIES
  PREFIX "" SUFFIX ".abi3.so"
  CUDA_ARCHITECTURES 80
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)
target_compile_options(_cmp_pair_tree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
install(TARGETS _cmp_pair_tree LIBRARY DESTINATION vllm COMPONENT _cmp_pair_tree)
