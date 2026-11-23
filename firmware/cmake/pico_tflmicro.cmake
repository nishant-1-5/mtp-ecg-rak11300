# Defines the `pico-tflmicro` static library from a checkout of
# https://github.com/raspberrypi/pico-tflmicro (TFLM + CMSIS-NN for RP2040/RP2350).
#
# Upstream's CMakeLists.txt is a standalone project (it calls project() and
# pico_sdk_init()), so it cannot be add_subdirectory()'d. This file copies only
# its library definition (from "add_library(pico-tflmicro STATIC)" up to the
# test library) and includes it here, so the source list stays in sync with
# whatever upstream revision is checked out.
#
# Input: PICO_TFLMICRO_PATH (cache var or environment variable).
if(NOT PICO_TFLMICRO_PATH)
  set(PICO_TFLMICRO_PATH $ENV{PICO_TFLMICRO_PATH})
endif()
if(NOT EXISTS "${PICO_TFLMICRO_PATH}/CMakeLists.txt")
  message(FATAL_ERROR "Set PICO_TFLMICRO_PATH to a pico-tflmicro checkout")
endif()
file(TO_CMAKE_PATH "${PICO_TFLMICRO_PATH}" PICO_TFLMICRO_PATH)  # backslashes break the generated file

file(READ "${PICO_TFLMICRO_PATH}/CMakeLists.txt" _tflm_cmake)
string(FIND "${_tflm_cmake}" "add_library(pico-tflmicro STATIC)" _tflm_begin)
string(FIND "${_tflm_cmake}" "add_library(pico-tflmicro_test STATIC)" _tflm_end)
if(_tflm_begin EQUAL -1 OR _tflm_end EQUAL -1)
  message(FATAL_ERROR "pico-tflmicro CMakeLists.txt layout changed; update pico_tflmicro.cmake")
endif()
math(EXPR _tflm_len "${_tflm_end} - ${_tflm_begin}")
string(SUBSTRING "${_tflm_cmake}" ${_tflm_begin} ${_tflm_len} _tflm_lib)
string(REPLACE "\${CMAKE_CURRENT_LIST_DIR}" "${PICO_TFLMICRO_PATH}" _tflm_lib "${_tflm_lib}")

# pico-tflmicro hard-codes "#define TF_LITE_PICO_MULTICORE" (marked experimental
# upstream), which runs part of every CONV_2D on core 1. We keep core 1 free for
# ECG acquisition, and single-core is what the rp2040js emulator can run, so by
# default a copy of that one file without the define is compiled instead.
option(ECG_TFLM_MULTICORE "Let CMSIS-NN use core 1 for CONV_2D" OFF)
set(_mm "src/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c")
if(NOT ECG_TFLM_MULTICORE)
  file(READ "${PICO_TFLMICRO_PATH}/${_mm}" _mm_src)
  string(REPLACE "\n#define TF_LITE_PICO_MULTICORE" "\n// #define TF_LITE_PICO_MULTICORE  (disabled by pico_tflmicro.cmake)"
         _mm_src_patched "${_mm_src}")
  if(_mm_src_patched STREQUAL _mm_src)
    message(FATAL_ERROR "TF_LITE_PICO_MULTICORE define not found in ${_mm}")
  endif()
  file(WRITE "${CMAKE_BINARY_DIR}/tflm_patched/arm_nn_mat_mult_nt_t_s8.c" "${_mm_src_patched}")
  string(REPLACE "${PICO_TFLMICRO_PATH}/${_mm}" "${CMAKE_BINARY_DIR}/tflm_patched/arm_nn_mat_mult_nt_t_s8.c"
         _tflm_lib "${_tflm_lib}")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/pico_tflmicro_lib.cmake" "${_tflm_lib}")
include("${CMAKE_BINARY_DIR}/pico_tflmicro_lib.cmake")

# The vendored flatbuffers (and some TFLM headers) use uint32_t etc. without
# including <cstdint>; GCC >= 15 no longer pulls it in transitively. Force-include
# it rather than patching the checkout. PUBLIC, because users include these headers.
target_compile_options(pico-tflmicro PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-include cstdint>)

# Upstream sets COMPILE_FLAGS several times; only the last (-nostdlib) survives,
# so its intended -Os never applies and TFLM builds at the project's -O3.
# ECG_TFLM_OPT picks the level explicitly (measured trade-off in docs/phase2_tflm.md).
set(ECG_TFLM_OPT "O3" CACHE STRING "Optimisation level for TFLM/CMSIS-NN (O3 or Os)")
target_compile_options(pico-tflmicro PRIVATE -${ECG_TFLM_OPT})
