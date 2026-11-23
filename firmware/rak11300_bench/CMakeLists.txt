# RAK11300 firmware targets (see firmware/rak11300_bench/README.md):
#   rak11300_bench     feature extraction only, per-stage timing
#   rak11300_pipeline  full pipeline incl. the INT8 CNN under TensorFlow Lite Micro
cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED ENV{PICO_SDK_PATH} AND NOT DEFINED PICO_SDK_PATH)
  message(FATAL_ERROR "Set PICO_SDK_PATH to a pico-sdk checkout")
endif()
if(NOT DEFINED PICO_SDK_PATH)
  set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
endif()

# The RAK11300 has an RP2040 with 2 MB of QSPI flash, like the Pico, so the
# generic Pico board definition is used. Only the flash chip and pins differ,
# and these targets use neither.
set(PICO_BOARD pico CACHE STRING "Board type")
include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(rak11300_bench C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)  # TFLM needs C++17; the ECG code is C++14-clean
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
pico_sdk_init()

set(FW ${CMAKE_CURRENT_LIST_DIR}/..)
set(ECG_SOURCES
  ${FW}/ecg_features/ecg_features.cpp
  ${FW}/ecg_pipeline/ecg_pipeline.cpp
)
set(ECG_WARNINGS -Wall -Wextra -Wdouble-promotion -Wno-unused-parameter)

# --- feature-stage benchmark --------------------------------------------------
add_executable(rak11300_bench main.cpp ${ECG_SOURCES})
target_compile_definitions(rak11300_bench PRIVATE ECG_PROFILE=1)
target_compile_options(rak11300_bench PRIVATE ${ECG_WARNINGS})
target_link_libraries(rak11300_bench pico_stdlib)
pico_enable_stdio_usb(rak11300_bench 1)
pico_enable_stdio_uart(rak11300_bench 1)
pico_add_extra_outputs(rak11300_bench)

# --- full pipeline with TFLM --------------------------------------------------
include(${FW}/cmake/pico_tflmicro.cmake)
add_executable(rak11300_pipeline pipeline_main.cpp ${ECG_SOURCES} ${FW}/ecg_cnn/ecg_cnn.cpp)
set_source_files_properties(pipeline_main.cpp ${ECG_SOURCES} PROPERTIES COMPILE_OPTIONS "${ECG_WARNINGS}")
target_link_libraries(rak11300_pipeline pico_stdlib pico-tflmicro)
pico_enable_stdio_usb(rak11300_pipeline 1)
pico_enable_stdio_uart(rak11300_pipeline 1)
pico_add_extra_outputs(rak11300_pipeline)

# --- same pipeline on N random fold-10 windows (emulator validation) -----------
# Needs build/bench_vectors_validation.h: python scripts/make_bench_vectors.py --validation 60
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/bench_vectors_validation.h)
  add_executable(rak11300_validation pipeline_main.cpp ${ECG_SOURCES} ${FW}/ecg_cnn/ecg_cnn.cpp)
  target_compile_definitions(rak11300_validation PRIVATE BENCH_VALIDATION=1)
  target_include_directories(rak11300_validation PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(rak11300_validation pico_stdlib pico-tflmicro)
  pico_enable_stdio_uart(rak11300_validation 1)
  pico_add_extra_outputs(rak11300_validation)
endif()
