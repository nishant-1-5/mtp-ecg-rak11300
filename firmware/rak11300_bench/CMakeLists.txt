# RAK11300 benchmark for the Lead I pipeline (see firmware/rak11300_bench/README.md).
cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED ENV{PICO_SDK_PATH} AND NOT DEFINED PICO_SDK_PATH)
  message(FATAL_ERROR "Set PICO_SDK_PATH to a pico-sdk checkout")
endif()
if(NOT DEFINED PICO_SDK_PATH)
  set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
endif()

# The RAK11300 has an RP2040 with 2 MB of QSPI flash, like the Pico, so the
# generic Pico board definition is used. Only the flash chip and pins differ,
# and this benchmark uses neither.
set(PICO_BOARD pico CACHE STRING "Board type")
include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(rak11300_bench C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
pico_sdk_init()

set(FW ${CMAKE_CURRENT_LIST_DIR}/..)
add_executable(rak11300_bench
  main.cpp
  ${FW}/ecg_features/ecg_features.cpp
  ${FW}/ecg_pipeline/ecg_pipeline.cpp
)
target_compile_definitions(rak11300_bench PRIVATE ECG_PROFILE=1)
target_compile_options(rak11300_bench PRIVATE -Wall -Wextra -Wdouble-promotion -Wno-unused-parameter)
target_link_libraries(rak11300_bench pico_stdlib)
pico_enable_stdio_usb(rak11300_bench 1)
pico_enable_stdio_uart(rak11300_bench 1)
pico_add_extra_outputs(rak11300_bench)  # .uf2 for drag-and-drop flashing
