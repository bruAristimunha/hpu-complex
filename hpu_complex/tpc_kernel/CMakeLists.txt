cmake_minimum_required(VERSION 3.16)
project(fft_tpc_kernels CXX C)

set(TPC_COMPILER_PATH "/usr/bin/tpc-clang" CACHE STRING "TPC compiler")
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE STRING "TPC includes")

add_library(fft_tpc_perf_lib SHARED src/entry_points.cpp src/fft_glue.cpp)
target_include_directories(fft_tpc_perf_lib PRIVATE src ${TPC_INCLUDE_DIR})

foreach(KERNEL fft_probe_f32_gaudi2 fft_r2c_f32_gaudi2)
    add_custom_command(
        OUTPUT ${KERNEL}_x86.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gaudi2/${KERNEL}.c
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gaudi2/${KERNEL}.c ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=gaudi2 -O2 -I${TPC_INCLUDE_DIR}/ ./${KERNEL}.c -c
        COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${KERNEL}.o ${KERNEL}_x86.o)
    add_library(${KERNEL} STATIC ${KERNEL}_x86.o)
    set_target_properties(${KERNEL} PROPERTIES LINKER_LANGUAGE C)
    target_link_libraries(fft_tpc_perf_lib ${KERNEL})
endforeach()
