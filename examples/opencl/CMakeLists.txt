set(opencl_test "opencl_test")
set(opencl_exe ${opencl_test})

set(${opencl_test}_private_srcs 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

set(opencl_test)

add_executable(${opencl_test} "")