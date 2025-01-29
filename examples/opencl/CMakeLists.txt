set(OPENCL_TEST "opencl_test")
set(OPENCL_EXE ${OPENCL_TEST})

set(    ${OPENCL_TEST}_PRIVATE_SRCS 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp
            )

set(    ${OPENCL_TEST}_PUBLIC_HDRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include/opencl_jet/graph.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/opencl_jet/graph_node.h
 )

add_executable(${OPENCL_TEST} "")
target_sources(${OPENCL_TEST} PRIVATE 
                                ${${OPENCL_TEST}_PRIVATE_SRCS} 
                                ${${OPENCL_TEST}_PUBLIC_HDRS})
target_link_libraries(${OPENCL_TEST} PRIVATE
                                absl::log_initialize
                                absl::flags
                                absl::flags_parse 
                                Ceres::ceres 
                                fmt::fmt 
                                ceres_internal)
target_include_directories(${OPENCL_TEST} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)