# build private test framework
set(CMAKE_CXX_STANDARD 14)
set(SWHOME /opt/sw_home)
set(INC_PATHES ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${SWHOME}/local/cuda/include ${SWHOME}}/local/include
                ${TARGET_DIR}/include ${TARGET_DIR}/include/eigen3 ${SWHOME}/ixinfer/include/
                ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)

set(LINK_PATHES ${CMAKE_CURRENT_BINARY_DIR}/../lib ${CUINFER_LIBRARY}
                ${TARGET_DIR}/lib64  ${SWHOME}/ixinfer/build/lib ${SWHOME}}/local/lib64 ${SWHOME}/local/cuda/lib64
                ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
                ${CMAKE_CURRENT_SOURCE_DIR}/googletest/build/lib)

add_executable(test_qdconv test_cuinferQDConvolutionForward.cpp)
message("INC_PATHES" ${INC_PATHES})
target_include_directories(test_qdconv PRIVATE ${INC_PATHES})
target_link_directories(test_qdconv PRIVATE ${LINK_PATHES} -lpthread -lm)

target_link_libraries(
  test_qdconv
  PRIVATE cuinfer cuda cudart cublas curand
          ${CUDA_DRIVER_LIBRARY}
          ${CUDART_LIBRARY}
          ${IXTHUNK_LIBRARY}
          ${IXLOGGER_LIBRARY}
          gtest
          pthread
          )

add_executable(test_qdeconv test_cuinferQDEConvolutionForward.cpp)
message("INC_PATHES" ${INC_PATHES})
target_include_directories(test_qdeconv PRIVATE ${INC_PATHES})
target_link_directories(test_qdeconv PRIVATE ${LINK_PATHES} -lpthread -lm)

target_link_libraries(
  test_qdeconv
  PRIVATE cuinfer cuda cudart cublas curand
          ${CUDA_DRIVER_LIBRARY}
          ${CUDART_LIBRARY}
          ${IXTHUNK_LIBRARY}
          ${IXLOGGER_LIBRARY}
          gtest
          pthread
          )

# add_executable(compare_cuinfer compare_cuinferConvlutionForwad.cpp)
# message("INC_PATHES" ${INC_PATHES})
#target_include_directories(compare_cuinfer PRIVATE ${INC_PATHES})
#target_link_directories(compare_cuinfer PRIVATE ${LINK_PATHES} -lpthread -lm)

#target_link_libraries(
 # compare_cuinfer
  #PRIVATE cuinfer cuda cudart cublas curand
   #       ${CUDA_DRIVER_LIBRARY}
    #      ${CUDART_LIBRARY}
     #     ${IXTHUNK_LIBRARY}
      #    ${IXLOGGER_LIBRARY}
       #   gtest
        #  pthread
         # )