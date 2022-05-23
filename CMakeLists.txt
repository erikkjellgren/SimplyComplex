cmake_minimum_required(VERSION 2.6)
project(cuda_test LANGUAGES CUDA CXX)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(runTests.run tests.cu SimplyComplex.cu)
target_link_libraries(runTests.run ${GTEST_LIBRARIES} pthread)

add_executable(runTiming.run timing.cu SimplyComplex.cu)
target_link_libraries(runTiming.run pthread)
