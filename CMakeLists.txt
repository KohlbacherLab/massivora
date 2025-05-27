cmake_minimum_required(VERSION 3.4...3.18)
project(compute)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.4 NO_MODULE)
if(APPLE)
    # Find libomp installation
    set(LIBOMP_PATH "/opt/homebrew/opt/libomp" CACHE PATH "Path to libomp installation")
    
    # Add this to make OpenMP available at compile time
    include_directories(SYSTEM "${LIBOMP_PATH}/include")
    
    # Set OpenMP flags correctly for Apple Clang
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PATH}/lib/libomp.dylib")
else()
    find_package(OpenMP REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

pybind11_add_module(compute compute.cpp)
# Add OpenMP flags differently based on platform
if(APPLE)
    target_include_directories(compute PRIVATE "${LIBOMP_PATH}/include")
    target_compile_options(compute PRIVATE -Wall -Xpreprocessor -fopenmp -lomp)
    target_link_libraries(compute PRIVATE "${OpenMP_omp_LIBRARY}")
else()
    target_compile_options(compute PRIVATE -Wall ${OpenMP_CXX_FLAGS})
    target_link_libraries(compute PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

if (TARGET Eigen3::Eigen)
  target_link_libraries (compute PRIVATE Eigen3::Eigen)
endif (TARGET Eigen3::Eigen)