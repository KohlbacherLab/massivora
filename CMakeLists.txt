cmake_minimum_required(VERSION 3.4...3.18)

# Add option to enable CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Configure project languages based on CUDA flag
if(ENABLE_CUDA)
    project(compute LANGUAGES CXX CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
    project(compute LANGUAGES CXX)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_dir}")
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.0 NO_MODULE REQUIRED)
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

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(compute src/massivora/compute.cpp)

# CUDA library (only if CUDA is enabled)
if(ENABLE_CUDA)
    add_library(cuda_compute SHARED cuda_kernels.cu)
    if (TARGET Eigen3::Eigen)
      target_link_libraries(cuda_compute PRIVATE Eigen3::Eigen)
    endif (TARGET Eigen3::Eigen)
    target_link_libraries(cuda_compute PRIVATE ${CUDA_LIBRARIES})
    set_property(TARGET cuda_compute PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    # Link CUDA library to main compute module
    target_link_libraries(compute PRIVATE cuda_compute)
endif()
# Configure main compute module
if (TARGET Eigen3::Eigen)
  target_link_libraries(compute PRIVATE Eigen3::Eigen)
endif (TARGET Eigen3::Eigen)

# Add OpenMP flags differently based on platform
if(APPLE)
    target_include_directories(compute PRIVATE "${LIBOMP_PATH}/include")
    target_compile_options(compute PRIVATE -O3 -Wall -Xpreprocessor -fopenmp -lomp)
    target_link_libraries(compute PRIVATE "${OpenMP_omp_LIBRARY}")
else()
    target_compile_options(compute PRIVATE -O3 -Wall ${OpenMP_CXX_FLAGS})
    target_link_libraries(compute PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()