cmake_minimum_required(VERSION 3.14)
project(massivora_memory_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find NLopt
find_package(NLopt REQUIRED)

# Find threads
find_package(Threads REQUIRED)

# Memory test executable
add_executable(test_memory test_memory.cpp)
target_include_directories(test_memory PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(test_memory PRIVATE Eigen3::Eigen NLopt::nlopt Threads::Threads)
target_compile_options(test_memory PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)

# Site optimizer executable (for startup overhead testing)
add_executable(site_optimizer site_optimizer.cpp)
target_include_directories(site_optimizer PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(site_optimizer PRIVATE Eigen3::Eigen NLopt::nlopt)
target_compile_options(site_optimizer PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)

# Site optimizer with shared memory support
add_executable(site_optimizer_shm site_optimizer_shm.cpp)
target_include_directories(site_optimizer_shm PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(site_optimizer_shm PRIVATE Eigen3::Eigen NLopt::nlopt rt)
target_compile_options(site_optimizer_shm PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)

# Blocked CPU plmDCA objective against the loop it replaced (bitwise)
add_executable(test_plm_objective test_plm_objective.cpp)
target_include_directories(test_plm_objective PRIVATE ${EIGEN3_INCLUDE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src/massivora/bindings_src)
target_link_libraries(test_plm_objective PRIVATE Eigen3::Eigen)
target_compile_options(test_plm_objective PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
