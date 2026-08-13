cmake_minimum_required(VERSION 3.18)

# GPU support is optional and auto-detected. ENABLE_CUDA accepts:
#   AUTO (default) - build the CUDA backend iff a CUDA compiler is found
#   ON             - require CUDA (configure fails if no CUDA compiler)
#   OFF            - force a CPU-only build
# So `pip install .[cuda12]` enables the GPU backend automatically on a CUDA
# host with no extra flags; override with `-C cmake.define.ENABLE_CUDA=ON|OFF`.
set(ENABLE_CUDA "AUTO" CACHE STRING "Enable CUDA support: ON, OFF, or AUTO (detect)")

project(massivora_native LANGUAGES CXX)

# Resolve AUTO -> ON/OFF by probing for a usable CUDA compiler.
if(ENABLE_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "massivora: CUDA compiler found (${CMAKE_CUDA_COMPILER}); "
                       "enabling GPU backend. Use -DENABLE_CUDA=OFF to disable.")
        set(ENABLE_CUDA ON)
    else()
        message(STATUS "massivora: no CUDA compiler found; building CPU-only. "
                       "Use -DENABLE_CUDA=ON to force.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Configure CUDA language + toolkit when enabled.
if(ENABLE_CUDA)
    # Default target GPU architectures (override with
    #   -C cmake.define.CMAKE_CUDA_ARCHITECTURES="80;90"  via pip).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;90")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    if(TARGET CUDA::cublas)
      set(MASSIVORA_CUBLAS_TARGET CUDA::cublas)
    elseif(TARGET CUDAToolkit::cublas)
      set(MASSIVORA_CUBLAS_TARGET CUDAToolkit::cublas)
    else()
      set(MASSIVORA_CUBLAS_TARGET cublas)
      if(CUDAToolkit_LIBRARY_DIR)
        list(APPEND MASSIVORA_CUDA_LIB_DIRS "${CUDAToolkit_LIBRARY_DIR}")
      endif()
      if(CUDAToolkit_LIBRARY_DIRS)
        list(APPEND MASSIVORA_CUDA_LIB_DIRS ${CUDAToolkit_LIBRARY_DIRS})
      endif()
    endif()
endif()

# --- Python + pybind11 ---
# scikit-build-core provides the Python hints. Development.Module is the
# correct component for building extension modules (works in manylinux too).
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# pybind11 is supplied as a build dependency (see pyproject build-system),
# so its CMake config is on CMAKE_PREFIX_PATH.
find_package(pybind11 CONFIG REQUIRED)

# --- Threads ---
# The GaussDCA core (gauss_dca.hpp) uses std::thread, so the standalone
# gauss_infer executable needs a threading library linked.
find_package(Threads REQUIRED)

# --- Eigen ---
# Eigen is header-only. Prefer its CMake config package, then pkg-config, then a
# plain header search; if it still isn't found, fetch the headers from source
# (controlled by MASSIVORA_FETCH_EIGEN). Always end up with the Eigen3::Eigen
# target so the rest of this file is uniform.
option(MASSIVORA_FETCH_EIGEN
  "Download Eigen from source if it cannot be found on the system" ON)
set(EIGEN_GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
    CACHE STRING "Git URL for Eigen (used only when fetching from source)")
set(EIGEN_GIT_TAG "3.4.0"
    CACHE STRING "Eigen git tag/commit to fetch when not found")

find_package(Eigen3 NO_MODULE QUIET HINTS "$ENV{CONDA_PREFIX}")
if(NOT TARGET Eigen3::Eigen)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(EIGEN3 QUIET eigen3)
  endif()
  if(EIGEN3_INCLUDE_DIRS AND NOT EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIRS}")
  endif()
  if(NOT EIGEN3_INCLUDE_DIR)
    find_path(EIGEN3_INCLUDE_DIR
      NAMES signature_of_eigen3_matrix_library Eigen/Dense
      PATH_SUFFIXES eigen3 include/eigen3 include
      HINTS "$ENV{EIGEN3_ROOT}" "$ENV{EIGEN_ROOT}" "$ENV{CONDA_PREFIX}"
    )
  endif()
  if(NOT EIGEN3_INCLUDE_DIR AND MASSIVORA_FETCH_EIGEN)
    message(STATUS "massivora: Eigen not found; fetching ${EIGEN_GIT_TAG} from source.")
    include(FetchContent)
    # Eigen is header-only: SOURCE_SUBDIR points at a missing dir so
    # MakeAvailable only downloads and does NOT add_subdirectory() (which would
    # pull in Eigen's tests/docs/install machinery). We just need the headers.
    FetchContent_Declare(eigen
      GIT_REPOSITORY "${EIGEN_GIT_REPOSITORY}"
      GIT_TAG "${EIGEN_GIT_TAG}"
      GIT_SHALLOW TRUE
      SOURCE_SUBDIR "massivora-no-add-subdirectory"
    )
    FetchContent_MakeAvailable(eigen)
    set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
  endif()
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR
      "Eigen3 not found. Install it (conda: `eigen`, apt: `libeigen3-dev`, "
      "brew: `eigen`), pass -DEIGEN3_INCLUDE_DIR=/path/to/eigen, or allow "
      "fetching from source with -DMASSIVORA_FETCH_EIGEN=ON.")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
  message(STATUS "massivora: using Eigen headers at ${EIGEN3_INCLUDE_DIR}")
else()
  message(STATUS "massivora: found Eigen3 config package (Eigen3::Eigen)")
endif()

# --- NLopt ---
# Prefer NLopt's CMake config package, then pkg-config; if it still isn't found,
# build it from source (controlled by MASSIVORA_FETCH_NLOPT). Unlike Eigen,
# NLopt is a compiled library, so the source fallback actually builds a static,
# PIC libnlopt in a private prefix (mirroring the cuLBFGSB ExternalProject) and
# imports it. Always end up with the NLopt::nlopt target.
option(MASSIVORA_FETCH_NLOPT
  "Build NLopt from source if it cannot be found on the system" ON)
set(NLOPT_GIT_REPOSITORY "https://github.com/stevengj/nlopt.git"
    CACHE STRING "Git URL for NLopt (used only when building from source)")
set(NLOPT_GIT_TAG "v2.8.0"
    CACHE STRING "NLopt git tag/commit to build when not found")

find_package(NLopt CONFIG QUIET HINTS "$ENV{CONDA_PREFIX}")

# pkg-config: create an imported target and alias it to NLopt::nlopt.
if(NOT TARGET NLopt::nlopt)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(NLOPT QUIET IMPORTED_TARGET nlopt)
    if(TARGET PkgConfig::NLOPT)
      add_library(NLopt::nlopt ALIAS PkgConfig::NLOPT)
      message(STATUS "massivora: using system NLopt (pkg-config)")
    endif()
  endif()
endif()

# Source fallback: build a minimal static, PIC NLopt into a private prefix.
if(NOT TARGET NLopt::nlopt AND MASSIVORA_FETCH_NLOPT)
  message(STATUS "massivora: NLopt not found; building ${NLOPT_GIT_TAG} from source.")
  include(FetchContent)
  include(ExternalProject)

  # Download at configure time; SOURCE_SUBDIR points at a missing dir so we do
  # NOT add_subdirectory() it (which would inject NLopt's own install rules into
  # our wheel). We drive its build via ExternalProject below.
  FetchContent_Declare(nlopt_src
    GIT_REPOSITORY "${NLOPT_GIT_REPOSITORY}"
    GIT_TAG "${NLOPT_GIT_TAG}"
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR "massivora-no-add-subdirectory"
  )
  FetchContent_MakeAvailable(nlopt_src)

  set(NLOPT_INSTALL_DIR "${nlopt_src_BINARY_DIR}/install")
  set(NLOPT_INCLUDE_DIR "${NLOPT_INSTALL_DIR}/include")
  set(NLOPT_STATIC_LIB
      "${NLOPT_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}nlopt${CMAKE_STATIC_LIBRARY_SUFFIX}")
  # The include dir must exist at configure time for INTERFACE_INCLUDE_DIRECTORIES.
  file(MAKE_DIRECTORY "${NLOPT_INCLUDE_DIR}")

  ExternalProject_Add(nlopt_ext
    SOURCE_DIR "${nlopt_src_SOURCE_DIR}"
    BINARY_DIR "${nlopt_src_BINARY_DIR}/build"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${NLOPT_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR=lib            # pin libdir so the .a path is predictable
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      # Let CMake >= 3.31 / 4.x configure NLopt even though its tag may declare an
      # old cmake_minimum_required (ignored as an unknown cache var on older CMake).
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
      -DBUILD_SHARED_LIBS=OFF
      -DNLOPT_PYTHON=OFF
      -DNLOPT_OCTAVE=OFF
      -DNLOPT_MATLAB=OFF
      -DNLOPT_GUILE=OFF
      -DNLOPT_SWIG=OFF
      -DNLOPT_TESTS=OFF
    BUILD_BYPRODUCTS "${NLOPT_STATIC_LIB}"
  )

  add_library(NLopt::nlopt STATIC IMPORTED)
  set_target_properties(NLopt::nlopt PROPERTIES
    IMPORTED_LOCATION "${NLOPT_STATIC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIR}")
  # Consumers depend on the build via add_dependencies(... nlopt_ext) below.
endif()

if(NOT TARGET NLopt::nlopt)
  message(FATAL_ERROR
    "NLopt not found. Install it (conda: `nlopt`, apt: `libnlopt-cxx-dev`, "
    "brew: `nlopt`), or allow building from source with -DMASSIVORA_FETCH_NLOPT=ON.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

set(MASSIVORA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/massivora")
set(BINDINGS_SRC "${MASSIVORA_SRC}/bindings_src")

# Install destinations (relative to the wheel platlib root, i.e. the package).
set(MASSIVORA_BINDINGS_INSTALL_DIR "massivora/cpp_bindings")
set(MASSIVORA_BIN_INSTALL_DIR "massivora/bin")

# --- Performance tuning: target ISA, precision, and a threaded LAPACK -------
# gauss_infer/cpp_bindings spend nearly all of their time in dense linear
# algebra. Three build-time choices dominate speed and memory:
#
#   MASSIVORA_ARCH              the ISA the compiler may emit. A plain -O3 build
#                               targets baseline x86-64 (SSE2, no FMA), which
#                               costs ~3.5x on the Cholesky/inverse. "native"
#                               tunes for the build machine; use e.g.
#                               x86-64-v3 for a redistributable build, or OFF.
#   MASSIVORA_USE_LAPACK        ?potrf + ?potri for the inverse, internally
#                               threaded. Without it a correct but
#                               single-threaded Eigen fallback is used, which
#                               is ~50x slower on a wide alignment.
#   MASSIVORA_DOUBLE_PRECISION  factorise in float64 instead of the float32
#                               default, doubling the memory a pair costs.
include(CheckCXXCompilerFlag)

set(MASSIVORA_ARCH "native" CACHE STRING
    "ISA to target for -march (native, x86-64-v3, ...); OFF for the compiler default")

set(MASSIVORA_ARCH_FLAGS "")
if(NOT MASSIVORA_ARCH STREQUAL "OFF")
  check_cxx_compiler_flag("-march=${MASSIVORA_ARCH}" MASSIVORA_HAS_ARCH_FLAG)
  if(MASSIVORA_HAS_ARCH_FLAG)
    set(MASSIVORA_ARCH_FLAGS "-march=${MASSIVORA_ARCH}")
    message(STATUS "massivora: building with -march=${MASSIVORA_ARCH}")
  else()
    message(WARNING "massivora: compiler rejected -march=${MASSIVORA_ARCH}; "
                    "using the default ISA (expect gauss_infer to be ~3.5x slower)")
  endif()
endif()

option(MASSIVORA_DOUBLE_PRECISION
       "Factorise the GaussDCA covariance in float64 instead of float32" OFF)
set(MASSIVORA_PRECISION_DEFS "")
if(MASSIVORA_DOUBLE_PRECISION)
  set(MASSIVORA_PRECISION_DEFS GAUSSDCA_DOUBLE_PRECISION)
  message(STATUS "massivora: GaussDCA covariance in float64 (2x the memory of the default)")
else()
  message(STATUS "massivora: GaussDCA covariance in float32 (default)")
endif()

option(MASSIVORA_USE_LAPACK "Use a threaded LAPACK for the GaussDCA inverse" ON)
set(MASSIVORA_LAPACK_LIBS "")
set(MASSIVORA_LAPACK_DEFS "")
if(MASSIVORA_USE_LAPACK)
  # find_package(LAPACK) resolves through find_library, which searches system
  # paths only. Every documented install path puts the BLAS somewhere else:
  #
  #   conda / micromamba    $CONDA_PREFIX          (set by `conda activate`)
  #   conda-build           $PREFIX / $BUILD_PREFIX
  #   pip install .         the target interpreter's prefix -- and pip's build
  #                         isolation means CONDA_PREFIX may not be exported at
  #                         all, so derive it from Python_EXECUTABLE too.
  #
  # Without these hints the env's openblas is missed, GAUSSDCA_USE_LAPACK is
  # never defined, and the inverse silently drops to the single-threaded Eigen
  # fallback.
  set(_massivora_blas_hints "")
  foreach(_v CONDA_PREFIX PREFIX BUILD_PREFIX)
    if(DEFINED ENV{${_v}})
      list(APPEND _massivora_blas_hints "$ENV{${_v}}")
    endif()
  endforeach()
  if(Python_EXECUTABLE)
    get_filename_component(_massivora_py_bin "${Python_EXECUTABLE}" DIRECTORY)
    get_filename_component(_massivora_py_prefix "${_massivora_py_bin}" DIRECTORY)
    list(APPEND _massivora_blas_hints "${_massivora_py_prefix}")
  endif()
  if(_massivora_blas_hints)
    list(REMOVE_DUPLICATES _massivora_blas_hints)
    list(PREPEND CMAKE_PREFIX_PATH ${_massivora_blas_hints})
  endif()

  find_package(LAPACK QUIET)
  if(LAPACK_FOUND)
    set(MASSIVORA_LAPACK_LIBS ${LAPACK_LIBRARIES})
    set(MASSIVORA_LAPACK_DEFS GAUSSDCA_USE_LAPACK)
    message(STATUS "massivora: GaussDCA inverse via LAPACK ${LAPACK_LIBRARIES}")
  else()
    # Hard error rather than a warning: `pip install .` hides build warnings, so
    # a degraded binary would install cleanly and simply be slow, which is a
    # miserable thing to debug from the far side.
    string(REPLACE ";" "\n      " _massivora_searched "${_massivora_blas_hints}")
    message(FATAL_ERROR
      "massivora: no BLAS/LAPACK found.\n"
      "\n"
      "  gauss_infer inverts the GaussDCA covariance with ?potrf/?potri. Without\n"
      "  a LAPACK it falls back to a single-threaded Eigen path that is roughly\n"
      "  50x slower on wide alignments and needs ~2.5x the memory.\n"
      "\n"
      "  Install one and reinstall:\n"
      "      conda install -c conda-forge liblapack\n"
      "      apt install libopenblas-dev          # Debian/Ubuntu\n"
      "      dnf install openblas-devel           # Fedora/RHEL\n"
      "\n"
      "  Prefixes searched, in addition to the system paths:\n"
      "      ${_massivora_searched}\n"
      "\n"
      "  To build the slow fallback deliberately:\n"
      "      cmake -DMASSIVORA_USE_LAPACK=OFF ...\n"
      "      pip install . -C cmake.define.MASSIVORA_USE_LAPACK=OFF\n")
  endif()
else()
  message(WARNING
    "massivora: MASSIVORA_USE_LAPACK=OFF -- gauss_infer will use the "
    "single-threaded Eigen inverse (~50x slower on wide alignments). "
    "Check `gauss_infer --info` reports lapack=0 as intended.")
endif()

# Apply the tuning above to a target.
function(massivora_tune tgt)
  if(MASSIVORA_ARCH_FLAGS)
    target_compile_options(${tgt} PRIVATE ${MASSIVORA_ARCH_FLAGS})
  endif()
  if(MASSIVORA_PRECISION_DEFS)
    target_compile_definitions(${tgt} PRIVATE ${MASSIVORA_PRECISION_DEFS})
  endif()
  if(MASSIVORA_LAPACK_DEFS)
    target_compile_definitions(${tgt} PRIVATE ${MASSIVORA_LAPACK_DEFS})
    target_link_libraries(${tgt} PRIVATE ${MASSIVORA_LAPACK_LIBS})
  endif()
endfunction()

# --- pybind11 extension module ---
if(ENABLE_CUDA)
    pybind11_add_module(cpp_bindings ${BINDINGS_SRC}/bindings.cpp)
    target_compile_definitions(cpp_bindings PRIVATE ENABLE_CUDA)
    target_include_directories(cpp_bindings PRIVATE
        ${BINDINGS_SRC}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    if(MASSIVORA_CUDA_LIB_DIRS)
      target_link_directories(cpp_bindings PRIVATE ${MASSIVORA_CUDA_LIB_DIRS})
    endif()
    target_link_libraries(cpp_bindings PRIVATE cuda_bindings ${MASSIVORA_CUBLAS_TARGET})
else()
    pybind11_add_module(cpp_bindings ${BINDINGS_SRC}/bindings.cpp)
endif()

# --- Standalone executable plm_opt_site ---
add_executable(plm_opt_site ${BINDINGS_SRC}/plm_opt_site.cpp)
target_link_libraries(plm_opt_site PRIVATE Eigen3::Eigen NLopt::nlopt)
massivora_tune(plm_opt_site)

# --- Standalone executable gauss_infer (GaussDCA couplings over shared memory) ---
add_executable(gauss_infer ${BINDINGS_SRC}/gauss_infer.cpp)
target_include_directories(gauss_infer PRIVATE ${BINDINGS_SRC})
target_link_libraries(gauss_infer PRIVATE Eigen3::Eigen Threads::Threads)
massivora_tune(gauss_infer)

# --- CUDA backend library (only if CUDA is enabled) ---
if(ENABLE_CUDA)
    add_library(cuda_bindings SHARED ${BINDINGS_SRC}/bindings.cu)
    target_include_directories(cuda_bindings PRIVATE
        ${BINDINGS_SRC}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    if(MASSIVORA_CUDA_LIB_DIRS)
      target_link_directories(cuda_bindings PRIVATE ${MASSIVORA_CUDA_LIB_DIRS})
    endif()
    target_link_libraries(cuda_bindings PRIVATE ${MASSIVORA_CUBLAS_TARGET})
    # RDC (for cooperative-groups grid.sync in the L-BFGS two-loop)
    set_property(TARGET cuda_bindings PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET cuda_bindings PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Build the standalone plm_opt_site executable against the CUDA backend.
    target_compile_definitions(plm_opt_site PRIVATE ENABLE_CUDA)
    target_include_directories(plm_opt_site PRIVATE
        ${BINDINGS_SRC}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    if(MASSIVORA_CUDA_LIB_DIRS)
      target_link_directories(plm_opt_site PRIVATE ${MASSIVORA_CUDA_LIB_DIRS})
    endif()
    target_link_libraries(plm_opt_site PRIVATE
        cuda_bindings
        ${MASSIVORA_CUBLAS_TARGET}
        CUDA::cudart)
endif()

# --- Link Eigen + NLopt into the bindings module ---
target_link_libraries(cpp_bindings PRIVATE Eigen3::Eigen NLopt::nlopt)
massivora_tune(cpp_bindings)

# If NLopt is built from source, make its consumers wait for that build.
if(TARGET nlopt_ext)
  add_dependencies(plm_opt_site nlopt_ext)
  add_dependencies(cpp_bindings nlopt_ext)
endif()

# When CUDA libs live next to the module, make them discoverable at runtime.
if(ENABLE_CUDA AND NOT APPLE)
  set_target_properties(cpp_bindings PROPERTIES INSTALL_RPATH "$ORIGIN")
  set_target_properties(plm_opt_site PROPERTIES INSTALL_RPATH "$ORIGIN/../cpp_bindings")
endif()

# --- Install rules (populate the wheel / install tree) ---
install(TARGETS cpp_bindings
    LIBRARY DESTINATION ${MASSIVORA_BINDINGS_INSTALL_DIR}
    RUNTIME DESTINATION ${MASSIVORA_BINDINGS_INSTALL_DIR})

install(TARGETS plm_opt_site
    RUNTIME DESTINATION ${MASSIVORA_BIN_INSTALL_DIR})

install(TARGETS gauss_infer
    RUNTIME DESTINATION ${MASSIVORA_BIN_INSTALL_DIR})

# Python package shim + the runtime CUDA kernel source used by CuPy at runtime.
install(FILES
    ${BINDINGS_SRC}/__init__.py
    ${BINDINGS_SRC}/cuda_kernels.cuh
    DESTINATION ${MASSIVORA_BINDINGS_INSTALL_DIR})

if(ENABLE_CUDA)
    install(TARGETS cuda_bindings
        LIBRARY DESTINATION ${MASSIVORA_BINDINGS_INSTALL_DIR})
endif()
