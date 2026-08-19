# GaussBLAS.cmake -- resolve an ILP64 OpenBLAS into the `gauss::blas` target.
#
# GaussDCA inverts a (20N x 20N) covariance with ?potrf/?potri. With a 32-bit
# LAPACK integer that is capped at n = 46340, i.e. about 2317 alignment
# columns. Past the cap the inverse falls back to a serial Eigen path; measured
# on a 28-core Skylake, going from 2310 to 2320 columns took the run from
# 46.6 s / 4.4 GiB to 2583 s / 16.4 GiB -- 55x slower for 0.4% more work. So
# ILP64 is required, not merely preferred.
#
# OpenBLAS is the only supported BLAS. It measured fastest for this workload on
# Intel -- 1.9x MKL at N=2320, because MKL's ?potri runs at less than half the
# rate of its own ?potrf -- and it is one package on every platform.
#
# Set GAUSS_BLAS_LIBRARY to link something else; whatever it names still has to
# pass the ILP64 self-check below.

include_guard(GLOBAL)

# A semicolon-separated list, so multi-library BLASes can be named too, e.g.
#   -DGAUSS_BLAS_LIBRARY="/opt/aocl/lib_ILP64/libflame.so;/opt/aocl/lib_ILP64/libblis-mt.so"
set(GAUSS_BLAS_LIBRARY "" CACHE STRING
    "ILP64 BLAS/LAPACK to link instead of the auto-detected OpenBLAS")

# ---------------------------------------------------------------------------
# Where to look, beyond the system paths.
#
# find_library only searches system paths by default, but every documented
# install path puts the BLAS somewhere else:
#   conda / micromamba   $CONDA_PREFIX     (set by `conda activate`)
#   conda-build          $PREFIX / $BUILD_PREFIX
#   pip install .        the target interpreter's prefix -- pip's build
#                        isolation may not export CONDA_PREFIX at all, so
#                        derive it from Python_EXECUTABLE too.
# ---------------------------------------------------------------------------
set(_gauss_hints "")
foreach(_v CONDA_PREFIX PREFIX BUILD_PREFIX OPENBLAS_ROOT)
  if(DEFINED ENV{${_v}})
    list(APPEND _gauss_hints "$ENV{${_v}}")
  endif()
endforeach()
if(Python_EXECUTABLE)
  get_filename_component(_gauss_py_bin "${Python_EXECUTABLE}" DIRECTORY)
  get_filename_component(_gauss_py_prefix "${_gauss_py_bin}" DIRECTORY)
  list(APPEND _gauss_hints "${_gauss_py_prefix}")
endif()
if(_gauss_hints)
  list(REMOVE_DUPLICATES _gauss_hints)
  list(PREPEND CMAKE_PREFIX_PATH ${_gauss_hints})
endif()

# Support libraries the BLAS may lean on but does not always pull in itself.
set(_gauss_support "")
if(UNIX)
  find_package(Threads QUIET)
  if(TARGET Threads::Threads)
    list(APPEND _gauss_support Threads::Threads)
  endif()
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(TARGET OpenMP::OpenMP_CXX)
    # OpenBLAS ships both pthreads and OpenMP builds; the latter needs this.
    list(APPEND _gauss_support OpenMP::OpenMP_CXX)
  endif()
  list(APPEND _gauss_support m)
endif()

# ---------------------------------------------------------------------------
# Configure-time self-check.
#
# Compiles and *runs* a program against the candidate library that
#   1. factorises a 2x2 SPD matrix and checks the factor, proving the symbol
#      links and the call convention is right, and
#   2. poisons the high word of `info`, asks for the factorisation of a
#      non-positive-definite 1x1 (which sets info = 1), and requires the whole
#      64-bit value to read back as 1. A 32-bit library writes only the low
#      word, so the poison survives.
#
# Step 2 is the one that matters: ILP64 and LP64 builds often export the same
# symbol names, so a plain link test would accept an LP64 library and silently
# reinstate the 2317-column cliff.
#
# It doubles as the symbol-suffix probe. OpenBLAS built with SYMBOLSUFFIX
# (conda-forge, Debian) renames its Fortran symbols to <name>_64_, while one
# built with just INTERFACE64=1 keeps the plain names and only widens the
# integers -- so the spelling cannot be assumed, it has to be measured.
#
# Sets, in the caller's scope:
#   <ok_var>      TRUE when some suffix passed
#   <suffix_var>  the Fortran symbol suffix ("_64_" or "_")
#   <detail_var>  the symbol that worked, or why nothing did
# ---------------------------------------------------------------------------
function(_gauss_blas_selfcheck libs ok_var suffix_var detail_var)
  # .cpp, not .c: the project only enables CXX, and try_run needs an enabled
  # language.
  set(_src "${CMAKE_CURRENT_BINARY_DIR}/gauss_blas_ilp64_check.cpp")
  file(WRITE "${_src}" [=[
#include <cstdint>
#include <cstdio>

extern "C" void GAUSS_POTRF(const char *uplo, const std::int64_t *n, double *a,
                            const std::int64_t *lda, std::int64_t *info);

int main() {
    /* (1) a 2x2 SPD factorisation must succeed and produce the right factor.
       Column-major lower triangle of [[4,2],[2,10]] -> L = [[2,0],[1,3]]. */
    double a[4] = {4.0, 2.0, 0.0, 10.0};
    std::int64_t n = 2, lda = 2, info = 0;
    GAUSS_POTRF("L", &n, a, &lda, &info);
    if (info != 0) { std::printf("potrf info=%lld\n", (long long)info); return 1; }
    if (a[0] != 2.0 || a[1] != 1.0 || a[3] != 3.0) {
        std::printf("wrong factor: %f %f %f\n", a[0], a[1], a[3]);
        return 2;
    }

    /* (2) prove the integer is 64 bits. Poison the high word of `info`, then
       factorise a 1x1 that is not positive definite so LAPACK sets info = 1.
       An LP64 library writes 4 bytes and the poison survives. */
    double b[1] = {-1.0};
    std::int64_t n1 = 1, lda1 = 1;
    info = static_cast<std::int64_t>(0x7FFFFFFF) << 32;
    GAUSS_POTRF("L", &n1, b, &lda1, &info);
    if (info != 1) {
        std::printf("integer width mismatch: info=0x%llx (expected 1)\n",
                    (unsigned long long)info);
        return 3;
    }
    std::printf("ok\n");
    return 0;
}
]=])

  # The try_run binary has to find the shared library at run time.
  set(_rpath "")
  foreach(_l IN LISTS libs)
    if(IS_ABSOLUTE "${_l}" AND EXISTS "${_l}")
      get_filename_component(_d "${_l}" DIRECTORY)
      list(APPEND _rpath "${_d}")
    endif()
  endforeach()
  set(_link_flags "")
  if(_rpath)
    list(REMOVE_DUPLICATES _rpath)
    string(REPLACE ";" ":" _rpath_str "${_rpath}")
    set(_link_flags "-Wl,-rpath,${_rpath_str}")
  endif()

  foreach(_suffix "_64_" "_")
    set(_sym "dpotrf${_suffix}")
    try_run(_run_rc _build_ok
            "${CMAKE_CURRENT_BINARY_DIR}/gauss_blas_check"
            "${_src}"
            LINK_LIBRARIES ${libs} ${_gauss_support}
            CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS=${_link_flags}"
            COMPILE_DEFINITIONS "-DGAUSS_POTRF=${_sym}"
            COMPILE_OUTPUT_VARIABLE _cout
            RUN_OUTPUT_VARIABLE _rout)
    if(_build_ok AND _run_rc EQUAL 0)
      set(${ok_var} TRUE PARENT_SCOPE)
      set(${suffix_var} "${_suffix}" PARENT_SCOPE)
      set(${detail_var} "${_sym}" PARENT_SCOPE)
      return()
    endif()
    if(_build_ok AND NOT _run_rc EQUAL 0)
      # It linked but misbehaved -- almost always an LP64 library. This is the
      # message that explains an otherwise baffling failure.
      set(_why "${_sym} linked but the check failed (rc=${_run_rc}): ${_rout}")
    elseif(NOT _build_ok AND NOT _link_why)
      # An unresolved dependency looks exactly like a missing library unless
      # the linker error is carried out to the user.
      string(REGEX MATCHALL "[^\n]*(undefined reference|cannot find|No such file)[^\n]*"
             _errs "${_cout}")
      if(_errs)
        list(GET _errs 0 _first)
        string(STRIP "${_first}" _first)
        set(_link_why "${_sym} would not link: ${_first}")
      endif()
    endif()
  endforeach()

  set(${ok_var} FALSE PARENT_SCOPE)
  set(${suffix_var} "" PARENT_SCOPE)
  if(_why)
    set(${detail_var} "${_why}" PARENT_SCOPE)
  elseif(_link_why)
    set(${detail_var} "${_link_why}" PARENT_SCOPE)
  else()
    set(${detail_var} "no ILP64 ?potrf symbol could be linked" PARENT_SCOPE)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Find the library, verify it, publish gauss::blas.
# ---------------------------------------------------------------------------
if(GAUSS_BLAS_LIBRARY)
  set(_gauss_libs "${GAUSS_BLAS_LIBRARY}")
  set(_gauss_how "GAUSS_BLAS_LIBRARY=${GAUSS_BLAS_LIBRARY}")
else()
  # conda-forge and Debian install the ILP64 build as libopenblas64_, but
  # FindBLAS only looks for `openblas64` / `openblas_64` / `openblas`. Without
  # this the search falls through to the LP64 libopenblas.so that usually sits
  # in the very same prefix.
  find_library(GAUSS_OPENBLAS_ILP64
    NAMES openblas64_ openblas_ilp64 openblas64 openblas_64
    HINTS ${_gauss_hints}
    PATH_SUFFIXES lib lib64)
  if(GAUSS_OPENBLAS_ILP64)
    set(_gauss_libs "${GAUSS_OPENBLAS_ILP64}")
    set(_gauss_how "${GAUSS_OPENBLAS_ILP64}")
  else()
    # Generic fallback, for builds that do use the plain library name.
    set(BLA_VENDOR "OpenBLAS")
    set(BLA_SIZEOF_INTEGER 8)
    find_package(LAPACK QUIET)
    unset(BLA_VENDOR)
    unset(BLA_SIZEOF_INTEGER)
    if(LAPACK_FOUND)
      set(_gauss_libs ${LAPACK_LIBRARIES})
      set(_gauss_how "FindLAPACK BLA_SIZEOF_INTEGER=8 (${LAPACK_LIBRARIES})")
    endif()
  endif()
endif()

set(GAUSS_BLAS_SYMBOL_SUFFIX "")
if(_gauss_libs)
  _gauss_blas_selfcheck("${_gauss_libs}" _gauss_ok GAUSS_BLAS_SYMBOL_SUFFIX _gauss_detail)
endif()

if(NOT _gauss_libs OR NOT _gauss_ok)
  if(_gauss_libs)
    set(_gauss_problem "  Found ${_gauss_how}, but ${_gauss_detail}.\n")
  else()
    set(_gauss_problem "  No ILP64 OpenBLAS was found.\n")
  endif()
  string(REPLACE ";" "\n          " _gauss_searched "${_gauss_hints}")
  message(FATAL_ERROR
    "massivora: no usable ILP64 BLAS/LAPACK.\n"
    "\n"
    "${_gauss_problem}"
    "\n"
    "  GaussDCA inverts a (20N x 20N) covariance with ?potrf/?potri. A 32-bit\n"
    "  LAPACK integer caps that at 2317 alignment columns, so the ILP64\n"
    "  interface is required -- the ordinary LP64 openblas will not do.\n"
    "\n"
    "  Install it and reinstall massivora:\n"
    "      conda install -c conda-forge openblas-ilp64\n"
    "      sudo apt install libopenblas64-openmp-dev       # Debian/Ubuntu\n"
    "\n"
    "  Prefixes searched, in addition to the system paths:\n"
    "          ${_gauss_searched}\n"
    "\n"
    "  To link a different ILP64 BLAS:\n"
    "      -DGAUSS_BLAS_LIBRARY=/path/to/libyourblas.so\n")
endif()

message(STATUS "massivora: ILP64 OpenBLAS at ${_gauss_how}")
message(STATUS "massivora: ILP64 self-check passed, Fortran symbols are ${_gauss_detail}")

add_library(gauss::blas INTERFACE IMPORTED)
target_link_libraries(gauss::blas INTERFACE ${_gauss_libs} ${_gauss_support})
target_compile_definitions(gauss::blas INTERFACE GAUSS_BLAS_ILP64)
if(GAUSS_BLAS_SYMBOL_SUFFIX STREQUAL "_64_")
  # SYMBOLSUFFIX build; the headers paste this onto every LAPACK symbol.
  target_compile_definitions(gauss::blas INTERFACE GAUSS_BLAS_SUFFIX_64)
endif()
