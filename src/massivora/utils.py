import importlib
import logging
import os

# Prefix applied to every POSIX shared-memory segment massivora allocates, so
# cleanup can recognise its own segments and never touch unrelated ones. Kept
# in this dependency-light module so every Python consumer shares one
# definition. The C++ optimizer applies the same prefix in
# bindings_src/plm_opt_site.cpp and must be kept in sync with this value.
SHM_PREFIX = "Massivora_"


def massivora_pkg_dir():
    spec = importlib.util.find_spec('massivora')
    if spec and spec.submodule_search_locations:
        return os.fspath(list(spec.submodule_search_locations)[0])

def worker_id():
    slurm_id = f"{os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID') or ''}".strip()
    return slurm_id or str(os.getpid())

def compute_id_range(portion, portion_start, total):
    if portion < 0:
        portion = 0.0
    if portion_start < 0:
        portion_start = 0.0
    if portion_start > 1:
        portion_start = 1.0

    portion_end = portion_start + portion
    if portion_end > 1:
        portion_end = 1.0

    starting_job_index = int(total * portion_start) + 1
    ending_job_index = int(total * portion_end) + 1

    if starting_job_index < 1:
        starting_job_index = 1
    if ending_job_index < starting_job_index:
        ending_job_index = starting_job_index
    if total > 0:
        ending_job_index = min(ending_job_index, total + 1)

    return starting_job_index, ending_job_index

def setup_logging(cfg):
    log_cfg = cfg.get('logging', {})
    if log_cfg and log_cfg.get('enabled'):
        log_file = log_cfg.get('file') or 'errors.log'
        level_name = str(log_cfg.get('level') or 'WARNING').upper()
        try:
            level = getattr(logging, level_name, logging.WARNING)
        except AttributeError:
            level = logging.WARNING
        log_dir = log_cfg.get('dir')
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)

        # Force reconfiguration so we can reliably control when logging is set up
        logging.basicConfig(
            filename=log_file,
            level=level,
            format='%(asctime)s %(levelname)s - From %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True,
        )

# CUDA helpers
_CUDA_KERNELS = {}
_DEFAULT_KERNEL_PATH = os.path.join(
    massivora_pkg_dir(), 'cpp_bindings', 'cuda_kernels.cuh')
_GPU_AVAILABLE = None


def gpu_is_available():
    """Return True if CuPy and CUDA devices are available."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    if importlib.util.find_spec('cupy') is None:
        _GPU_AVAILABLE = False
        return _GPU_AVAILABLE

    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        _GPU_AVAILABLE = False
    else:
        _GPU_AVAILABLE = True

    return _GPU_AVAILABLE


def get_cuda_module(kernel_source_path=None, device_id=None):
    if not gpu_is_available():
        raise RuntimeError(
            "CUDA is not available; install CuPy and ensure a CUDA-capable GPU is visible."
        )

    import cupy as cp

    if kernel_source_path is None:
        kernel_source_path = _DEFAULT_KERNEL_PATH
    if device_id is None:
        device_id = cp.cuda.Device().id

    cache_key = (device_id, kernel_source_path)

    if cache_key not in _CUDA_KERNELS:
        with open(kernel_source_path, 'r', encoding="utf-8") as f:
            code = f.read()
        _CUDA_KERNELS[cache_key] = cp.RawModule(code=code)

    return _CUDA_KERNELS[cache_key]

