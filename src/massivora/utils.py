import fcntl
import importlib
import logging
import mmap
import os
import struct
import subprocess
import zlib
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

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

# ----------- msv shards -----------
#
# Coupling scores are written into a few large shard files instead of one
# directory tree per pair. A pair's matrix is symmetric, so only the upper
# triangle (diagonal included) is stored, uncompressed: the scores are
# high-entropy, and zstd buys under 10% on them while costing decode time on
# every read.
#
# Each record opens with the bytes 4D 53 56 FF: "MSV" followed by 0xFF. 0xFF
# never occurs in UTF-8, so the marker cannot turn up inside a pair name.
# Records are self-describing, so a shard whose database index is lost, or one
# damaged in the middle, can still be recovered by scanning for that marker --
# see `scan_shard()`.
#
#   offset  size  field
#   0       4     magic b"MSV\xff"
#   4       2     format version
#   6       2     payload dtype code
#   8       4     n, the matrix width
#   12      4     name length
#   16      4     bitmask length
#   20      8     payload length
#   28      4     crc32 of the payload
#   32      ..    name (utf-8), saved-column bitmask, then the payload

MSV_MAGIC = b"MSV\xff"
MSV_VERSION = 1
MSV_DTYPE_FLOAT16 = 0
MSV_EXT = "msv"
MSV_HEADER = struct.Struct("<4sHHIIIQI")
MSV_HEADER_SIZE = MSV_HEADER.size
# A shard is closed and a new one started once it would grow past this.
MSV_SHARD_MAX_BYTES = 100 * 1024 ** 3
# Shard numbers are three digits, so a run spans at most this many shards.
MSV_MAX_SHARDS = 1000
MSV_SCAN_MIN_REGION_BYTES = 1024 ** 3
#: How far back from a shard's end to look for the record that ends it
MSV_TAIL_SEARCH_BYTES = 256 * 1024 ** 2


def shard_name(prefix, file_id):
    """
    Build the file name of one shard.

    Parameters
    ----------
    `prefix` — str
        Shard name prefix, normally the configured couplings table name
    `file_id` — int
        Shard number, rendered as three digits

    Returns
    -------
    `str`
        File name in the form `prefix_000.msv`
    """
    file_id = int(file_id)
    if not 0 <= file_id < MSV_MAX_SHARDS:
        raise ValueError(f"Shard number {file_id} does not fit in three digits")
    return f"{prefix}_{file_id:03d}.{MSV_EXT}"


def shard_path(output_dir, prefix, file_id):
    """
    Full path of one shard.

    Parameters
    ----------
    `output_dir` — str
        Directory holding the shards
    `prefix` — str
        Shard name prefix
    `file_id` — int
        Shard number

    Returns
    -------
    `str`
        Absolute or relative path to the shard file
    """
    return os.path.join(output_dir, shard_name(prefix, file_id))


def saved_columns_to_bitmask(saved_columns, lengthes):
    """
    Pack per-monomer retained-column lists into a bitmask blob.

    Parameters
    ----------
    `saved_columns` — list of list of int
        Retained column indices, per monomer, 0-based and local to the monomer
    `lengthes` — list of int
        Full column count of each monomer, in the same order

    Returns
    -------
    `bytes`
        The bitmask blob stored in an msv record
    """
    parts = [list(cols) for cols in (saved_columns or [])]
    lens = [int(length) for length in (lengthes or [])]
    if len(parts) != len(lens):
        raise ValueError(
            f"saved_columns describes {len(parts)} monomers but lengthes has {len(lens)}")

    out = bytearray(struct.pack("<H", len(parts)))
    for length in lens:
        if length < 0:
            raise ValueError(f"Monomer length {length} is negative")
        out += struct.pack("<I", length)
    for cols, length in zip(parts, lens):
        mask = bytearray((length + 7) // 8)
        for col in cols:
            col = int(col)
            if not 0 <= col < length:
                raise ValueError(
                    f"saved_columns holds column {col} for a monomer of length {length}")
            mask[col >> 3] |= 1 << (col & 7)
        out += mask
    return bytes(out)


def bitmask_to_saved_columns(blob):
    """
    Unpack a bitmask blob back into retained-column lists.

    Parameters
    ----------
    `blob` — bytes
        Blob produced by `saved_columns_to_bitmask()`

    Returns
    -------
    `tuple`
        `(saved_columns, lengthes)`, matching the inputs that were packed
    """
    blob = bytes(blob or b"")
    if len(blob) < 2:
        return [], []
    (n_parts,) = struct.unpack_from("<H", blob, 0)
    pos = 2
    lens = []
    for _ in range(n_parts):
        (length,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        lens.append(int(length))

    saved = []
    for length in lens:
        n_bytes = (length + 7) // 8
        chunk = blob[pos:pos + n_bytes]
        if len(chunk) != n_bytes:
            raise ValueError("Bitmask blob is truncated")
        pos += n_bytes
        bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8), bitorder='little')
        saved.append(np.flatnonzero(bits[:length]).tolist())
    return saved, lens


def upper_triangle_bytes(matrix):
    """
    Serialise the upper triangle of a symmetric coupling matrix.

    Parameters
    ----------
    `matrix` — numpy.ndarray
        Square coupling matrix

    Returns
    -------
    `bytes`
        Uncompressed float16 upper triangle, diagonal included, row-major
    """
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Coupling matrix must be square, got shape {m.shape}")
    rows, cols = np.triu_indices(m.shape[0])
    return np.ascontiguousarray(m[rows, cols], dtype=np.float16).tobytes()


def matrix_from_upper_triangle(payload, n):
    """
    Rebuild a symmetric coupling matrix from its stored upper triangle.

    Parameters
    ----------
    `payload` — bytes
        Upper triangle as written by `upper_triangle_bytes()`
    `n` — int
        Matrix width

    Returns
    -------
    `numpy.ndarray`
        The full `(n, n)` float16 matrix
    """
    n = int(n)
    tri = np.frombuffer(payload, dtype=np.float16)
    expected = n * (n + 1) // 2
    if tri.size != expected:
        raise ValueError(
            f"Upper triangle holds {tri.size} values but a {n}x{n} matrix needs {expected}")
    m = np.zeros((n, n), dtype=np.float16)
    rows, cols = np.triu_indices(n)
    m[rows, cols] = tri
    return m + np.triu(m, 1).T


def pack_score_record(name, matrix, bitmask=b""):
    """
    Serialise one pair into a shard record.

    Parameters
    ----------
    `name` — str
        Pair name, embedded in the record so a scan can identify it
    `matrix` — numpy.ndarray
        Square coupling matrix
    `bitmask` — bytes (optional)
        Saved-column bitmask from `saved_columns_to_bitmask()` (default: `b""`)

    Returns
    -------
    `bytes`
        The complete record, ready to append to a shard
    """
    payload = upper_triangle_bytes(matrix)
    name_bytes = str(name).encode('utf-8')
    bitmask = bytes(bitmask or b"")
    header = MSV_HEADER.pack(
        MSV_MAGIC, MSV_VERSION, MSV_DTYPE_FLOAT16, int(np.asarray(matrix).shape[0]),
        len(name_bytes), len(bitmask), len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    return header + name_bytes + bitmask + payload


def unpack_score_record(buf, verify=True):
    """
    Parse one shard record.

    Parameters
    ----------
    `buf` — bytes
        Exactly one record, as located by `(offset, len)` or by `scan_shard()`
    `verify` — bool (optional)
        Check the payload against its stored crc32 (default: `True`)

    Returns
    -------
    `dict`
        Keys `name`, `n`, `bitmask`, `saved_columns`, `lengthes`, `matrix`

    Raises
    ------
    `ValueError`
        If the record is corrupt, truncated, or has an unsupported version or dtype
    """
    buf = memoryview(buf)
    if len(buf) < MSV_HEADER_SIZE:
        raise ValueError("Record is shorter than an msv header")
    magic, version, dtype, n, name_len, mask_len, payload_len, crc = \
        MSV_HEADER.unpack_from(buf, 0)
    if magic != MSV_MAGIC:
        raise ValueError(f"Record does not start with {MSV_MAGIC!r}")
    if version != MSV_VERSION:
        raise ValueError(f"Unsupported msv record version {version}")
    if dtype != MSV_DTYPE_FLOAT16:
        raise ValueError(f"Unsupported msv payload dtype {dtype}")

    pos = MSV_HEADER_SIZE
    name = bytes(buf[pos:pos + name_len]).decode('utf-8')
    pos += name_len
    bitmask = bytes(buf[pos:pos + mask_len])
    pos += mask_len
    payload = bytes(buf[pos:pos + payload_len])
    if len(payload) != payload_len:
        raise ValueError(f"Record for {name} is truncated")
    if verify and (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ValueError(f"Checksum mismatch for {name}")

    saved_columns, lengthes = bitmask_to_saved_columns(bitmask)
    return {
        'name': name,
        'n': int(n),
        'bitmask': bitmask,
        'saved_columns': saved_columns,
        'lengthes': lengthes,
        'matrix': matrix_from_upper_triangle(payload, n),
    }


def _ends_on_a_record(fd, size):
    """
    Whether a shard's last bytes are a complete record whose payload checks out.

    Searches back from the end for the marker of the record that ends the file.
    A writer that closed cleanly leaves exactly that, so continuing its shard
    costs one record's read rather than a walk over every header.
    """
    lo = max(0, size - MSV_TAIL_SEARCH_BYTES)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    try:
        hi = size
        while True:
            at = mm.rfind(MSV_MAGIC, lo, hi)
            if at < 0:
                return False
            try:
                _, name_len, mask_len, payload_len, crc = _read_record_header(mm, at)
            except ValueError:
                hi = at + len(MSV_MAGIC) - 1        # a stray marker; look further back
                continue
            payload_at = at + MSV_HEADER_SIZE + name_len + mask_len
            return (payload_at + payload_len == size
                    and zlib.crc32(mm[payload_at:size]) & 0xFFFFFFFF == crc)
    finally:
        mm.close()


def _clean_end(fd, path):
    """
    Where a shard's last complete record ends.

    That is its size, unless a writer died part way through a record. Only then
    are all its headers walked, to find the last record that is whole; nothing
    past it can have been handed out as a location.
    """
    size = os.fstat(fd).st_size
    if size == 0 or _ends_on_a_record(fd, size):
        return size
    logging.info(f"Shard {os.path.basename(path)} does not end on a complete record; "
                 "reading its record headers to find where they end")
    return max((offset + length for offset, length, _, _ in scan_shard_headers(path)), default=0)


class ShardWriter(object):
    """
    Append coupling records to `prefix_NNN.msv` shards.

    A shard has one writer at a time, which holds an exclusive lock on it for as
    long as it has it open: the same kind of advisory file lock SQLite needs to
    share a job database between nodes. The lock goes when the writer closes the
    shard or its process dies, so the next writer continues the lowest-numbered
    shard nobody holds that its first record fits in, appending from where the
    shard's last complete record ends. Only when every shard is held or full is a
    new one started, at the lowest number nobody has created yet.

    Records are appended and the shard rolled over once it would pass
    `max_bytes`, so a shard never has to be sized in advance.
    """

    def __init__(self, output_dir, prefix, max_bytes=MSV_SHARD_MAX_BYTES):
        """
        Parameters
        ----------
        `output_dir` — str
            Directory to write shards into; created if absent
        `prefix` — str
            Shard name prefix, normally the configured couplings table name
        `max_bytes` — int (optional)
            Roll over once a shard would exceed this (default: 100 GiB)
        """
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = int(max_bytes)
        os.makedirs(output_dir, exist_ok=True)
        self._fd = None
        self._file_id = None
        self._offset = 0
        self._locking = True

    @property
    def file_id(self):
        """Number of the shard currently held open, or `None` before the first append."""
        return self._file_id

    def _claim_shard(self, need):
        """
        Hold the lowest-numbered shard nobody holds that a record of `need` bytes
        fits in, or else start the lowest shard number nobody has created yet.
        """
        if self._locking:
            for file_id in self._existing_shards():
                if self._hold(file_id, need, create=False):
                    return
        for file_id in range(MSV_MAX_SHARDS):
            if self._hold(file_id, need, create=True):
                return
        raise RuntimeError(
            f"All {MSV_MAX_SHARDS} shard numbers for prefix {self.prefix!r} are taken")

    def _existing_shards(self):
        """Numbers of this prefix's shards in the output directory, lowest first."""
        head, tail = f"{self.prefix}_", f".{MSV_EXT}"
        try:
            entries = os.listdir(self.output_dir)
        except OSError:
            return []
        stems = (entry[len(head):-len(tail)] for entry in entries
                 if entry.startswith(head) and entry.endswith(tail))
        return sorted(int(stem) for stem in stems if len(stem) == 3 and stem.isdigit())

    def _hold(self, file_id, need, create):
        """
        Try to become the writer of shard `file_id` for a record of `need` bytes.

        With `create`, only a shard that does not exist yet is taken. Returns
        whether this writer now holds the shard.
        """
        path = shard_path(self.output_dir, self.prefix, file_id)
        try:
            if create:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
            else:
                # A shard is never shorter than where its records end, so one this
                # rules out is not worth opening.
                size = os.path.getsize(path)
                if size and size + need > self.max_bytes:
                    return False
                fd = os.open(path, os.O_RDWR)
        except (FileExistsError, FileNotFoundError):
            return False

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False                    # another writer holds it
        except OSError as e:
            # Without file locks no shard can be shared safely, so keep to new
            # shards, which the exclusive create makes this writer's own.
            if self._locking:
                logging.warning(f"File locks are unavailable in {self.output_dir} ({e}); "
                                "every writer will start a shard of its own")
                self._locking = False
            if not create:
                os.close(fd)
                return False

        try:
            end = _clean_end(fd, path)
            if end and end + need > self.max_bytes:
                os.close(fd)                # leave it for a record that fits
                return False
            if os.fstat(fd).st_size > end:
                # A writer died part way through a record. No location past `end`
                # was handed out, and a header walk could not see past half a record.
                os.ftruncate(fd, end)
        except BaseException:
            os.close(fd)
            raise
        self._fd, self._file_id, self._offset = fd, file_id, end
        if end:
            logging.info(f"Continuing coupling scores in shard {shard_name(self.prefix, file_id)} "
                         f"from byte {end}")
        else:
            logging.info(f"Writing coupling scores to shard {shard_name(self.prefix, file_id)}")
        return True

    def append(self, name, matrix, bitmask=b""):
        """
        Write one pair's scores and return where they landed.

        Parameters
        ----------
        `name` — str
            Pair name
        `matrix` — numpy.ndarray
            Square coupling matrix
        `bitmask` — bytes (optional)
            Saved-column bitmask (default: `b""`)

        Returns
        -------
        `tuple`
            `(score_file_id, offset, len)` to record against the pair in the database
        """
        record = pack_score_record(name, matrix, bitmask)
        if self._fd is None:
            self._claim_shard(len(record))
        elif self._offset and self._offset + len(record) > self.max_bytes:
            self.close()
            self._claim_shard(len(record))

        offset = self._offset
        written = os.pwrite(self._fd, record, offset)
        if written != len(record):
            raise IOError(
                f"Short write for {name}: {written} of {len(record)} bytes")
        self._offset += written
        return self._file_id, offset, written

    def flush(self):
        """Push the shard to disk. Call before committing rows that point into it."""
        if self._fd is not None:
            os.fsync(self._fd)

    def close(self):
        """Flush and release the shard currently held open, leaving it for the next writer."""
        if self._fd is not None:
            if os.fstat(self._fd).st_size > self._offset:
                # A write that failed part way. Trimming it leaves the shard ending on
                # a complete record, which the next writer confirms with one read.
                os.ftruncate(self._fd, self._offset)
            os.fsync(self._fd)
            os.close(self._fd)              # which also gives up the lock
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def read_score_record(output_dir, prefix, file_id, offset, length, verify=True):
    """
    Read back one pair using the location recorded in the database.

    Parameters
    ----------
    `output_dir` — str
        Directory holding the shards
    `prefix` — str
        Shard name prefix
    `file_id` — int
        `score_file_id` from the couplings row
    `offset` — int
        `score_offset` from the couplings row
    `length` — int
        `score_len` from the couplings row
    `verify` — bool (optional)
        Check the payload checksum (default: `True`)

    Returns
    -------
    `dict`
        As returned by `unpack_score_record()`
    """
    path = shard_path(output_dir, prefix, file_id)
    with open(path, 'rb') as f:
        f.seek(int(offset))
        buf = f.read(int(length))
    if len(buf) != int(length):
        raise ValueError(
            f"Shard {path} holds {len(buf)} bytes at offset {offset}, expected {length}")
    return unpack_score_record(buf, verify=verify)


def scan_shard(path, verify=True):
    """
    Recover records from a shard without consulting the database.

    Walks the file looking for the start marker and yields every record that
    parses and passes its checksum, skipping damaged stretches. Use this when
    the index is lost or part of a shard is corrupt.

    Parameters
    ----------
    `path` — str
        Path to a shard file
    `verify` — bool (optional)
        Skip records failing their checksum (default: `True`)

    Raises
    ------
    `tuple`
        `(offset, len, record)` for each intact record, where `record` is the
        `unpack_score_record()` dict: `name`, `n`, `bitmask`, `saved_columns`,
        `lengthes` and the decoded `matrix`
    """
    size = os.path.getsize(path)
    if size < MSV_HEADER_SIZE:
        return
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            pos = 0
            while True:
                found = mm.find(MSV_MAGIC, pos)
                if found < 0:
                    return
                try:
                    _, _, _, _, name_len, mask_len, payload_len, _ = \
                        MSV_HEADER.unpack_from(mm, found)
                    total = MSV_HEADER_SIZE + name_len + mask_len + payload_len
                    if found + total > size:
                        raise ValueError("record runs past the end of the shard")
                    record = unpack_score_record(mm[found:found + total], verify=verify)
                except Exception as e:
                    logging.debug(f"Skipping bad record at offset {found} of {path}: {e}")
                    pos = found + len(MSV_MAGIC)
                    continue
                yield found, total, record
                pos = found + total
        finally:
            mm.close()


def _usable_cpu_count():
    """
    Number of CPU cores this process may run on.

    Honours the affinity mask SLURM and cgroups set on Linux, so a job granted
    8 cores on a 128-core node is counted as 8, not 128.

    Returns
    -------
    `int`
        Usable cores, at least 1
    """
    if hasattr(os, 'process_cpu_count'):
        return os.process_cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _read_record_header(buf, offset=0):
    """
    Parse and check a record's fixed header without touching its body.

    The start marker is not unique. It cannot occur inside a pair name, but any
    four bytes of a bitmask, payload or checksum can match it by chance. Every
    header field is therefore checked from the 32 header bytes alone, so a stray
    marker is turned away before anything reads the body it appears to describe.

    Parameters
    ----------
    `buf` — bytes, memoryview or mmap
        Buffer holding the header
    `offset` — int (optional)
        Where the header starts within `buf` (default: `0`)

    Returns
    -------
    `tuple`
        `(n, name_len, mask_len, payload_len, crc)`

    Raises
    ------
    `ValueError`
        If the bytes at `offset` are not a valid record header
    """
    if len(buf) - offset < MSV_HEADER_SIZE:
        raise ValueError("Record is shorter than an msv header")
    (magic, version, dtype, n, name_len, mask_len,
     payload_len, crc) = MSV_HEADER.unpack_from(buf, offset)
    if magic != MSV_MAGIC:
        raise ValueError(f"Record does not start with {MSV_MAGIC!r}")
    if version != MSV_VERSION:
        raise ValueError(f"Unsupported msv record version {version}")
    if dtype != MSV_DTYPE_FLOAT16:
        raise ValueError(f"Unsupported msv payload dtype {dtype}")
    if payload_len != n * (n + 1):
        raise ValueError(
            f"Payload of {payload_len} bytes does not match a {n}x{n} upper triangle")
    return n, name_len, mask_len, payload_len, crc


def _scan_header_region(path, start, stop):
    """
    Read the headers of the records whose start marker lies in `[start, stop)`.

    `start` usually falls inside some record, so the region searches forward for
    the nearest marker and resynchronises from there. A record belongs to the
    region it begins in, even when it runs past `stop`; the region beginning at
    `stop` searches past it, so every record is reported by exactly one region
    wherever the cuts fall.

    Only a record's header, name and bitmask are read. Its payload is stepped
    over, so a damaged payload goes unnoticed here and is caught by the checksum
    when the record is read. Runs in a worker process, so it opens and maps the
    shard itself.

    Parameters
    ----------
    `path` — str
        Path to a shard file
    `start` — int
        First byte a marker may begin at for this region
    `stop` — int
        Markers beginning at or past this byte belong to the next region

    Returns
    -------
    `list`
        `(offset, len, name, n)` for each record found, in file order
    """
    size = os.path.getsize(path)
    start, stop = int(start), min(int(stop), size)
    headers = []
    if size < MSV_HEADER_SIZE or start >= stop:
        return headers

    # Reaches just far enough to see a marker that begins before `stop`.
    search_end = min(size, stop + len(MSV_MAGIC) - 1)
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            pos = start
            while pos < stop:
                found = mm.find(MSV_MAGIC, pos, search_end)
                if found < 0:
                    break
                try:
                    n, name_len, mask_len, payload_len, _ = _read_record_header(mm, found)
                    total = MSV_HEADER_SIZE + name_len + mask_len + payload_len
                    if found + total > size:
                        raise ValueError("record runs past the end of the shard")
                    name_at = found + MSV_HEADER_SIZE
                    mask_at = name_at + name_len
                    name = mm[name_at:mask_at].decode('utf-8')
                    bitmask_to_saved_columns(mm[mask_at:mask_at + mask_len])
                except Exception as e:
                    logging.debug(f"Skipping bad record at offset {found} of {path}: {e}")
                    # Move past the marker only, never past the body it claimed, so a
                    # stray marker cannot carry the scan over a real record.
                    pos = found + len(MSV_MAGIC)
                    continue
                headers.append((found, total, name, int(n)))
                pos = found + total
        finally:
            mm.close()
    return headers


def scan_shard_headers(path, workers=None):
    """
    List a shard's records by location and name, using every CPU core.
    Meant for rebuilding the database index. Payloads are not read. Fetch a 
    matrix afterwards with `read_score_record()`, or use `scan_shard()` 
    to walk the records themselves.

    Parallel regions pay off on SSD, NVMe and parallel filesystems. On a single
    spinning disk the head seeks between regions and the scan slows down, so
    pass `workers=1` there.

    Parameters
    ----------
    `path` — str
        Path to a shard file
    `workers` — int (optional)
        Worker processes (default: every usable CPU core). Reduced so no region
        is smaller than `MSV_SCAN_MIN_REGION_BYTES`; `1` scans in-process.

    Returns
    -------
    `list`
        `(offset, len, name, n)` for each record found, in file order
    """
    size = os.path.getsize(path)
    if size < MSV_HEADER_SIZE:
        return []
    if workers is None:
        workers = _usable_cpu_count()
    workers = max(1, min(int(workers), size // MSV_SCAN_MIN_REGION_BYTES))
    if workers == 1:
        return _scan_header_region(path, 0, size)

    bounds = [size * i // workers for i in range(workers + 1)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        regions = pool.map(_scan_header_region, repeat(path), bounds[:-1], bounds[1:])
        return [header for region in regions for header in region]


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

def visible_gpu_devices():
    """
    Device ids this process is allowed to use, in the order CUDA sees them.

    Returns
    -------
    `list`
        Device ids as strings, at least one entry.
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible is not None and cuda_visible.strip() != "":
        devices = [d.strip() for d in cuda_visible.split(',') if d.strip()]
        if devices:
            return devices
        logging.warning("CUDA_VISIBLE_DEVICES is set but empty; falling back to all GPUs")

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, check=True)
        devices = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
        if devices:
            return devices
    except Exception as e:
        logging.warning(f"Could not query nvidia-smi for the GPU count ({e})")

    logging.warning("Could not detect GPU count; defaulting to device 0")
    return ['0']