import glob
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import zarr
from Bio import AlignIO, ExPASy, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from massivora import cpp_bindings
from massivora.utils import (get_cuda_module, gpu_is_available,
                            saved_columns_to_bitmask)

logger = logging.getLogger(__name__)

# What the concatenation stages do with a shared-source pair
SHARED_SOURCE_STRATEGIES = ('skip', 'fix')
# Residue codes are 0..21 once lower case is folded onto upper case
_RESIDUE_BITS = 5
# Where the shared-source run tables of an alignment are kept (see
# `resolve_run_table_cache`): in memory only, next to the alignment, or in a
# directory given by the caller
RUN_TABLE_CACHE_MEMORY = None
RUN_TABLE_CACHE_MONOMER = 'monomer'
# Cache roots whose first failed save was already logged by this process
_run_table_save_warned = set()


class SharedSourceError(RuntimeError):
    """
    A concatenated pair whose rows pair a sequence with itself, left unrun
    because the shared-source strategy is 'skip' (or because 'fix' left no
    species to run on).
    """


def check_shared_source_strategy(strategy):
    """
    Validate a `concatenate.shared-source_strategy` value.

    Parameters
    ----------
    `strategy` — str
        'skip' or 'fix', in any case

    Returns
    -------
    `str`
        The strategy in lower case

    Raises
    ------
    `ValueError`
        If the strategy is not one of `SHARED_SOURCE_STRATEGIES`
    """
    value = str(strategy).strip().lower()
    if value not in SHARED_SOURCE_STRATEGIES:
        raise ValueError(f"concatenate.shared-source_strategy must be one of "
                         f"{', '.join(SHARED_SOURCE_STRATEGIES)}, got {strategy!r}")
    return value


def _residue_runs(block, k, row_bits):
    """
    Every run of `k` consecutive residues of each row of `block`, packed into
    one integer per run with the row index in the low `row_bits` bits.

    Gaps are dropped before the runs are taken, so a gap does not break a run,
    and lower-case residues count as their upper-case letter.
    """
    codes = np.abs(block.astype(np.int64))
    n_rows, n_cols = codes.shape
    if n_cols < k:
        return np.empty(0, dtype=np.int64)
    # Move each row's residues to the front, keeping their order
    order = np.argsort(codes == 0, axis=1, kind='stable')
    codes = np.take_along_axis(codes, order, axis=1)
    n_residues = np.count_nonzero(codes, axis=1)

    n_runs = n_cols - k + 1
    runs = np.zeros((n_rows, n_runs), dtype=np.int64)
    for i in range(k):
        runs = (runs << _RESIDUE_BITS) | codes[:, i:i + n_runs]
    valid = np.arange(n_runs)[None, :] <= (n_residues - k)[:, None]
    rows = np.broadcast_to(np.arange(n_rows, dtype=np.int64)[:, None], runs.shape)
    return (runs[valid] << row_bits) | rows[valid]


def _shared_source_rows(part_a, part_b, k=10):
    """
    Which rows of two column blocks share an exact run of `k` residues.

    Two different proteins essentially never share such a run, so a row that
    does pairs one source sequence with itself: the homologue searches of both
    proteins recruited the same sequence in that species.

    Parameters
    ----------
    `part_a`, `part_b` — numpy.ndarray
        Encoded columns of the first and of the second protein, one row per
        sequence
    `k` — int (optional)
        Length of the residue run, at most 12 (default: `10`)

    Returns
    -------
    `numpy.ndarray`
        One boolean per row
    """
    k = int(k)
    if not 1 <= k <= 12:
        raise ValueError(f"k must be between 1 and 12, got {k}")
    n_rows = part_a.shape[0]
    shared = np.zeros(n_rows, dtype=bool)
    # A run takes 5 bits per residue; the row index takes the bits above it,
    # which bounds how many rows one chunk can test at once.
    row_bits = min(10, 62 - _RESIDUE_BITS * k)
    chunk = 1 << row_bits
    mask = chunk - 1
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        runs_a = np.unique(_residue_runs(part_a[start:stop], k, row_bits))
        runs_b = np.unique(_residue_runs(part_b[start:stop], k, row_bits))
        common = np.intersect1d(runs_a, runs_b, assume_unique=True)
        shared[start + (common & mask)] = True
    return shared


def build_run_table(block, k=10):
    """
    The sorted unique runs of `k` residues of each row of `block`, as a CSR
    pair (codes uint64, offsets int64): row i's codes are
    codes[offsets[i]:offsets[i + 1]]. Same runs as `_residue_runs` (gaps
    dropped, lower case folded), without the row index packed in.
    """
    codes = np.abs(block.astype(np.int64))
    n_rows, n_cols = codes.shape
    n_runs = n_cols - k + 1
    if n_runs <= 0 or n_rows == 0:
        return np.empty(0, dtype=np.uint64), np.zeros(n_rows + 1, dtype=np.int64)
    order = np.argsort(codes == 0, axis=1, kind='stable')
    codes = np.take_along_axis(codes, order, axis=1).astype(np.uint64)
    n_residues = np.count_nonzero(codes, axis=1)
    runs = np.zeros((n_rows, n_runs), dtype=np.uint64)
    shift = np.uint64(_RESIDUE_BITS)
    for i in range(k):
        runs = (runs << shift) | codes[:, i:i + n_runs]
    valid = np.arange(n_runs)[None, :] <= (n_residues - k)[:, None]
    sentinel = np.uint64(2**64 - 1)         # above every code: sorts last
    runs[~valid] = sentinel
    runs.sort(axis=1)
    keep = runs != sentinel
    keep[:, 1:] &= runs[:, 1:] != runs[:, :-1]
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(keep.sum(axis=1), out=offsets[1:])
    return runs[keep], offsets


def store_signature(path):
    """
    The files of the `align` array of the zarr store at `path` (relative
    name, size, mtime in ns), as bytes: it changes whenever the store is
    written again, however little of the matrix changed. Empty when `path`
    holds no `align` directory (an alignment that was never stored).
    """
    root = os.path.join(path, 'align')
    if not os.path.isdir(root):
        return b''
    entries = []
    for directory, _, files in os.walk(root):
        for name in files:
            full = os.path.join(directory, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            entries.append(f'{os.path.relpath(full, root)}\0{st.st_size}\0{st.st_mtime_ns}')
    return '\n'.join(sorted(entries)).encode('utf-8')


def alignment_fingerprint(matrix, species_index_map=(), species_list=(), store=None):
    """
    16 bytes that identify the content of an alignment matrix for the run
    table cache: its shape, its species index map and list, every
    `n_rows // 256`-th row and its last row (at most 512 rows, so ≈ 0.1 ms
    per monomer; hashing all rows would cost 25 ms per median monomer, 60 %
    of its zarr load, on every pair with the executor's 8-entry monomer
    cache), plus the `store_signature` of the zarr store `store` when one
    is given. A re-generated alignment is therefore told from the old one
    when its store was written again (new chunk files: size or mtime), when
    its shape, species or a sampled row changed, or when it has more than
    512 rows and the change lies in an unsampled row of a store rewritten
    with the old files' sizes and mtimes to the nanosecond — the one case
    that is not detected. A table built for the old alignment is then not
    reused (`RunTable.load`).
    """
    matrix = np.asarray(matrix)
    n_rows = int(matrix.shape[0])
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array(matrix.shape, dtype=np.int64).tobytes())
    h.update(np.asarray(list(species_index_map), dtype=np.int64).tobytes())
    h.update('\n'.join(str(s) for s in species_list).encode('utf-8'))
    if n_rows:
        step = max(1, n_rows // 256)
        h.update(np.ascontiguousarray(matrix[::step], dtype=np.int8).tobytes())
        h.update(np.ascontiguousarray(matrix[-1], dtype=np.int8).tobytes())
    signature = store_signature(store) if store else b''
    if signature:
        h.update(b'\0store\0' + signature)
    return h.digest()


class RunTable(object):
    """
    The residue runs of some rows of one protein's alignment, computed once
    and reused by every pair the protein is part of (issue #15): the runs of a
    row do not depend on the partner protein. Held as `build_run_table`'s CSR
    arrays for `rows` of the matrix, and optionally saved next to the
    alignment or in a cache directory, where every process maps the same copy.
    A saved table carries the `alignment_fingerprint` of the matrix it was
    built from and is only reused for that matrix.
    """
    FORMAT = 2
    FINGERPRINT_BYTES = 16
    __slots__ = ('k', 'codes', 'offsets', 'rows', 'shape', 'fingerprint')

    def __init__(self, k, codes, offsets, rows, shape, fingerprint=b''):
        self.k, self.codes, self.offsets = int(k), codes, offsets
        self.rows, self.shape = np.asarray(rows, dtype=np.int64), tuple(int(s) for s in shape)
        self.fingerprint = bytes(fingerprint)

    @classmethod
    def build(cls, matrix, rows, k=10, fingerprint=None):
        rows = np.asarray(rows, dtype=np.int64)
        codes, offsets = build_run_table(matrix[rows], k)
        if fingerprint is None:
            fingerprint = alignment_fingerprint(matrix)
        return cls(k, codes, offsets, rows, matrix.shape, fingerprint)

    @staticmethod
    def directory(path, k):
        return os.path.join(path, f'shared_source_k{int(k)}')

    @staticmethod
    def remove_all(path):
        """Remove every run table saved under `path` (any k): the alignment
        there is about to be overwritten."""
        for directory in glob.glob(os.path.join(glob.escape(path), 'shared_source_k*')):
            shutil.rmtree(directory, ignore_errors=True)

    def _meta(self):
        fp = np.frombuffer(self.fingerprint.ljust(self.FINGERPRINT_BYTES, b'\0')
                           [:self.FINGERPRINT_BYTES], dtype=np.int64)
        return np.array([self.FORMAT, self.k, *self.shape, *fp], dtype=np.int64)

    @classmethod
    def load(cls, path, k, shape, fingerprint=None):
        """The table saved under `path` for runs of `k`, mapped read-only, or
        None when there is none or it was built for another matrix (another
        shape, or another `fingerprint` when one is given)."""
        directory = cls.directory(path, k)
        meta = os.path.join(directory, 'meta.npy')
        if not os.path.exists(meta):
            return None
        try:
            values = np.load(meta)
            if len(values) != 4 + cls.FINGERPRINT_BYTES // 8:
                return None
            fmt, kk, n_rows, n_cols = (int(v) for v in values[:4])
            saved_fp = np.ascontiguousarray(values[4:], dtype=np.int64).tobytes()
            if fmt != cls.FORMAT or kk != int(k) or (n_rows, n_cols) != tuple(int(s) for s in shape):
                return None
            if fingerprint is not None and saved_fp != bytes(fingerprint):
                logger.info(f"Ignoring the shared-source run table under {directory}: "
                            "built for another alignment of the same shape")
                return None
            codes = np.load(os.path.join(directory, 'codes.npy'), mmap_mode='r')
            offsets = np.load(os.path.join(directory, 'offsets.npy'), mmap_mode='r')
            rows = np.load(os.path.join(directory, 'rows.npy'))
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring the shared-source run table under {directory}: {e}")
            return None
        return cls(k, codes, offsets, rows, shape, saved_fp)

    def save(self, path, replace=False):
        """Save under `path` (atomically: a finished table appears or none).
        A table already there is kept unless `replace` (it is stale: built
        for another matrix). False if the location cannot be written: logged
        once per cache root and process, since every pair then pays for
        building the table."""
        directory = self.directory(path, self.k)
        if os.path.exists(os.path.join(directory, 'meta.npy')):
            if not replace:
                return True
            shutil.rmtree(directory, ignore_errors=True)
        tmp = f'{directory}.{os.getpid()}.tmp'
        try:
            os.makedirs(path, exist_ok=True)
            os.makedirs(tmp, exist_ok=True)
            np.save(os.path.join(tmp, 'codes.npy'), np.ascontiguousarray(self.codes))
            np.save(os.path.join(tmp, 'offsets.npy'), np.ascontiguousarray(self.offsets))
            np.save(os.path.join(tmp, 'rows.npy'), self.rows)
            np.save(os.path.join(tmp, 'meta.npy'), self._meta())
            os.rename(tmp, directory)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            if os.path.exists(os.path.join(directory, 'meta.npy')):
                return True                 # another process won the race
            root = os.path.dirname(os.path.normpath(path)) or path
            if root not in _run_table_save_warned:
                _run_table_save_warned.add(root)
                logger.warning(
                    f"Cannot save the shared-source run table under {directory}: {e}. "
                    "Every process builds the tables itself (about 0.3 s per pair); point "
                    "concatenate.shared-source_cache at a writable directory")
            return False
        return True

    def positions(self, rows):
        """Table positions of matrix `rows`; KeyError if one is not cached."""
        rows = np.asarray(rows, dtype=np.int64)
        pos = np.searchsorted(self.rows, rows)
        pos[pos == len(self.rows)] = 0
        if not np.array_equal(self.rows[pos], rows):
            raise KeyError("rows not in the run table")
        return pos


def shared_source_rows_cached(table_a, table_b, rows_a, rows_b):
    """
    `_shared_source_rows` on cached run tables: one boolean per pair row, row r
    pairing matrix row rows_a[r] of the first protein with rows_b[r] of the
    second. Both tables must be built for the same `k`.
    """
    if table_a.k != table_b.k:
        raise ValueError("the run tables were built for different k")
    pos_a, pos_b = table_a.positions(rows_a), table_b.positions(rows_b)
    shared_rows = getattr(cpp_bindings, 'sharedSourceRows', None)
    if shared_rows is not None:
        return np.asarray(shared_rows(table_a.codes, table_a.offsets, table_b.codes,
                                      table_b.offsets, pos_a, pos_b), dtype=bool)
    # numpy: prefix each code with its pair row and search the (sorted) codes
    # of the second protein in those of the first, a chunk of rows at a time
    n = len(pos_a)
    shared = np.zeros(n, dtype=bool)
    code_bits = _RESIDUE_BITS * table_a.k
    chunk = 1 << (64 - code_bits)
    for start in range(0, n, chunk):
        a, la = _gather_rows(table_a, pos_a[start:start + chunk])
        b, lb = _gather_rows(table_b, pos_b[start:start + chunk])
        if a.size == 0 or b.size == 0:
            continue
        a |= la << np.uint64(code_bits)
        b |= lb << np.uint64(code_bits)
        at = np.searchsorted(a, b)
        at[at == a.size] = 0
        shared[start + lb[a[at] == b].astype(np.int64)] = True
    return shared


def _gather_rows(table, pos):
    """Flat codes of the table rows at `pos`, and the local row of each code."""
    starts = table.offsets[pos]
    lens = table.offsets[pos + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    local = np.repeat(np.arange(len(pos), dtype=np.uint64), lens)
    first = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return np.array(table.codes[first + np.arange(total)], dtype=np.uint64), local


def resolve_run_table_cache(setting, project_path=None):
    """
    The `cache` argument of `resolve_shared_source` for a
    `concatenate.shared-source_cache` setting:
      off:       None, the run tables stay in memory, per process
      monomer:   next to each monomer alignment (writes into the MSA tree)
      auto:      `<project_path>/shared_source_cache` (the default; None
                 without a project path, so nothing is ever written)
      directory: that directory, relative to `project_path` when given
    """
    if setting is None:
        return RUN_TABLE_CACHE_MEMORY
    mode = str(setting).strip()
    if mode.lower() in ('', 'off', 'none', 'memory'):
        return RUN_TABLE_CACHE_MEMORY
    if mode.lower() == RUN_TABLE_CACHE_MONOMER:
        return RUN_TABLE_CACHE_MONOMER
    if mode.lower() == 'auto':
        return os.path.join(project_path, 'shared_source_cache') if project_path else None
    if project_path and not os.path.isabs(mode):
        return os.path.join(project_path, mode)
    return mode


def resolve_shared_source(alignment, strategy='skip', name=None, k=10, threshold=0.05,
                          cache=None):
    """
    Apply a shared-source strategy to a freshly concatenated pair.

    Parameters
    ----------
    `alignment` — TextAlignment or BinaryAlignment
        Alignment concatenated from two proteins; changed in place under 'fix'
    `strategy` — str (optional)
        'skip' or 'fix' (default: `'skip'`)
    `name` — str (optional)
        Pair name for the log and the error message (default: `None`)
    `k` — int (optional)
        Length of the residue run that marks a shared source (default: `10`)
    `threshold` — float (optional)
        Share of species above which the pair counts as shared-source (default: `0.05`)
    `cache` — str (optional)
        For a BinaryAlignment pair, where the monomers' run tables are kept:
        None in memory only, 'monomer' next to each monomer, anything else a
        directory (`resolve_run_table_cache`) (default: `None`)

    Returns
    -------
    TextAlignment or BinaryAlignment
        The alignment, with its shared-source species removed under 'fix'

    Raises
    ------
    `SharedSourceError`
        Under 'skip', for a shared-source pair; under 'fix', when removing the
        shared-source species leaves only the query
    """
    strategy = check_shared_source_strategy(strategy)
    # The run-table cache is a BinaryAlignment feature
    is_binary = isinstance(alignment, BinaryAlignment)
    kwargs = {'cache': cache} if is_binary else {}
    try:
        if not alignment.Is_Shared_Source(k=k, threshold=threshold, **kwargs):
            return alignment
        label = f"Pair '{name}'" if name else "Pair"
        share = alignment.shared_source_share
        if strategy == 'skip':
            raise SharedSourceError(
                f"{label} is a shared-source concatenation: {share:.1%} of its species pair "
                f"one sequence with itself (concatenate.shared-source_strategy: skip)")
        removed = alignment.Remove_Shared_Source(k=k, **kwargs)
        if alignment.matrix.shape[0] <= 1:
            raise SharedSourceError(
                f"{label} is a shared-source concatenation and no species is left after "
                f"removing the {removed} that pair one sequence with itself")
        logger.info(f"{label}: removed {removed} shared-source species ({share:.1%}); "
                    f"{alignment.matrix.shape[0]} sequences left")
        return alignment
    finally:
        if is_binary:
            # The check is done: let go of the monomers the pair was made
            # from, so the caller's `del` of them frees their matrices
            alignment._pair_source = None


class TextAlignment(MultipleSeqAlignment):
    def __init__(self, records):
        # Allowing contruction from file path
        if type(records) == str:
            if os.path.isfile(records):
                logger.info(f'Loading alignment from file: {records}')
                _, ext = os.path.splitext(records)
                self.alignmentName = os.path.basename(_)
                if ext.lower() == ".sto":
                    fmt = "stockholm"
                elif ext.lower().startswith(".fa") or ext.lower() == ".a2m":
                    fmt = "fasta"
                else:
                    logger.error(f'Input file {os.path.basename(_)} has an unsupported format {ext}')
                    raise NotImplementedError
                records = AlignIO.read(records, fmt)
            else: raise FileNotFoundError(f"File not found: {records}")
        elif isinstance(records, SeqRecord):
            # construct with single record
            records = [records]
        # construct with biopython supported iterable
        super().__init__(records)
        try:
            self.querySequence = self._records[0]
            self._update_info()
        except IndexError:
            logger.error("Alignment is empty")
            raise ValueError("Alignment is empty")
        self.saved_columns = []
        logger.debug(f'Alignment loaded: {len(self)} sequences, {len(self[0])} positions')

    def _parse_description(self, description):
        # TODO: Try the taxid first, if not exist, try the species name
        pattern = r"OS=(.*?)(?=\s+(?:OX|GN|PE|SV)=|$)"\
                + r"(?:\sOX=(.*?))?(?=\s+(?:GN|PE|SV)=|$)"\
                + r"(?:\sGN=(.*?))?(?=\s+(?:PE|SV)=|$)"\
                + r"(?:\sPE=(.*?))?(?=\s+(?:SV)=|$)"\
                + r"(?:\sSV=(.*?))?$"
        #r"OS=(.*?)(?=\s+(?:GN|PE|SV)=|$)(?:\s(?:GN)=(.*?))?(?=\s+(?:PE|SV)=|$)(?:\s(?:PE)=(.*?))?(?=\s+(?:SV)=|$)(?:\s(?:SV)=(.*?))?$"
        match = re.search(pattern, description)
        if match:
            # matching pattern OS=... OX=... GN=... PE=... SV=...
            return match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)
        return None, None, None, None, None

    def _analyze_descriptions(self):
        self._desc_list = []
        for record in self._records:
            os_, ox, gn, pe, sv = self._parse_description(record.description)
            self._desc_list.append({
                "id": record.id,
                "OS": os_,
                "OX": int(ox) if ox else None,
                "GN": gn,
                "PE": pe,
                "SV": sv
            })

    def _update_info(self):
        self.sequenceLength = len(self._records[0].seq)
        self.sequenceCount = len(self._records)
        self.matrix = self.Get_Numpy_Array()
        self.encoded_matrix = self.Get_Numpy_Array(mapped=True)
        # Width of each protein in the alignment. A concatenation sets its two
        # widths itself; a single protein follows its own width.
        if len(getattr(self, 'lengthes', None) or []) <= 1:
            self.lengthes = [self.sequenceLength]

    @classmethod
    def From_SwissProt(cls, protein_id):
        """
        Create an instance from SwissProt record. This function will download 
        protein record from the SwissProt server and then load it. 
        
        If the user wish to load an already downloaded file, they should load this file 
        with BioPython, and then use the constructor of this class to create an instance.

        >>> with open("protein.txt") as f:
        >>>     record = SeqIO.read(f, 'swiss')
        >>> alignment = TextAlignment(record)

        Parameters
        ----------
        `protein_id` — str
            Uniprot ID of the protein

        Returns
        -------
        `TextAlignment`
            The alignment instance of that only contains this protein
        """
        try:
            handle = ExPASy.get_sprot_raw(protein_id)
            record = SeqIO.read(handle, 'swiss')
            return cls(record)
        except ValueError as e:
            logger.error(f"Failed to fetch protein {protein_id}: {e}")
            return None

    def AnalogueSearch(self, query=None, threshold=0.2, output_prefix=None, jackhmmer_path=None, iterations=5, threads=4, database=None):
        if not query:
            query = output_prefix + '.fasta'
            SeqIO.write(self._records[0], query, "fasta")
            search_threshold = str(threshold * self.sequenceLength)
        else:
            search_threshold = str(threshold * len(query))
        cmd = [
            jackhmmer_path,
            "-N", str(iterations),
            "-o", '/dev/null',
            "-A", output_prefix + '.sto',
            "--tblout", output_prefix + '.tblout',
            "--domtblout", output_prefix + '.domtblout',
            "--noali",
            "--notextw",
            "-T", search_threshold,
            "--domT", search_threshold,
            "--incT", search_threshold,
            "--incdomT", search_threshold,
            "--cpu", str(threads),
            query, database
        ]
        logger.debug("Running Jackhmmer with the following command line: "+ ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            logger.error(f"Jackhmmer failed with return code {r.returncode}")
            raise RuntimeError(f"Jackhmmer failed with return code {r.returncode}")
        else:
            self.__init__(output_prefix + '.sto')

    def Get_Numpy_Array(self, mapped=False, string=None):
        """
        Transform the MSA into a numpy array.

        Parameters
        ----------
        `mapped` — bool (optional)
            Whether to map the characters into numbers (default: `False`)
        `string` — str (optional)
            The string that contains all sequences in the MSA, 
            concatenated one after another (default: `None`)

        Returns
        -------
        `numpy.ndarray`
            The transformed MSA as a numpy array
        """
        if not string: string = ''.join([str(record.seq) for record in self])
        matrix = cpp_bindings.getAlignmentInNumpy(string, self.sequenceLength, self.sequenceCount)
        # First change all the '.' to '-'
        matrix[matrix == b'.'] = b'-'
        if mapped:
            # Then map the characters to numbers
            # All the invalid characters are mapped to 21
            mapping = np.full(128, 21, dtype=np.int8)
            valid_chars = 'acdefghiklmnpqrstvwy-ACDEFGHIKLMNPQRSTVWY'
            values = np.arange(-20, 21, dtype=np.int8)
            for char, value in zip(valid_chars, values): mapping[ord(char)] = value
            ascii_codes = np.frombuffer(matrix.tobytes(), dtype=np.int8)
            mapped_flat = mapping[ascii_codes]
            matrix = mapped_flat.reshape(matrix.shape)
        return matrix

    def Filtering_MSA_Invalid(self):
        """
        Filtering out the sequences who has invalid characters in the MSA. 
        Valid characters include 20 natural amino acids and the gap character '-'.
        """
        # Remove sequences with invalid characters from the alignment
        valid_rows = np.where(np.max(self.encoded_matrix, axis=1) <= 20)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), valid_rows)
 
        if not hasattr(self, "_desc_list"):
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
        else:
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
                del self._desc_list[i]
        self._update_info()
        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Filtering_MSA_Gap(self, row_keep_ratio=0.5, col_keep_ratio=0.5):
        """
        Filtering out the gaps in the MSA according to the set ratios. 
        This function does: 1. Remove the all-gap columns. 2. Remove sequences 
        whose gap ratio are higher than `row_keep_ratio`. 3. Create masks for 
        columns whose gap ratio are higher than `col_keep_ratio`.

        Parameters
        ----------
        `row_keep_ratio` — float (optional)
            Ratio threshold for keep rows (default: `0.5`)
        `col_keep_ratio` — float (optional)
            Ratio threshold for keep columns (default: `0.5`)
        """
        # Compute sequence identity of every sequence to the query
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        
        # Remove gaps if the input is a stockholm file
        nongap_columns_to_keep = np.array([char != '-' for char in self._records[0].seq])
        matrix = self.matrix.copy()
        encoded_matrix = self.encoded_matrix.copy()
        encoded_matrix = encoded_matrix[:, nongap_columns_to_keep]

        # Calculate the ratio of '-' in each row and filter out rows with a ratio higher than row_keep_ratio
        logger.debug("Calculate rows to keep")
        row_gap_ratios = np.sum(encoded_matrix == 0, axis=1) / encoded_matrix.shape[1]
        indices_to_keep = np.where(row_gap_ratios <= row_keep_ratio)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), indices_to_keep)
        encoded_matrix = encoded_matrix[indices_to_keep]
        logger.debug(f"Matrix shape after row filtering: {encoded_matrix.shape}")

        # Calculate the ratio of '-' in each column and filter out columns with a ratio higher than col_keep_ratio
        logger.debug("Calculate columns to keep")
        L = encoded_matrix.shape[0]
        col_gap_ratios = np.sum(encoded_matrix == 0, axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < col_keep_ratio)[0]
        self.saved_columns = cols_to_keep.tolist() # Just keep it as a mask

        # Calculate identity to query for remaining sequences
        temp_matrix = encoded_matrix.copy()[:, cols_to_keep]
        query_seq = temp_matrix[0]
        ident_to_query = np.round(
            (
                np.sum(temp_matrix == query_seq.reshape(1, -1), axis=1)
                + encoded_matrix.shape[1] # removed columns are treated as matches
                - temp_matrix.shape[1]
            ) / encoded_matrix.shape[1]
        , 3)
        for i, identity in zip(indices_to_keep, ident_to_query):
            self._desc_list[i]["query_identity"] = float(identity)

        logger.debug("Reconstructing alignment")
        matrix = matrix[:, nongap_columns_to_keep]
        matrix = matrix[indices_to_keep]

        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
            del self._desc_list[i]
        assert len(self._records) == L
        for i, record in enumerate(self):
            record.letter_annotations = {}
            byte_t = bytes()
            for j in matrix[i]: byte_t += j
            record.seq = Seq(byte_t.decode())

        self._update_info()

        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Best_Reciprocal_Hit(self, paralog_threshold=0.9, allowed_error=0.02):
        """
        Perform the best reciprocal hit scheme for paralog filtering on this 
        alignment instance. 

        Parameters
        ----------
        `paralog_threshold` — float (optional)
            Threshold for similarity check (default: `0.9`)
        `allowed_error` — float (optional)
            Allowed similarity error (default: `0.02`)
        """
        # Compute sequence identity of every sequence to the query
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        encoded_matrix = self.encoded_matrix.copy()
        if self._desc_list[0].get("query_identity") is None:
            # If the identity to query has not been calculated, calculate it now
            if self.saved_columns:
                # Columns that are masked out are treated as matches
                reduced_matrix = encoded_matrix[:, self.saved_columns]
                query_seq = reduced_matrix[0]
                ident_to_query = np.round(np.sum(reduced_matrix == query_seq.reshape(1, -1), axis=1)
                                          + self.sequenceLength
                                          - len(self.saved_columns), 3) / self.sequenceLength
            else:
                query_seq = encoded_matrix[0]
                ident_to_query = np.round(np.sum(encoded_matrix == query_seq.reshape(1, -1), axis=1), 3) / self.sequenceLength
            for i, identity in enumerate(ident_to_query):
                self._desc_list[i]["query_identity"] = float(identity)

        desc_df = pd.DataFrame(self._desc_list)
        indices_to_keep1 = (
            desc_df
            .sort_values(by="query_identity", ascending=False)   # highest -> lowest
            .groupby("OX", sort=False)
            .head(1)                            # keep first (highest identity) per OX
            .index.to_numpy()                   # these are the row indices in self._records
        )
        logger.debug("This alignment has {} unique species.".format(len(indices_to_keep1)))

        # Get query taxid
        query_name = self[0].name.split('/')[0]
        for i, rec in enumerate(self[1:]):
            if query_name in rec.name:
                paralog_taxid = self._desc_list[i+1]["OX"]
                break
        else:
            logger.warning("No paralog taxid could be identified; skipping paralog filtering.")
            return

        # Collect all paralog records sharing this taxid
        paralog_indices = []
        for i, desc in enumerate(self._desc_list):
            if desc["OX"] == paralog_taxid and self._desc_list[i]["query_identity"] < paralog_threshold:
                paralog_indices.append(i)
        if len(paralog_indices) == 0:
            logger.warning(f"No paralog sequences found for identified taxid {paralog_taxid}; skipping paralog filtering.")
            return

        # Calculate identity of each sequence to the paralog group
        if self.saved_columns:
            reduced_matrix = encoded_matrix[:, self.saved_columns]
            paralog_seqs = reduced_matrix[paralog_indices]
            reduced_matrix = reduced_matrix[indices_to_keep1, :]
            ident_to_paralog = np.round(np.sum(reduced_matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2)
                                        + self.sequenceLength - len(self.saved_columns), 3) / self.sequenceLength
        else:
            paralog_seqs = encoded_matrix[paralog_indices]
            ident_to_paralog = np.round(np.sum(encoded_matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2), 3) / self.sequenceLength

        # Compare similarities, if sequence is more similar to any paralog than the query, remove it
        indices_to_keep2 = [0]  # always keep query
        for i, i_all in enumerate(indices_to_keep1):
            max_paralog_id = np.max(ident_to_paralog[:, i])
            if max_paralog_id < self._desc_list[i_all]["query_identity"] + allowed_error:
                indices_to_keep2.append(i_all)
        
        all_indices = np.arange(1, len(self._records))
        indices_to_delete = np.setdiff1d(all_indices, indices_to_keep2)
        logger.debug("Best reciprocal hits retained. The alignment now has {} sequences.".format(len(indices_to_keep2)))

        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
            del self._desc_list[i]

        self._update_info()

    def __add__(self, other):
        """
        Contatenate two MSAs according to the sequence taxids. For paralogous 
        sequences, only the one with the highest similarity to the query will 
        be kept.

        Parameters
        ----------
        `other` — TextAlignment
            The other alignment instance

        Returns
        -------
        `TextAlignment`
            The concatenated alignment instance

        Raises
        ------
        `NotImplementedError`
            If the other operand is not an alignment instance
        """
        if not isinstance(other, MultipleSeqAlignment):
            raise NotImplementedError("Only MultipleSeqAlignment can be concatenated")
        
        # Create species-based dictionary for both alignments
        self_species = {}
        other_species = {}

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        if not hasattr(other, "_desc_list"):
            other._analyze_descriptions()

        for rec, desc in zip(self[1:], self._desc_list[1:]):
            taxid = desc["OX"]
            if taxid and taxid not in self_species:
                self_species[taxid] = []
            self_species[taxid].append((rec, desc))

        for rec, desc in zip(other[1:], other._desc_list[1:]):
            taxid = desc["OX"]
            if taxid and taxid not in other_species:
                other_species[taxid] = []
            other_species[taxid].append((rec, desc))

        logger.debug(f"Found {len(self_species)} species in alignment 1")
        logger.debug(f"Found {len(other_species)} species in alignment 2")

        # Merge records from both alignments
        merged = [SeqRecord(self[0].seq + other[0].seq, id='Query', name='Query', description=self[0].id+'_'+other[0].id)]
        common_species = set(self_species.keys()) & set(other_species.keys())
        logger.debug(f"Found {len(common_species)} common species")
        redundant_species = []
        ## First concatenate the sequences that are unique to each alignment
        logger.debug("Looping through common species")
        for species in common_species:
            if len(self_species[species]) == 1 and len(other_species[species]) == 1:
                rec1 = self_species[species][0][0]
                rec2 = other_species[species][0][0]
                # logger.debug(f'rec1.name: {rec1.name}, rec2.name: {rec2.name}, species: {species}')
                new_rec = SeqRecord(rec1.seq + rec2.seq, id=str(species), name=str(species), description=rec1.id+'_'+rec2.id)
                merged.append(new_rec)
            else:
                redundant_species.append(species)
        ## Then concatenate the sequences that are redundant in both alignments
        logger.debug(f"Found {len(redundant_species)} redundant species")
        for species in redundant_species:
            identity = 0
            for rec, desc in self_species[species]:
                if desc["query_identity"] > identity:
                    identity = desc["query_identity"]
                    most_similar_seq_in_self = rec
            identity = 0
            for rec, desc in other_species[species]:
                if desc["query_identity"] > identity:
                    identity = desc["query_identity"]
                    most_similar_seq_in_other = rec
            new_rec = SeqRecord(most_similar_seq_in_self.seq + most_similar_seq_in_other.seq, id=str(species), name=str(species), description=most_similar_seq_in_self.id+'_'+most_similar_seq_in_other.id)
            merged.append(new_rec)
        logger.info(f'Alignment merged: {len(merged)} sequences, {len(merged[0])} positions')
        concatenated = TextAlignment(merged)
        concatenated.lengthes = [len(self[0].seq), len(other[0].seq)]
        return concatenated

    def _pair_blocks(self):
        """
        The columns of each protein of an alignment concatenated from two.

        Raises
        ------
        `ValueError`
            If this alignment is not concatenated from exactly two proteins
        """
        lengthes = [int(length) for length in (getattr(self, 'lengthes', None) or [])]
        if len(lengthes) != 2 or sum(lengthes) != self.sequenceLength:
            raise ValueError(
                "Shared-source checks need an alignment concatenated from two proteins; "
                f"this one holds {len(lengthes)} protein(s) of widths {lengthes} "
                f"over {self.sequenceLength} columns")
        return self.encoded_matrix[:, :lengthes[0]], self.encoded_matrix[:, lengthes[0]:]

    def Is_Shared_Source(self, k=10, threshold=0.05):
        """
        Whether this concatenated pair pairs one sequence with itself.

        When the homologue searches of both proteins recruit the same sequence
        in a species, concatenation glues that sequence to itself and its
        matching columns covary perfectly. A species counts as shared-source
        when its two halves share an exact run of `k` residues (gaps removed),
        which two different proteins essentially never do. The share of such
        species, the query excluded, is kept in `shared_source_share`.

        Parameters
        ----------
        `k` — int (optional)
            Length of the residue run, at most 12 (default: `10`)
        `threshold` — float (optional)
            Share of species from which the pair counts as shared-source (default: `0.05`)

        Returns
        -------
        `bool`
            True if at least `threshold` of the species are shared-source

        Raises
        ------
        `ValueError`
            If this alignment is not concatenated from exactly two proteins
        """
        part_a, part_b = self._pair_blocks()
        shared = _shared_source_rows(part_a[1:], part_b[1:], k)
        self.shared_source_share = float(shared.mean()) if shared.size else 0.0
        return self.shared_source_share >= threshold

    def Remove_Shared_Source(self, k=10):
        """
        Remove every species whose two halves share an exact run of `k`
        residues, i.e. that pairs one sequence with itself. The query is kept.

        Parameters
        ----------
        `k` — int (optional)
            Length of the residue run, at most 12 (default: `10`)

        Returns
        -------
        `int`
            Number of sequences removed

        Raises
        ------
        `ValueError`
            If this alignment is not concatenated from exactly two proteins
        """
        part_a, part_b = self._pair_blocks()
        drop = np.flatnonzero(_shared_source_rows(part_a[1:], part_b[1:], k)) + 1
        for i in sorted(drop, reverse=True):
            del self._records[i]
            if hasattr(self, "_desc_list"):
                del self._desc_list[i]
        if len(drop):
            self._update_info()
        logger.debug(f"Removed {len(drop)} shared-source sequences; {len(self)} left")
        return int(len(drop))

    def Downsample_Randomly(self, to):
        """
        Downsample the alignment randomly to a lower size

        Parameters
        ----------
        `to` — int
            The number of sequences to downsample to
        """
        if to >= self.sequenceCount:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.sequenceCount} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.sequenceCount), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)
        chosen_records = [self._records[i] for i in chosen_indices]
        super().__init__(chosen_records)
        self._update_info()

    def To_Zarr(self, filename=None, overwrite=False):
        """
        Save the alignment as a Zarr format. The sequences will be grouped by 
        species, and the one with the highest identity to the query will be 
        put at the first position for each species. The species list and saved
        columns will be saved as Zarr attributes. This function will also create
        a species_index_map indicating the starting index of a species.

        Parameters
        ----------
        `filename` — str (optional)
            Filename of the Zarr archive. If not set, 
            will use alignmentName as default (default: `None`)
        `overwrite` — bool (optional)
            Whether to overwrite the existing file (default: `False`)
        """
        if not filename: filename = f'{self.alignmentName}'
        if os.path.exists(os.path.join(filename, 'align', 'zarr.json')):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                shutil.rmtree(os.path.join(filename, 'align'), ignore_errors=True)
                RunTable.remove_all(filename)      # built for the old matrix
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        species_list = ['Query']
        species_index_map = [0]

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()

        # Group sequences by species
        species = {}
        for rec, desc in zip(self[1:], self._desc_list[1:]):
            taxid = desc["OX"]
            if taxid is None: continue
            if taxid not in species.keys(): species[taxid] = []
            species[taxid].append(str(rec.seq))

        p_seq = 1
        tempstr = str(self[0].seq)
        for species, sequences in species.items():
            sequence_count = len(sequences)
            # for multiple records of the same species,
            # put the one with the highest identity to the query sequence at the first position
            if sequence_count != 1:
                identity = 0
                top_identity_index = 0
                q_arr = np.array(list(self.querySequence.seq))
                for this_index, rec in enumerate(sequences):
                    r_arr = np.array(list(rec))
                    this_identity = np.sum(q_arr == r_arr)
                    if this_identity > identity:
                        identity = this_identity
                        top_identity_index = this_index
                sequences.insert(0, sequences.pop(top_identity_index))
            for seq in sequences: tempstr += seq
            species_index_map.append(p_seq)
            species_list.append(species)
            p_seq += sequence_count
        matrix = self.Get_Numpy_Array(mapped=True, string=tempstr)
        grp = zarr.open_group(store=filename)
        # Store all-gap chunks too (and say so), so BinaryAlignment can treat any missing chunk as corruption
        z = grp.create_array(name='align', shape=(self.sequenceCount, self.sequenceLength), dtype='int8',
                             config={'write_empty_chunks': True})
        z.attrs.update({
            'species_list': species_list, 
            'species_index_map': species_index_map,
            'saved_columns': [list(self.saved_columns)],
            'lengthes': [len(self[0].seq)],
            'all_chunks_stored': True
        })
        z[:] = matrix


class BinaryAlignment(object):
    def __init__(self, filename=None):
        self.matrix = np.zeros((0, 0), dtype=np.int8)
        self.species_list_map = {}
        self.saved_columns = []
        self.lengthes = []
        # Where this alignment was loaded from (its run table may be kept there)
        self._path = filename if isinstance(filename, str) else None
        self._run_table = None
        self._fingerprint = None
        # Set by __add__: the two monomers and the rows taken from each, so
        # the shared-source check can use their run tables
        self._pair_source = None

        if isinstance(filename, str):
            try:
                arr = zarr.open_group(filename, mode='r')
                align = arr['align']
                # zarr fills missing chunks with fill_value (0, i.e. gap) instead of
                # raising, so a store with zarr.json but no chunk data reads as all gaps.
                # Stores written before 'all_chunks_stored' legitimately omit all-gap
                # chunks, so for those only require some data (the query row is never all gaps)
                expected = align.nchunks if align.attrs.get('all_chunks_stored') else 1
                if align.nchunks_initialized < expected:
                    raise ValueError(
                        f"only {align.nchunks_initialized}/{align.nchunks} chunks of 'align' "
                        f"are on disk (incomplete write?); regenerate it with overwrite=True"
                    )
                self.matrix = align[:]
                self.weights = align.attrs.get('weights', None)
                self.Beff = align.attrs.get('Beff', None)
                self.saved_columns = align.attrs.get('saved_columns', [])
                self.lengthes = align.attrs.get('lengthes', [self.matrix.shape[1]])
                self.species_list = align.attrs.get('species_list', [])
                self.species_index_map = align.attrs.get('species_index_map', {})
                self.species_list_map = {species: i for i, species in enumerate(self.species_list)}
            except Exception as e:
                raise FileNotFoundError(f"Error loading Zarr file {filename}: {e}") from e

    def __repr__(self):
        return "<%s instance (%i records of length %i) at %x>" % (
            self.__class__,
            self.matrix.shape[0],
            self.matrix.shape[1],
            id(self),
        )

    def __str__(self):
        return f"Alignment from Zarr with {len(self.species_list)} species"

    def __add__(self, other):
        """
        Concatenate the two MSAs according to the sequence taxids

        Parameters
        ----------
        `other` — BinaryAlignment
            The incoming alignment instance

        Returns
        -------
        `BinaryAlignment`
            The concatenated alignment instance
        """
        # Ensure that Query is the first species in both alignments
        common_species = sorted(list(set(self.species_list[1:]) & set(other.species_list[1:])))
        common_species.insert(0, self.species_list[0])

        new_instance = BinaryAlignment()
        new_instance.species_list = common_species
        new_instance.species_index_map = list(range(len(common_species)))
        new_instance.saved_columns = self.saved_columns + other.saved_columns
        new_instance.lengthes = self.lengthes + other.lengthes
        new_instance.species_list_map = {species: i for i, species in enumerate(new_instance.species_list)}

        # species_list_map[species] is the index of the species in the species_list
        # species_index_map[species_list_map[species]] is the index of the species in the matrix
        indices1 = np.array([self.species_index_map[self.species_list_map[species]] for species in common_species])
        indices2 = np.array([other.species_index_map[other.species_list_map[species]] for species in common_species])
        new_instance.matrix = np.concatenate((self.matrix[indices1], other.matrix[indices2]), axis=1)
        new_instance._pair_source = (self, other, indices1, indices2)

        return new_instance

    def fingerprint(self):
        """`alignment_fingerprint` of this alignment's matrix and, for one
        loaded from a store, of the store's `align` files; computed once."""
        if self._fingerprint is None:
            self._fingerprint = alignment_fingerprint(
                self.matrix, getattr(self, 'species_index_map', ()),
                getattr(self, 'species_list', ()), store=self._path)
        return self._fingerprint

    def run_table(self, k=10, cache=None):
        """
        This alignment's residue runs for the shared-source check, for the
        rows a concatenation can take from it (one per species), built once
        and kept. With a `cache` location (`run_table_path`) an alignment
        loaded from a zarr store also keeps the table on disk
        (`shared_source_k<k>/`), so the next process maps it instead of
        building it again; a saved table is used only when its fingerprint
        is this alignment's.
        """
        table = self._run_table
        if table is not None and table.k == int(k) and table.shape == self.matrix.shape:
            return table
        path = self.run_table_path(cache)
        table = RunTable.load(path, k, self.matrix.shape, self.fingerprint()) if path else None
        if table is None:
            rows = np.asarray(self.species_index_map if len(self.species_index_map) else
                              np.arange(self.matrix.shape[0]), dtype=np.int64)
            table = RunTable.build(self.matrix, rows, k, self.fingerprint())
            if path:
                # A table that is there but was not loaded is stale: replace it
                stale = os.path.exists(os.path.join(RunTable.directory(path, k), 'meta.npy'))
                table.save(path, replace=stale)
        self._run_table = table
        return table

    def run_table_path(self, cache=None):
        """
        Where this alignment's run tables are kept, or None for in memory
        only: `cache` None (or 'off') keeps them in memory, 'monomer' puts
        them next to the alignment, and any other value is a directory that
        holds one subdirectory per alignment name (on a file system where
        mapping is cheap, such as tmpfs; ZFS is not). 'auto' is resolved by
        the executors (`resolve_run_table_cache`); here it means in memory.
        An alignment not loaded from a store keeps its tables in memory.
        """
        if cache is None or self._path is None:
            return None
        mode = str(cache).strip()
        if mode.lower() in ('', 'off', 'none', 'memory', 'auto'):
            return None
        if mode.lower() == RUN_TABLE_CACHE_MONOMER:
            return self._path
        return os.path.join(mode, os.path.basename(os.path.normpath(self._path)))

    def _shared_rows(self, k=10, cache=None):
        """One boolean per row but the query: whether its two halves share a
        run of `k` residues. From the monomers' run tables when this pair was
        concatenated here (`__add__`) and its matrix still has the shape of
        that concatenation: `Downsample_Randomly`, `Remove_Shared_Source`,
        `Drop_Masked_Gap_Columns`, `resolve_shared_source` and a reassigned
        matrix of another shape all send it back to its rows; an in-place
        edit of `matrix` that keeps the shape is not noticed (no production
        path does that between `+` and the check). Else from its rows."""
        part_a, part_b = self._pair_blocks()         # also checks it is a pair
        source = self._pair_source
        if source is not None:
            a, b, rows_a, rows_b = source
            if (self.matrix.shape == (len(rows_a), a.matrix.shape[1] + b.matrix.shape[1])
                    and len(rows_a) == len(rows_b)):
                try:
                    return shared_source_rows_cached(a.run_table(k, cache), b.run_table(k, cache),
                                                     rows_a[1:], rows_b[1:])
                except KeyError:
                    pass                           # rows the tables do not hold
        return _shared_source_rows(part_a[1:], part_b[1:], k)

    def _pair_blocks(self):
        """
        The columns of each protein of an alignment concatenated from two,
        before or after `Drop_Masked_Gap_Columns`.

        Raises
        ------
        `ValueError`
            If this alignment is not concatenated from exactly two proteins
        """
        lengthes = [int(length) for length in (self.lengthes or [])]
        n_cols = self.matrix.shape[1]
        if len(lengthes) != 2:
            raise ValueError(
                "Shared-source checks need an alignment concatenated from two proteins; "
                f"this one holds {len(lengthes)} protein(s) of widths {lengthes}")
        if n_cols == sum(lengthes):
            split = lengthes[0]
        else:
            saved = self.saved_columns or []
            if len(saved) == 2 and n_cols == len(saved[0]) + len(saved[1]):
                split = len(saved[0])   # the masked gap columns are already dropped
            else:
                raise ValueError(
                    f"The alignment's {n_cols} columns match neither its protein widths "
                    f"{lengthes} nor its saved columns")
        return self.matrix[:, :split], self.matrix[:, split:]

    def _row_species(self):
        """The species of every row, from species_list and species_index_map."""
        species_list = list(getattr(self, 'species_list', None) or [])
        starts = [int(i) for i in (getattr(self, 'species_index_map', None) or [])]
        n_rows = self.matrix.shape[0]
        if not species_list or len(starts) != len(species_list):
            raise ValueError("species_list and species_index_map do not describe the rows")
        rows = []
        for species, start, stop in zip(species_list, starts, starts[1:] + [n_rows]):
            rows += [species] * (stop - start)
        if len(rows) != n_rows:
            raise ValueError("species_list and species_index_map do not describe the rows")
        return rows

    def Is_Shared_Source(self, k=10, threshold=0.05, cache=None):
        """
        Whether this concatenated pair pairs one sequence with itself.

        When the homologue searches of both proteins recruit the same sequence
        in a species, concatenation glues that sequence to itself and its
        matching columns covary perfectly. A species counts as shared-source
        when its two halves share an exact run of `k` residues (gaps removed),
        which two different proteins essentially never do. The share of such
        species, the query excluded, is kept in `shared_source_share`.

        A pair concatenated with `+` is tested from its monomers' run tables
        (`run_table`), computed once per protein; anything else from its rows.

        Parameters
        ----------
        `k` — int (optional)
            Length of the residue run, at most 12 (default: `10`)
        `threshold` — float (optional)
            Share of species from which the pair counts as shared-source (default: `0.05`)
        `cache` — str (optional)
            None keeps the monomers' run tables in memory only, 'monomer'
            next to them on disk, anything else is a directory to keep them
            in, one subdirectory per monomer (default: `None`)

        Returns
        -------
        `bool`
            True if at least `threshold` of the species are shared-source

        Raises
        ------
        `ValueError`
            If this alignment is not concatenated from exactly two proteins
        """
        shared = self._shared_rows(k, cache)
        self.shared_source_share = float(shared.mean()) if shared.size else 0.0
        return self.shared_source_share >= threshold

    def Remove_Shared_Source(self, k=10, cache=None):
        """
        Remove every species whose two halves share an exact run of `k`
        residues, i.e. that pairs one sequence with itself. The query is kept.

        Sequence weights computed before the removal no longer apply and are
        cleared; call `Reweight_Sequence` again when needed.

        Parameters
        ----------
        `k` — int (optional)
            Length of the residue run, at most 12 (default: `10`)
        `cache` — str (optional)
            As for `Is_Shared_Source` (default: `None`)

        Returns
        -------
        `int`
            Number of sequences removed

        Raises
        ------
        `ValueError`
            If this alignment is not concatenated from exactly two proteins
        """
        shared = self._shared_rows(k, cache)
        self._pair_source = None                # the rows are about to change
        n_removed = int(shared.sum())
        if not n_removed:
            return 0
        keep = np.concatenate(([True], ~shared))

        if getattr(self, 'species_list', None):
            kept_species = [s for s, kept in zip(self._row_species(), keep) if kept]
            species_list, species_index_map = [], []
            for row, species in enumerate(kept_species):
                if row == 0 or species != kept_species[row - 1]:
                    species_list.append(species)
                    species_index_map.append(row)
            self.species_list = species_list
            self.species_index_map = species_index_map
            self.species_list_map = {species: i for i, species in enumerate(species_list)}

        self.matrix = np.ascontiguousarray(self.matrix[keep])
        if getattr(self, 'weights', None) is not None:
            self.weights, self.Beff = None, None
        logger.debug(f"Removed {n_removed} shared-source sequences; {self.matrix.shape[0]} left")
        return n_removed

    def Mask_Gap_Columns(self, gap_ratio=0.5):
        """
        Before this call, saved_columns is a list of per-monomer masks.
        After this call, it becomes a list of per-monomer per-column index maps
        that reflect the gap ratio control. The alignment matrix is not modified.
        Any existing mask is ignored and recomputed from scratch.
        If available, monomer boundaries are derived from lengthes.

        The gap ratio is calculated as the number of gaps in a column divided
        by the total number of sequences. Columns with a gap ratio higher than
        the specified threshold will be excluded from the saved_columns mask.

        Parameters
        ----------
        `gap_ratio` — float (optional)
            Gap ratio in columns (default: `0.5`)
        """
        L, n_cols = self.matrix.shape

        # Ignore any prior mask: all columns are eligible
        if self.lengthes:
            total_len = int(np.sum(self.lengthes))
            if total_len == n_cols:
                block_offsets = [0]
                cursor = 0
                for block_len in self.lengthes[:-1]:
                    cursor += int(block_len)
                    block_offsets.append(cursor)
            else:
                logger.warning(
                    "lengthes sum (%d) does not match alignment width (%d); treating as single monomer",
                    total_len,
                    n_cols,
                )
                block_offsets = [0]
        else:
            # No per-monomer lengths available: treat as single monomer
            block_offsets = [0]

        col_gap_ratios = np.sum(self.matrix == 0, axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < gap_ratio)[0]

        # Rebuild saved_columns as per-monomer per-column maps
        block_ends = block_offsets[1:] + [n_cols]
        new_saved_columns = [[] for _ in range(len(block_offsets))]
        for col in cols_to_keep:
            for block_idx, (start, end) in enumerate(zip(block_offsets, block_ends)):
                if start <= col < end:
                    new_saved_columns[block_idx].append(int(col - start))
                    break

        self.saved_columns = new_saved_columns

    def Saved_Columns_To_Bitmask(self):
        """
        Encode `saved_columns` as the bitmask an msv record stores.

        One bit per column of each monomer, which reproduces the retained-column
        lists exactly in a fraction of the space the JSON attribute took. The
        per-monomer lengths ride along in the blob, so a reader can restore the
        residue numbering without the alignment.

        Returns
        -------
        `bytes`
            The bitmask blob, or `b""` when this alignment has no saved columns
        """
        saved = getattr(self, 'saved_columns', None)
        if not saved:
            return b""

        lengthes = [int(length) for length in (self.lengthes or [])]
        if not lengthes:
            # No per-monomer split recorded: treat the alignment as one monomer
            # spanning the whole matrix.
            if len(saved) != 1:
                logger.warning(
                    "saved_columns describes %d monomers but lengthes is empty; "
                    "cannot build a bitmask", len(saved))
                return b""
            lengthes = [int(self.matrix.shape[1])]
        if len(saved) != len(lengthes):
            logger.warning(
                "saved_columns describes %d monomers but lengthes has %d; "
                "cannot build a bitmask", len(saved), len(lengthes))
            return b""

        try:
            return saved_columns_to_bitmask(saved, lengthes)
        except ValueError as e:
            logger.warning(f"Could not build a saved-column bitmask: {e}")
            return b""

    def Drop_Masked_Gap_Columns(self):
        """
        Shrink `matrix` to the columns that `saved_columns` keeps.

        The sequence weights can be slightly different after dropping columns.
        Call `Reweight_Sequence` when needed.
        """
        saved = getattr(self, 'saved_columns', None)
        if not saved or not self.lengthes or len(saved) != len(self.lengthes):
            logger.warning(
                "saved_columns does not describe this alignment; keeping every column")
            return

        columns = []
        offset = 0
        for local_cols, length in zip(saved, self.lengthes):
            length = int(length)
            for col in local_cols:
                col = int(col)
                if not 0 <= col < length:
                    logger.warning(
                        "saved_columns holds column %d for a monomer of length %d; "
                        "keeping every column", col, length)
                    return
                columns.append(offset + col)
            offset += length

        n_cols = self.matrix.shape[1]
        if n_cols != offset:
            if n_cols == len(columns):
                logger.debug("Alignment is already reduced to its saved columns")
            else:
                logger.warning(
                    "lengthes sum (%d) does not match the alignment width (%d); "
                    "keeping every column", offset, n_cols)
            return
        if len(columns) == n_cols:
            logger.debug("The saved columns cover the whole alignment; nothing to drop")
            return

        self.matrix = np.ascontiguousarray(self.matrix[:, columns])
        self._pair_source = None                # runs of the reduced columns differ
        logger.info(f'Alignment reduced: {n_cols} -> {self.matrix.shape[1]} columns')

    def Downsample_Randomly(self, to):
        if to >= self.matrix.shape[0]:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.matrix.shape[0]} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.matrix.shape[0]), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)  # Ensure the first row is included
        self.matrix = self.matrix[chosen_indices]
        self._pair_source = None

        # Update metadata to reflect the downsampled alignment
        kept_count = len(chosen_indices)
        new_species_list = ['Query']
        new_species_index_map = list(range(kept_count))

        for i in range(kept_count-1):
            new_species_list.append(self.species_list[chosen_indices[i + 1]])

        self.species_list = new_species_list
        self.species_index_map = new_species_index_map
        self.species_list_map = {species: i for i, species in enumerate(new_species_list)}

    def Reweight_Sequence(self, x=0.8, use_GPU=False, n_threads=0):
        B, N = self.matrix.shape
        identical_threshold = x * N

        if use_GPU:
            if not gpu_is_available():
                logger.error("CuPy is not available. Please install CuPy to use GPU acceleration.")
                raise ImportError("CuPy is not available. Please install CuPy to use GPU acceleration.")
            import cupy as cp

            # The kernel compares four residues at a time
            Nw = (N + 3) // 4
            padN = Nw * 4
            MSA = cp.zeros((B, padN), dtype=cp.int8)
            MSA[:, :N] = cp.asarray(self.matrix, dtype=cp.int8)
            simM = cp.ones(B, dtype=cp.int32)   # every sequence matches itself

            TILE, TY = 32, 8                    # must match RW_TILE / RW_TY
            ntile = math.ceil(B / TILE)
            cuda_module = get_cuda_module()
            pairwise_similarity = cuda_module.get_function('pairwise_similarity_tiled')
            pairwise_similarity(
                (ntile, ntile), (TILE, TY),
                (MSA.view(cp.uint32), simM, np.int32(B), np.int32(Nw),
                 np.int32(padN - N), np.float32(identical_threshold)))
            cp.cuda.runtime.deviceSynchronize()

            m = cp.asnumpy(simM).astype(np.int64)
            del MSA, simM
        else:
            m = np.asarray(
                cpp_bindings.reweightNeighbourCounts(
                    np.ascontiguousarray(self.matrix, dtype=np.int8),
                    identical_threshold, int(n_threads)),
                dtype=np.int64)

        w = 1.0 / m
        self.weights = w.tolist()
        self.Beff = float(w.sum())

    def To_Text(self, filename=None, restore_gaps=True):
        # Create reverse mapping from numbers back to characters
        valid_chars = 'acdefghiklmnpqrstvwy-ACDEFGHIKLMNPQRSTVWY'
        values = np.arange(-20, 21, dtype=np.int8)
        reverse_mapping = np.full(128, ord('X'), dtype=np.uint8)
        for char, value in zip(valid_chars, values):
            reverse_mapping[value + 20] = ord(char)

        # Convert numeric matrix back to character matrix
        ascii_codes = reverse_mapping[self.matrix + 20]
        char_matrix = np.frombuffer(ascii_codes.tobytes(), dtype='S1').reshape(self.matrix.shape)
        if restore_gaps:
            monomer_matrices = []
            monomer_widths = []
            cursor = 0
            for monomer_idx, monomer_saved_cols in enumerate(self.saved_columns):
                n = len(monomer_saved_cols)
                monomer_char_matrix = char_matrix[:, cursor:cursor + n]
                cursor += n

                if self.lengthes and monomer_idx < len(self.lengthes):
                    monomer_original_length = int(self.lengthes[monomer_idx])
                elif monomer_saved_cols:
                    monomer_original_length = max(monomer_saved_cols) + 1
                else:
                    monomer_original_length = 0

                monomer_original_matrix = np.full((self.matrix.shape[0], monomer_original_length), b'-', dtype='S1')
                if monomer_saved_cols:
                    monomer_original_matrix[:, monomer_saved_cols] = monomer_char_matrix
                monomer_matrices.append(monomer_original_matrix)
                monomer_widths.append(monomer_original_length)

            original_matrix = np.concatenate(monomer_matrices, axis=1)
        else:
            original_matrix = char_matrix.copy()
            n_cols = char_matrix.shape[1]
            saved = self.saved_columns or []
            if self.lengthes and sum(int(x) for x in self.lengthes) == n_cols:
                monomer_widths = [int(x) for x in self.lengthes]
            elif saved and sum(len(c) for c in saved) == n_cols:
                monomer_widths = [len(c) for c in saved]
            else:
                monomer_widths = [n_cols]
        original_matrix = original_matrix.astype('U1')

        # Convert matrix back to SeqRecord objects
        seq_records = []
        i = 0
        for species, starting_idx in zip(self.species_list[:-1], self.species_index_map[1:]):
            for _ in range(i, starting_idx):
                row = original_matrix[i]
                sequence_str = ''.join(row.tolist())
                seq_record = SeqRecord(
                    Seq(sequence_str),
                    id=str(i),
                    name=str(i) + ' OX=' + str(species),
                    description='OX=' + str(species)
                )
                seq_records.append(seq_record)
                i += 1
        species = self.species_list[-1]
        for _ in range(i, original_matrix.shape[0]):
            row = original_matrix[i]
            sequence_str = ''.join(row.tolist())
            seq_record = SeqRecord(
                Seq(sequence_str),
                id=str(i),
                name=str(i) + ' OX=' + str(species),
                description='OX=' + str(species)
            )
            seq_records.append(seq_record)
            i += 1

        if filename:
            SeqIO.write(seq_records, filename, "fasta")

        text = TextAlignment(seq_records)
        if len(monomer_widths) > 1:
            text.lengthes = monomer_widths
        return text

    def To_Zarr(self, filename=None, overwrite=False):
        if not filename: filename = 'merged_alignment'
        if os.path.exists(os.path.join(filename, 'align', 'zarr.json')):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                shutil.rmtree(os.path.join(filename, 'align'), ignore_errors=True)
                RunTable.remove_all(filename)      # built for the old matrix
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        grp = zarr.open_group(store=filename)
        # Store all-gap chunks too (and say so), so BinaryAlignment can treat any missing chunk as corruption
        z = grp.create_array(name='align', shape=self.matrix.shape, dtype='int8',
                             config={'write_empty_chunks': True})
        z.attrs.update({
            'saved_columns': getattr(self, 'saved_columns', None),
            'species_list': getattr(self, 'species_list', None),
            'species_index_map': getattr(self, 'species_index_map', None),
            'lengthes': getattr(self, 'lengthes', [self.matrix.shape[1]]),
            'weights': getattr(self, 'weights', None),
            'Beff': getattr(self, 'Beff', None),
            'all_chunks_stored': True
        })

        z[:] = self.matrix

    def To_Npy(self, filename=None):
        if filename:
            np.save(filename, self.matrix)
        return self.matrix




