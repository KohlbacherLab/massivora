"""Checks for the msv coupling-score shards.

Each pair's scores are appended to a large shard file as a self-describing
record: a 32-byte header opening with MSV_MAGIC, then the pair name, the
saved-column bitmask and the uncompressed float16 upper triangle. The writer
hands back (score_file_id, offset, len) for the database, and when that index is
lost the shard is scanned for the marker instead. Most failures here would be
silent -- a column numbered one off, a matrix read back subtly wrong, a record
skipped or reported twice where a scan region was cut -- so the tests pin the
byte layout and those invariants.

These build their own fixtures, so they run on a bare checkout.
"""
import errno
import logging
import os
import signal
import sqlite3
import struct
import zlib

import numpy as np
import pandas as pd
import pytest

from massivora import utils
from massivora.alignment import BinaryAlignment
from massivora.db import STATUS, create_job_db
from massivora.utils import (MSV_HEADER, MSV_HEADER_SIZE, MSV_MAGIC, MSV_MAX_SHARDS,
                             ShardWriter, bitmask_to_saved_columns,
                             matrix_from_upper_triangle, pack_score_record,
                             read_score_record, saved_columns_to_bitmask, scan_shard,
                             scan_shard_headers, shard_name, shard_path,
                             unpack_score_record, upper_triangle_bytes)


def _symmetric(n, seed=0):
    """A symmetric float16 matrix of coupling-score-like values."""
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n, n)) * 0.16).astype(np.float16)
    return np.triu(a) + np.triu(a, 1).T


def _write_shard(path, records):
    path.write_bytes(b''.join(records))
    return str(path)


def _flip_byte(path, at):
    with open(path, 'r+b') as f:
        f.seek(at)
        byte = f.read(1)[0]
        f.seek(at)
        f.write(bytes([byte ^ 0xFF]))


def _headers(path, verify):
    """What scan_shard finds, cut down to the fields the header reader returns."""
    return [(offset, length, record['name'], record['n'])
            for offset, length, record in scan_shard(path, verify=verify)]


def _cut(path, cuts):
    """Scan a shard as consecutive regions split at `cuts`, as the worker pool does."""
    size = os.path.getsize(path)
    bounds = [0] + sorted({c for c in cuts if 0 < c < size}) + [size]
    found = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        found.extend(utils._scan_header_region(path, start, stop))
    return found


def _markers(data):
    found, at = [], data.find(MSV_MAGIC)
    while at >= 0:
        found.append(at)
        at = data.find(MSV_MAGIC, at + 1)
    return found


# ---------------------------------------------------------------------------
# Shard names
# ---------------------------------------------------------------------------

def test_shard_names_carry_the_prefix_and_three_digits(tmp_path):
    assert shard_name('couplings', 0) == 'couplings_000.msv'
    assert shard_name('my_couplings', 123) == 'my_couplings_123.msv'
    assert shard_path(str(tmp_path), 'couplings', 7) == os.path.join(str(tmp_path), 'couplings_007.msv')


@pytest.mark.parametrize('file_id', [-1, MSV_MAX_SHARDS])
def test_shard_numbers_beyond_three_digits_are_refused(file_id):
    with pytest.raises(ValueError, match='three digits'):
        shard_name('couplings', file_id)


# ---------------------------------------------------------------------------
# Saved-column bitmask
# ---------------------------------------------------------------------------

def test_bitmask_layout():
    """Monomer count, each monomer's full length, then one mask per monomer with
    column c at bit c % 8 of byte c // 8."""
    blob = saved_columns_to_bitmask([[0, 2, 3, 9], [1, 4]], [10, 6])

    assert blob == bytes.fromhex('0200' '0a000000' '06000000' '0d02' '12')
    assert bitmask_to_saved_columns(blob) == ([[0, 2, 3, 9], [1, 4]], [10, 6])


def test_bitmask_keeps_a_dropped_last_column():
    """A dropped last column is a zero bit, which looks exactly like padding:
    only the stored length tells a 5-column monomer from a 4-column one."""
    blob = saved_columns_to_bitmask([[0, 1, 2, 3]], [5])

    assert bitmask_to_saved_columns(blob) == ([[0, 1, 2, 3]], [5])


@pytest.mark.parametrize('saved_columns, lengthes, expected', [
    ([[9, 3, 3, 0]], [10], [[0, 3, 9]]),                        # sorted, repeats gone
    ([(5, 1), np.array([2, 0])], [8, 4], [[1, 5], [0, 2]]),     # tuples and arrays in
    ([[], [0]], [3, 1], [[], [0]]),                             # a monomer keeping nothing
    ([[], [1]], [0, 2], [[], [1]]),                             # a monomer of length 0
    ([], [], []),                                               # no monomers at all
])
def test_bitmask_round_trip_returns_sorted_lists_of_ints(saved_columns, lengthes, expected):
    decoded, decoded_lengthes = bitmask_to_saved_columns(
        saved_columns_to_bitmask(saved_columns, lengthes))

    assert decoded == expected
    assert decoded_lengthes == lengthes
    assert all(type(col) is int for cols in decoded for col in cols)


@pytest.mark.parametrize('saved_columns, lengthes, message', [
    ([[0]], [1, 2], 'monomers'),        # counts disagree
    ([[5]], [5], 'column 5'),           # a column outside its monomer
    ([[]], [-1], 'negative'),
])
def test_bitmask_refuses_what_it_cannot_encode(saved_columns, lengthes, message):
    with pytest.raises(ValueError, match=message):
        saved_columns_to_bitmask(saved_columns, lengthes)


def test_truncated_bitmask_is_refused():
    blob = saved_columns_to_bitmask([[0, 2], [1]], [3, 2])

    with pytest.raises(ValueError, match='truncated'):
        bitmask_to_saved_columns(blob[:-1])
    # Cut inside the lengths, struct raises before any mask is reached.
    with pytest.raises((ValueError, struct.error)):
        bitmask_to_saved_columns(blob[:5])


@pytest.mark.parametrize('blob', [b'', None])
def test_missing_bitmask_decodes_to_nothing(blob):
    assert bitmask_to_saved_columns(blob) == ([], [])


# ---------------------------------------------------------------------------
# Upper triangle
# ---------------------------------------------------------------------------

def test_upper_triangle_is_row_major_with_the_diagonal():
    matrix = np.array([[1, 2, 3],
                       [2, 4, 5],
                       [3, 5, 6]], dtype=np.float16)

    assert upper_triangle_bytes(matrix) == np.arange(1, 7, dtype=np.float16).tobytes()


@pytest.mark.parametrize('n', [1, 2, 7, 64])
def test_upper_triangle_round_trip_rebuilds_the_matrix(n):
    matrix = _symmetric(n, seed=n)

    payload = upper_triangle_bytes(matrix)
    rebuilt = matrix_from_upper_triangle(payload, n)

    assert len(payload) == n * (n + 1)
    assert rebuilt.dtype == np.float16
    np.testing.assert_array_equal(rebuilt, matrix)


@pytest.mark.parametrize('shape', [(2, 3), (4,), (2, 2, 2)])
def test_upper_triangle_needs_a_square_matrix(shape):
    with pytest.raises(ValueError, match='square'):
        upper_triangle_bytes(np.zeros(shape, dtype=np.float16))


def test_payload_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match='needs'):
        matrix_from_upper_triangle(upper_triangle_bytes(_symmetric(3)), 4)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def test_record_layout():
    """Byte for byte: header, name, bitmask, then the triangle the crc covers."""
    matrix = np.array([[1.0, 0.5, -2.0],
                       [0.5, 3.0, 0.25],
                       [-2.0, 0.25, 4.0]], dtype=np.float16)
    bitmask = saved_columns_to_bitmask([[0], [0, 1]], [1, 2])
    triangle = np.array([1.0, 0.5, -2.0, 3.0, 0.25, 4.0], dtype=np.float16).tobytes()

    record = pack_score_record('A-B', matrix, bitmask)

    header = struct.pack('<4sHHIIIQI', bytes.fromhex('4d5356ff'), 1, 0, 3,
                         3, len(bitmask), len(triangle), zlib.crc32(triangle))
    assert MSV_HEADER_SIZE == 32
    assert record == header + b'A-B' + bitmask + triangle


@pytest.mark.parametrize('name', [
    'sp|P0MSV|RL4_ECOLI',       # contains the old marker's text
    'MSV|MSV|MSV|',
    'MSVÿ',                     # U+00FF encodes as C3 BF, never as a bare FF
    'x' + chr(0x10FFFF),
])
def test_a_name_never_contains_the_marker(name):
    """0xFF does not occur in UTF-8, so the only marker is the record's own."""
    record = pack_score_record(name, np.zeros((3, 3), dtype=np.float16))

    assert record.count(MSV_MAGIC) == 1


def test_unpack_returns_what_was_packed():
    matrix = _symmetric(12)
    bitmask = saved_columns_to_bitmask([[0, 3, 5], [0, 1, 4]], [6, 6])

    record = unpack_score_record(pack_score_record('café-B', matrix, bitmask))

    assert (record['name'], record['n'], record['bitmask']) == ('café-B', 12, bitmask)
    assert (record['saved_columns'], record['lengthes']) == ([[0, 3, 5], [0, 1, 4]], [6, 6])
    np.testing.assert_array_equal(record['matrix'], matrix)


@pytest.mark.parametrize('damage, message', [
    (lambda r: r[:3] + b'|' + r[4:], 'does not start with'),
    (lambda r: r[:4] + b'\x09\x00' + r[6:], 'version'),
    (lambda r: r[:6] + b'\x01\x00' + r[8:], 'dtype'),
    (lambda r: r[:-1] + bytes([r[-1] ^ 0xFF]), 'Checksum'),
    (lambda r: r[:-2], 'truncated'),
    (lambda r: r[:MSV_HEADER_SIZE - 1], 'shorter'),
], ids=['old-marker', 'version', 'dtype', 'payload', 'truncated', 'short'])
def test_unpack_refuses_a_damaged_record(damage, message):
    record = pack_score_record('A-B', _symmetric(5))

    with pytest.raises(ValueError, match=message):
        unpack_score_record(damage(record))


# ---------------------------------------------------------------------------
# ShardWriter and reading back by database location
# ---------------------------------------------------------------------------

def test_writer_hands_back_where_each_record_landed(tmp_path):
    records = [('a', _symmetric(1)), ('bb', _symmetric(6)), ('ccc', _symmetric(25))]
    writer = ShardWriter(str(tmp_path), 'couplings')
    assert writer.file_id is None

    with writer:
        locations = [writer.append(name, matrix) for name, matrix in records]

    sizes = [len(pack_score_record(name, matrix)) for name, matrix in records]
    assert locations == [(0, 0, sizes[0]),
                         (0, sizes[0], sizes[1]),
                         (0, sizes[0] + sizes[1], sizes[2])]
    assert os.path.getsize(shard_path(str(tmp_path), 'couplings', 0)) == sum(sizes)


def test_reading_by_database_location_returns_the_record_as_written(tmp_path):
    matrix = _symmetric(5)
    bitmask = saved_columns_to_bitmask([[0, 2], [1]], [3, 2])
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        writer.append('before', _symmetric(9))
        location = writer.append('A-B', matrix, bitmask)

    record = read_score_record(str(tmp_path), 'couplings', *location)

    assert (record['name'], record['saved_columns'], record['lengthes']) == \
        ('A-B', [[0, 2], [1]], [3, 2])
    assert record['matrix'].dtype == np.float16
    np.testing.assert_array_equal(record['matrix'], matrix)


def test_reading_a_damaged_payload_fails_its_checksum(tmp_path):
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        file_id, offset, length = writer.append('A-B', _symmetric(5))
    _flip_byte(shard_path(str(tmp_path), 'couplings', file_id), offset + length - 1)

    with pytest.raises(ValueError, match='Checksum'):
        read_score_record(str(tmp_path), 'couplings', file_id, offset, length)
    unchecked = read_score_record(str(tmp_path), 'couplings', file_id, offset, length, verify=False)
    assert unchecked['name'] == 'A-B'


def test_reading_past_the_end_of_a_shard_is_refused(tmp_path):
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        file_id, offset, length = writer.append('A-B', _symmetric(5))

    with pytest.raises(ValueError, match='expected'):
        read_score_record(str(tmp_path), 'couplings', file_id, offset, length + 1)
    with pytest.raises(FileNotFoundError):
        read_score_record(str(tmp_path), 'couplings', file_id + 1, offset, length)


def test_writer_rolls_over_before_a_shard_would_pass_max_bytes(tmp_path):
    matrix = _symmetric(8)
    size = len(pack_score_record('p0', matrix))

    # Two records fill a shard exactly; only a third would take it past the limit.
    with ShardWriter(str(tmp_path), 'couplings', max_bytes=2 * size) as writer:
        locations = [writer.append(f'p{i}', matrix) for i in range(5)]

    assert [(file_id, offset) for file_id, offset, _ in locations] == \
        [(0, 0), (0, size), (1, 0), (1, size), (2, 0)]
    assert sorted(os.listdir(tmp_path)) == \
        ['couplings_000.msv', 'couplings_001.msv', 'couplings_002.msv']


def test_a_record_larger_than_max_bytes_still_gets_written(tmp_path):
    """Rolling over cannot make room for an oversized record, so it goes into a
    fresh shard on its own rather than rolling forever."""
    with ShardWriter(str(tmp_path), 'couplings', max_bytes=10) as writer:
        locations = [writer.append(f'p{i}', _symmetric(4)) for i in range(2)]

    assert [(file_id, offset) for file_id, offset, _ in locations] == [(0, 0), (1, 0)]


def test_writers_never_share_a_shard(tmp_path):
    """Each writer locks the shard it holds, so a second writer moves on to the
    next one, while an empty file nobody holds is simply used."""
    (tmp_path / 'couplings_000.msv').touch()
    first = ShardWriter(str(tmp_path), 'couplings')
    second = ShardWriter(str(tmp_path), 'couplings')

    try:
        first_id = first.append('a', _symmetric(3))[0]
        second_id = second.append('b', _symmetric(3))[0]
    finally:
        first.close()
        second.close()

    assert (first_id, second_id) == (0, 1)


def test_writer_gives_up_when_every_shard_number_is_taken(tmp_path):
    """Every number is in use by a shard without room for the record."""
    for file_id in range(MSV_MAX_SHARDS):
        (tmp_path / shard_name('couplings', file_id)).write_bytes(b'x')
    writer = ShardWriter(str(tmp_path), 'couplings',
                         max_bytes=len(pack_score_record('a', _symmetric(3))))

    with pytest.raises(RuntimeError, match='taken'):
        writer.append('a', _symmetric(3))


# ---------------------------------------------------------------------------
# Continuing a shard nobody holds
# ---------------------------------------------------------------------------

_HOLD_SHARD = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.flock(fd, fcntl.LOCK_EX)
print('held', flush=True)
sys.stdin.read()
"""


def _names(path):
    return [name for _, _, name, _ in scan_shard_headers(str(path))]


def test_the_next_writer_continues_the_shard_the_last_one_closed(tmp_path):
    """One coordinator run after another fills the same shard, with nothing
    written beside it."""
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        a = writer.append('a', _symmetric(4))
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        b = writer.append('b', _symmetric(6))

    assert b == (0, a[2], len(pack_score_record('b', _symmetric(6))))
    assert _names(shard_path(str(tmp_path), 'couplings', 0)) == ['a', 'b']
    assert os.listdir(tmp_path) == ['couplings_000.msv']


def test_a_shard_another_process_holds_is_left_alone_until_it_lets_go(tmp_path):
    """A coordinator on another node keeps its shard locked while it writes."""
    import subprocess
    import sys
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        a = writer.append('a', _symmetric(4))
    holder = subprocess.Popen([sys.executable, '-c', _HOLD_SHARD,
                               shard_path(str(tmp_path), 'couplings', 0)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'held'
        with ShardWriter(str(tmp_path), 'couplings') as writer:
            while_held = writer.append('b', _symmetric(4))
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        after = writer.append('c', _symmetric(4))

    assert while_held[:2] == (1, 0)
    assert after[:2] == (0, a[2])


def test_a_free_shard_goes_to_exactly_one_of_two_writers(tmp_path):
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        writer.append('a', _symmetric(4))
    first = ShardWriter(str(tmp_path), 'couplings')
    second = ShardWriter(str(tmp_path), 'couplings')

    try:
        ids = (first.append('b', _symmetric(4))[0], second.append('c', _symmetric(4))[0])
    finally:
        first.close()
        second.close()

    assert ids == (0, 1)


def test_a_killed_jobs_shard_is_continued_after_cutting_its_half_record(tmp_path):
    """Two whole records, then the front of a third the job died writing. The
    kernel dropped the job's lock, so the shard is free; the half record has to
    go first, or a header walk could not see past it to what is appended."""
    whole = [pack_score_record('a', _symmetric(4)), pack_score_record('b', _symmetric(5))]
    torn = pack_score_record('c', _symmetric(9))
    path = _write_shard(tmp_path / 'couplings_000.msv', whole + [torn[:len(torn) // 2]])
    end = sum(map(len, whole))

    with ShardWriter(str(tmp_path), 'couplings') as writer:
        location = writer.append('d', _symmetric(4))

    assert location[:2] == (0, end)
    assert os.path.getsize(path) == end + location[2]
    assert _names(path) == ['a', 'b', 'd']


def test_a_cleanly_closed_shard_is_continued_without_walking_its_headers(tmp_path, monkeypatch):
    """The record that ends the file settles it; a 100 GiB shard holds around a
    hundred thousand headers."""
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        for i in range(5):
            writer.append(f'p{i}', _symmetric(6, seed=i))

    def no_walk(*args, **kwargs):
        raise AssertionError('walked the headers of a shard that ends on a whole record')

    monkeypatch.setattr(utils, 'scan_shard_headers', no_walk)
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        location = writer.append('q', _symmetric(6))

    assert location[0] == 0


def test_a_whole_last_record_is_kept_even_when_its_payload_is_damaged(tmp_path):
    """A row may point at it, and the checksum catches it when read; only bytes
    past the last whole record are ever cut."""
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        writer.append('a', _symmetric(4))
        b = writer.append('b', _symmetric(5))
    path = shard_path(str(tmp_path), 'couplings', 0)
    _flip_byte(path, b[1] + b[2] - 1)

    with ShardWriter(str(tmp_path), 'couplings') as writer:
        c = writer.append('c', _symmetric(4))

    assert c[:2] == (0, b[1] + b[2])
    assert _names(path) == ['a', 'b', 'c']


def test_a_failed_write_is_trimmed_when_the_writer_closes(tmp_path, monkeypatch):
    """A short write leaves the front of a record behind, which nothing points at."""
    pwrite = os.pwrite
    path = shard_path(str(tmp_path), 'couplings', 0)
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        a = writer.append('a', _symmetric(4))
        monkeypatch.setattr(utils.os, 'pwrite',
                            lambda fd, data, offset: pwrite(fd, data[:len(data) // 2], offset))
        with pytest.raises(IOError, match='Short write'):
            writer.append('torn', _symmetric(9))
        monkeypatch.setattr(utils.os, 'pwrite', pwrite)
        assert os.path.getsize(path) > a[2]

    assert os.path.getsize(path) == a[2]
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        b = writer.append('b', _symmetric(4))
    assert b[:2] == (0, a[2])
    assert _names(path) == ['a', 'b']


def test_a_record_that_does_not_fit_goes_elsewhere_and_leaves_the_room(tmp_path):
    small = len(pack_score_record('s0', _symmetric(3)))
    big = len(pack_score_record('b0', _symmetric(30)))
    max_bytes = small + big - 1             # past one small record, a big one no longer fits
    with ShardWriter(str(tmp_path), 'couplings', max_bytes=max_bytes) as writer:
        writer.append('s0', _symmetric(3))

    with ShardWriter(str(tmp_path), 'couplings', max_bytes=max_bytes) as writer:
        big_at = writer.append('b0', _symmetric(30))
    with ShardWriter(str(tmp_path), 'couplings', max_bytes=max_bytes) as writer:
        small_at = writer.append('s1', _symmetric(3))

    assert big_at[:2] == (1, 0)
    assert small_at[:2] == (0, small)


def test_without_file_locks_every_writer_keeps_to_new_shards(tmp_path, monkeypatch, caplog):
    """No lock means no proof a shard is free, so none is continued."""
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        writer.append('a', _symmetric(4))

    def unavailable(fd, operation):
        raise OSError(errno.ENOLCK, 'No locks available')

    monkeypatch.setattr(utils.fcntl, 'flock', unavailable)
    with caplog.at_level(logging.WARNING):
        with ShardWriter(str(tmp_path), 'couplings') as writer:
            first = writer.append('b', _symmetric(4))
            second = writer.append('c', _symmetric(4))

    assert first[:2] == (1, 0) and second[:2] == (1, first[2])
    assert 'File locks are unavailable' in caplog.text


def test_writer_creates_its_output_directory(tmp_path):
    output_dir = tmp_path / 'nested' / 'scores'

    with ShardWriter(str(output_dir), 'couplings') as writer:
        writer.append('a', _symmetric(3))

    assert os.listdir(output_dir) == ['couplings_000.msv']


# ---------------------------------------------------------------------------
# scan_shard: recovering whole records without the database
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def damaged_shard(tmp_path_factory):
    """A shard mixing intact records with the damage a recovery scan has to get
    past: bad checksums, a bad header, junk carrying markers and a fake header,
    stray markers inside real records, and a truncated tail.

    Returns the shard path with the names scan_shard should report with and
    without checksum verification, in file order.
    """
    rng = np.random.default_rng(7)
    records, intact, unverified = [], [], []

    def add(record, name=None, checksum_ok=True):
        records.append(record)
        if name is not None:
            unverified.append(name)
            if checksum_ok:
                intact.append(name)

    for i in range(60):
        matrix = _symmetric(int(rng.integers(1, 40)), seed=i)
        kind = i % 10
        if kind == 1:                           # the old marker's text in a name
            add(pack_score_record(f'sp|P{i:05d}MSV|X', matrix), f'sp|P{i:05d}MSV|X')
        elif kind == 2:                         # marker bytes inside the bitmask
            bitmask = struct.pack('<HI', 1, 64) + b'\x01\x02' + MSV_MAGIC + b'\x03\x04'
            add(pack_score_record(f'mask{i}', matrix, bitmask), f'mask{i}')
        elif kind == 3:                         # marker straddling name and payload
            matrix[0, 0] = np.frombuffer(b'\xff\x3c', dtype='<f2')[0]
            add(pack_score_record(f'name{i}MSV', matrix), f'name{i}MSV')
        elif kind == 4:                         # payload damaged
            record = bytearray(pack_score_record(f'payload{i}', matrix))
            record[-1] ^= 0xFF
            add(bytes(record), f'payload{i}', checksum_ok=False)
        elif kind == 5:                         # header damaged
            record = bytearray(pack_score_record(f'header{i}', matrix))
            record[4] = 9
            add(bytes(record))
        elif kind == 6:                         # junk with markers and a fake header
            junk = bytearray(rng.integers(0, 256, 120, dtype=np.uint8).tobytes())
            for at in (5, 40, 90):
                junk[at:at + 4] = MSV_MAGIC
            add(bytes(junk) + MSV_HEADER.pack(MSV_MAGIC, 1, 0, 10, 3, 0, 999, 0))
            add(pack_score_record(f'after_junk{i}', matrix), f'after_junk{i}')
        elif kind == 7:                         # checksum zeroed
            record = bytearray(pack_score_record(f'checksum{i}', matrix))
            record[28:32] = bytes(4)
            add(bytes(record), f'checksum{i}', checksum_ok=False)
        elif kind == 8:
            add(pack_score_record(f'café-ÿ{i}', matrix), f'café-ÿ{i}')
        else:
            add(pack_score_record(f'P{i:05d}-Q{i:05d}', matrix), f'P{i:05d}-Q{i:05d}')
    tail = pack_score_record('truncated', _symmetric(20))
    add(tail[:len(tail) // 2])

    path = _write_shard(tmp_path_factory.mktemp('damaged') / 'couplings_000.msv', records)
    return path, intact, unverified


def test_scan_shard_yields_every_matrix_where_the_writer_put_it(tmp_path):
    matrices = {f'P{i:05d}-Q{i:05d}': _symmetric(n, seed=i) for i, n in enumerate([1, 2, 17, 120])}
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        locations = [writer.append(name, matrix) for name, matrix in matrices.items()]

    found = list(scan_shard(shard_path(str(tmp_path), 'couplings', 0)))

    assert [(offset, length) for offset, length, _ in found] == \
        [(offset, length) for _, offset, length in locations]
    for _, _, record in found:
        np.testing.assert_array_equal(record['matrix'], matrices[record['name']])


def test_scan_shard_skips_damage_and_stray_markers(damaged_shard):
    path, intact, _ = damaged_shard

    assert [record['name'] for _, _, record in scan_shard(path)] == intact


def test_scan_shard_without_verification_keeps_checksum_failures(damaged_shard):
    path, _, unverified = damaged_shard

    assert [record['name'] for _, _, record in scan_shard(path, verify=False)] == unverified


def test_scan_shard_recovers_the_records_around_an_overwritten_stretch(tmp_path):
    matrix = _symmetric(30)
    with ShardWriter(str(tmp_path), 'couplings') as writer:
        locations = [writer.append(f'pair{i}', matrix) for i in range(5)]
    path = shard_path(str(tmp_path), 'couplings', 0)
    _, offset, length = locations[2]
    with open(path, 'r+b') as f:
        f.seek(offset + length // 2)
        f.write(b'\xde\xad\xbe\xef' * 16)

    recovered = list(scan_shard(path))

    assert [record['name'] for _, _, record in recovered] == ['pair0', 'pair1', 'pair3', 'pair4']
    for _, _, record in recovered:
        np.testing.assert_array_equal(record['matrix'], matrix)


@pytest.mark.parametrize('content', [
    b'',
    MSV_MAGIC + b'junk',                                    # shorter than a header
    pack_score_record('cut', _symmetric(6))[:-1],           # last byte missing
    b'MSV|' + pack_score_record('old', _symmetric(6))[4:],  # written with the old marker
], ids=['empty', 'short', 'truncated', 'old-marker'])
def test_no_whole_record_means_nothing_is_found(tmp_path, content):
    path = tmp_path / 'couplings_000.msv'
    path.write_bytes(content)

    assert list(scan_shard(str(path))) == []
    assert scan_shard_headers(str(path)) == []


# ---------------------------------------------------------------------------
# scan_shard_headers: rebuilding the index across regions
# ---------------------------------------------------------------------------

def test_header_reader_finds_what_scan_shard_finds_without_verification(damaged_shard):
    path, _, _ = damaged_shard

    assert scan_shard_headers(path) == _headers(path, verify=False)


def test_header_reader_lists_records_whose_payload_is_damaged(damaged_shard):
    """It never reads payloads, so a bad checksum only shows up on reading."""
    path, intact, unverified = damaged_shard

    names = [name for _, _, name, _ in scan_shard_headers(path)]

    assert names == unverified
    assert set(unverified) - set(intact)


def test_header_scan_rebuilds_the_locations_the_writer_handed_back(tmp_path):
    widths = np.random.default_rng(3).integers(5, 40, 30)
    with ShardWriter(str(tmp_path), 'couplings', max_bytes=4000) as writer:
        written = {f'pair{i}': writer.append(f'pair{i}', _symmetric(int(n), seed=i))
                   for i, n in enumerate(widths)}

    shards = sorted(entry for entry in os.listdir(tmp_path) if entry.endswith('.msv'))
    rebuilt = {}
    for file_id in range(len(shards)):
        for offset, length, name, _ in scan_shard_headers(shard_path(str(tmp_path), 'couplings', file_id)):
            rebuilt[name] = (file_id, offset, length)

    assert len(shards) > 1
    assert rebuilt == written


@pytest.mark.parametrize('regions', [2, 3, 7, 16, 64])
def test_each_record_is_reported_once_however_the_shard_is_cut(damaged_shard, regions):
    path, _, _ = damaged_shard
    size = os.path.getsize(path)

    found = _cut(path, [size * i // regions for i in range(1, regions)])

    assert found == _headers(path, verify=False)


def test_cuts_at_and_inside_every_marker_lose_and_repeat_nothing(damaged_shard):
    """A cut landing mid-marker, mid-header or right on a record boundary is where
    a region could claim a record twice or let one fall between two regions."""
    path, _, _ = damaged_shard
    expected = _headers(path, verify=False)
    with open(path, 'rb') as f:
        data = f.read()
    markers = _markers(data)
    ends = [offset + length for offset, length, _, _ in expected]
    cuts = ({m + d for m in markers for d in (*range(-4, 5), 31, 32, 33)}
            | {e + d for e in ends for d in (-1, 0, 1)})

    wrong = sorted(c for c in cuts if 0 < c < len(data) and _cut(path, [c]) != expected)

    assert len(markers) > len(expected)     # stray markers are really in play
    assert wrong == []


def test_a_record_spanning_many_regions_is_reported_once(tmp_path):
    records = [pack_score_record(f's{i}', _symmetric(6, seed=i)) for i in range(3)]
    records.append(pack_score_record('wide', _symmetric(700)))
    records += [pack_score_record(f't{i}', _symmetric(6, seed=i)) for i in range(3)]
    path = _write_shard(tmp_path / 'couplings_000.msv', records)
    expected = _headers(path, verify=True)
    size = os.path.getsize(path)

    wrong = [k for k in range(2, 80) if _cut(path, [size * i // k for i in range(1, k)]) != expected]

    assert len(expected) == 7
    assert wrong == []


def test_large_shards_are_split_across_worker_processes(damaged_shard, monkeypatch):
    path, _, _ = damaged_shard
    started = []

    class RecordingPool(utils.ProcessPoolExecutor):
        def __init__(self, max_workers):
            started.append(max_workers)
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(utils, 'ProcessPoolExecutor', RecordingPool)
    monkeypatch.setattr(utils, 'MSV_SCAN_MIN_REGION_BYTES', 1)     # treat this shard as large

    assert scan_shard_headers(path, workers=4) == _headers(path, verify=False)
    assert started == [4]


def test_small_shards_are_scanned_without_a_pool(damaged_shard, monkeypatch):
    path, _, _ = damaged_shard

    def refuse(**kwargs):
        raise AssertionError('a worker pool was started for a shard under the region threshold')

    monkeypatch.setattr(utils, 'ProcessPoolExecutor', refuse)

    assert scan_shard_headers(path, workers=8) == _headers(path, verify=False)


# ---------------------------------------------------------------------------
# Where the bitmask and the shard location meet the rest of massivora
# ---------------------------------------------------------------------------

def _alignment(width, saved_columns, lengthes):
    a = BinaryAlignment()
    a.matrix = np.zeros((2, width), dtype=np.int8)
    a.saved_columns = saved_columns
    a.lengthes = lengthes
    return a


def test_alignment_bitmask_decodes_to_its_saved_columns():
    a = _alignment(7, [[0, 2, 4], [1]], [5, 2])

    assert bitmask_to_saved_columns(a.Saved_Columns_To_Bitmask()) == ([[0, 2, 4], [1]], [5, 2])


def test_alignment_without_lengthes_is_one_monomer_spanning_the_matrix():
    a = _alignment(6, [[0, 3, 5]], [])

    assert bitmask_to_saved_columns(a.Saved_Columns_To_Bitmask()) == ([[0, 3, 5]], [6])


@pytest.mark.parametrize('saved_columns, lengthes', [
    ([], [5, 2]),                       # nothing saved
    ([[0], [1]], []),                   # two monomers but no split recorded
    ([[0, 1]], [4, 3]),                 # fewer mask entries than monomers
    ([[0], [0, 9]], [5, 2]),            # a column outside its monomer
])
def test_alignment_bitmask_is_empty_on_unusable_metadata(saved_columns, lengthes):
    assert _alignment(7, saved_columns, lengthes).Saved_Columns_To_Bitmask() == b''


@pytest.fixture
def complex_analyzer():
    # The analyzer imports OpenStructure, which is conda-only and can be installed
    # yet fail to load; either way these checks skip instead of erroring.
    pytest.importorskip('ost', exc_type=ImportError)
    from massivora.analyzer import ComplexAnalyzer
    return ComplexAnalyzer()


def test_analyzer_restores_residue_numbers_from_the_bitmask(complex_analyzer):
    saved_columns, lengthes = [[0, 2, 3], [1, 4]], [4, 5]
    block = pd.DataFrame(np.arange(20, dtype=float).reshape(4, 5),
                         index=range(1, 5), columns=range(1, 6))

    labelled = complex_analyzer.FilterMaskedByBitmask(
        saved_columns_to_bitmask(saved_columns, lengthes), block)

    assert list(labelled.index) == [1, 3, 4]
    assert list(labelled.columns) == [2, 5]
    pd.testing.assert_frame_equal(
        labelled, complex_analyzer.FilterMaskedBySavedColumns(saved_columns, block))


def test_analyzer_needs_two_monomers_in_the_bitmask(complex_analyzer):
    with pytest.raises(RuntimeError, match='two monomers'):
        complex_analyzer.FilterMaskedByBitmask(saved_columns_to_bitmask([[0, 1]], [2]), pd.DataFrame())


def test_database_keeps_file_hash_beside_the_score_location(tmp_path):
    """Keeping every MSA as zarr is still supported, so file_hash stays next to
    the shard location rather than being replaced by it."""
    create_job_db({'project': {'project_path': str(tmp_path)},
                   'database': {'backend': 'sqlite', 'filename': 'job.db'}})

    conn = sqlite3.connect(tmp_path / 'job.db')
    columns = {row[1]: row[2] for row in conn.execute('PRAGMA table_info(couplings)')}
    conn.close()

    assert columns['file_hash'] == 'BLOB'
    assert [columns[c] for c in ('score_file_id', 'score_offset', 'score_len')] == ['INTEGER'] * 3


# ---------------------------------------------------------------------------
# Coupling executors: storing each pair's score in msv shards
# ---------------------------------------------------------------------------

_PAIR_MASK = ([[0, 2, 4], [1, 3]], [5, 4])      # monomer A 5 wide, B 4 wide


@pytest.fixture
def couple():
    pytest.importorskip('ost', exc_type=ImportError)    # the executors import the analyzer
    from massivora.executors import couple
    return couple


@pytest.fixture
def keep_signals():
    """The coordinator installs SIGINT/SIGTERM handlers; put pytest's back afterwards."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _couple_project(tmp_path, n_pairs=3, **couple_cfg):
    """A project config and job database holding `n_pairs` coupling jobs ready to be claimed."""
    cfg = {
        'project': {'project_path': str(tmp_path)},
        'paths': {'monomers': 'msa', 'couplings': 'ecs'},
        'database': {'backend': 'sqlite', 'filename': 'job.db'},
        'align': {'db_table': 'alignments', 'col_gap_threshold': 0.5},
        'concatenate': {'downsample': {'enabled': False, 'to': 30000}},
        'couple': {'db_table': 'couplings', 'claim_batch_size': 8, **couple_cfg},
    }
    create_job_db(cfg)
    _, lengthes = _PAIR_MASK
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.executemany('INSERT INTO alignments (pid, length) VALUES (?, ?)',
                     [(f'A{i}', lengthes[0]) for i in range(n_pairs)]
                     + [(f'B{i}', lengthes[1]) for i in range(n_pairs)])
    conn.executemany('INSERT INTO couplings (name, pid1, pid2, status, length) VALUES (?, ?, ?, ?, ?)',
                     [(f'A{i}-B{i}', f'A{i}', f'B{i}', STATUS['NOOPT'], sum(lengthes))
                      for i in range(n_pairs)])
    conn.commit()
    conn.close()
    return cfg


def _couplings(tmp_path):
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.row_factory = sqlite3.Row
    rows = {row['name']: dict(row) for row in conn.execute('SELECT * FROM couplings')}
    conn.close()
    return rows


def _fake_plm_executor(couple, cfg, fail=()):
    """A coordinator-driven executor whose tasks skip the optimizer and return a result directly."""
    from dask import delayed

    class FakeExecutor(couple.BaseCouplingExecutor):
        def collect(self, pair):
            if pair['name'] in fail:
                raise RuntimeError('optimizer failed')
            saved, lengthes = _PAIR_MASK
            params = {'name': pair['name'], 'saved_columns': saved, 'lengthes': lengthes}
            return {'id': pair['id'], 'effnumber': 12.5, 'length': sum(lengthes), 'number': 40,
                    'file_hash': None, 'pcontact': 0.25,
                    **self._score_fields(_symmetric(sum(lengthes), seed=pair['id']), params)}

        def calc_graph(self, pair, priority=0, **kwargs):
            return delayed(self.collect)(pair)

    return FakeExecutor(cfg)


def _check_stored(tmp_path, rows):
    """Each row points at a shard record holding its pair's score and saved columns."""
    saved, lengthes = _PAIR_MASK
    for name, row in rows.items():
        record = read_score_record(str(tmp_path / 'ecs'), 'couplings', row['score_file_id'],
                                   row['score_offset'], row['score_len'])
        assert record['name'] == name
        assert (record['saved_columns'], record['lengthes']) == (saved, lengthes)
        np.testing.assert_array_equal(record['matrix'], _symmetric(sum(lengthes), seed=row['id']))


def test_coordinator_stores_scores_and_records_where_they_went(tmp_path, couple, keep_signals):
    from dask.distributed import Client
    cfg = _couple_project(tmp_path, n_pairs=4)
    executor = _fake_plm_executor(couple, cfg, fail={'A3-B3'})

    with Client(processes=False, dashboard_address=None) as client:
        executor.run(client=client)

    rows = _couplings(tmp_path)
    done = {name: row for name, row in rows.items() if row['status'] == STATUS['DONE']}
    assert sorted(done) == ['A0-B0', 'A1-B1', 'A2-B2']
    assert all(row['pcontact'] == 0.25 for row in done.values())
    _check_stored(tmp_path, done)
    failed = rows['A3-B3']
    # Claimed again within the run until its retries ran out, and never stored.
    assert failed['status'] == STATUS['FAILED']
    assert (failed['score_file_id'], failed['score_offset'], failed['score_len']) == (None, None, None)
    assert os.listdir(tmp_path / 'ecs') == ['couplings_000.msv']
    shard = str(tmp_path / 'ecs' / 'couplings_000.msv')
    assert sorted(name for _, _, name, _ in scan_shard_headers(shard)) == sorted(done)


def test_shards_are_flushed_before_the_rows_pointing_at_them_commit(tmp_path, couple, keep_signals,
                                                                  monkeypatch):
    """A committed row must never point at bytes a crash could still lose."""
    from dask.distributed import Client
    cfg = _couple_project(tmp_path, n_pairs=3)
    executor = _fake_plm_executor(couple, cfg)
    calls = []
    for method in ('_flush_scores', '_db_apply_results'):
        real = getattr(executor, method)
        monkeypatch.setattr(executor, method,
                            lambda *a, _real=real, _name=method: (calls.append(_name), _real(*a))[1])

    with Client(processes=False, dashboard_address=None) as client:
        executor.run(client=client)

    commits = [i for i, call in enumerate(calls) if call == '_db_apply_results']
    assert commits
    assert all(i > 0 and calls[i - 1] == '_flush_scores' for i in commits)


@pytest.mark.parametrize('jij_cfg', [
    {},                                                                 # nothing asked for
    {'keep_coupling_matrix': True},                                     # keep the Jij tensor
    {'keep_coupling_matrix': True, 'coupling_matrix_keep_option': 'top'},
])
def test_scores_are_stored_whatever_is_asked_of_the_jij_tensor(tmp_path, couple, keep_signals,
                                                               jij_cfg):
    """`keep_coupling_matrix` and its option are about Jij; CN scores are kept either way."""
    from dask.distributed import Client
    cfg = _couple_project(tmp_path, n_pairs=2, **jij_cfg)
    executor = _fake_plm_executor(couple, cfg)

    with Client(processes=False, dashboard_address=None) as client:
        executor.run(client=client)

    rows = _couplings(tmp_path)
    assert all(row['status'] == STATUS['DONE'] and row['pcontact'] == 0.25
               for row in rows.values())
    _check_stored(tmp_path, rows)


def test_keeping_the_jij_tensor_warns_that_nothing_stores_it_yet(tmp_path, couple, caplog):
    cfg = _couple_project(tmp_path, n_pairs=1, keep_coupling_matrix=True)

    with caplog.at_level(logging.WARNING):
        couple.BaseCouplingExecutor(cfg).close_db_connection()

    assert 'Jij' in caplog.text and 'not implemented' in caplog.text


def test_an_asymmetric_matrix_keeps_its_upper_triangle_and_warns(tmp_path, couple, caplog):
    cfg = _couple_project(tmp_path, n_pairs=1)
    executor = couple.BaseCouplingExecutor(cfg)
    score = _symmetric(9)
    score[7, 2] += np.float16(0.5)
    saved, lengthes = _PAIR_MASK

    try:
        with caplog.at_level(logging.WARNING):
            result = executor._store_score({'id': 1, 'name': 'A0-B0', 'score': score,
                                            'bitmask': saved_columns_to_bitmask(saved, lengthes)})
    finally:
        executor._close_scores()
        executor.close_db_connection()

    assert 'not exactly symmetric' in caplog.text
    record = read_score_record(str(tmp_path / 'ecs'), 'couplings', result['score_file_id'],
                               result['score_offset'], result['score_len'])
    np.testing.assert_array_equal(record['matrix'], np.triu(score) + np.triu(score, 1).T)


def _fake_gauss_cuda(couple, monkeypatch, score=None, free_mb=1024,
                     pair_bytes=lambda N, B, q=21: 1 << 20):
    """Stand-in for the gauss_cuda module: `score(MSA)` replaces the device pipeline."""
    import sys
    import types

    class DeviceLostError(RuntimeError):
        pass

    class Session:
        last_pair_bytes = 0

        def __init__(self, ordinal):
            self.ordinal = ordinal

        def score(self, MSA, W, q, pseudocount):
            if score is not None:
                return score(MSA)
            return _symmetric(MSA.shape[0], seed=int(MSA[0, 0])).astype(np.float32)

        def neighbour_counts(self, MSA, identical_threshold):
            from massivora import cpp_bindings      # the counts the device computes
            Session.reweighted.append(int(MSA[0, 0]))
            return np.asarray(cpp_bindings.reweightNeighbourCounts(
                np.ascontiguousarray(MSA.T), identical_threshold, 1), dtype=np.int32)

    Session.reweighted = []
    fake = types.SimpleNamespace(
        Session=Session, DeviceLostError=DeviceLostError, pair_bytes=pair_bytes,
        device_memory=lambda d: {'free_mb': free_mb, 'total_mb': free_mb})
    monkeypatch.setitem(sys.modules, 'massivora.cpp_bindings.gauss_cuda', fake)
    monkeypatch.setattr(couple.cpp_bindings, 'gauss_cuda', fake, raising=False)
    monkeypatch.setattr(couple, 'visible_gpu_devices', lambda: ['0'])
    return fake


def _fake_gpu_executor(couple, monkeypatch, cfg, depth=lambda pair: 4):
    """A GPU GaussDCA executor whose pairs are `depth(pair)` rows filled with the pair id."""
    saved, lengthes = _PAIR_MASK
    executor = couple.GaussCouplingExecutorGPU(cfg)
    executor.gpu_host_processes = 0     # the stand-ins below cannot be sent to a process

    def load_pair_arrays(pair, **kwargs):
        N, B = sum(lengthes), depth(pair)
        return (np.full((N, B), pair['id'], dtype=np.int8), np.ones(B),
                {'name': pair['name'], 'pair_id': pair['id'], 'B': B, 'N': N, 'q': 21,
                 'Beff': float(B), 'file_hash': None, 'saved_columns': saved,
                 'lengthes': lengthes})

    monkeypatch.setattr(executor, 'load_pair_arrays', load_pair_arrays)
    monkeypatch.setattr(executor, 'Calibrate', lambda analyzer, n_eff: {'pcontact': 0.25})
    return executor


def test_gpu_gaussdca_threads_store_their_scores(tmp_path, couple, monkeypatch):
    """GPU GaussDCA scores pairs in threads of the coordinator itself, not in Dask workers."""
    cfg = _couple_project(tmp_path, n_pairs=6, gpu_sessions=1, gpu_workers=3)
    _fake_gauss_cuda(couple, monkeypatch)
    executor = _fake_gpu_executor(couple, monkeypatch, cfg)

    executor.run()

    rows = _couplings(tmp_path)
    assert all(row['status'] == STATUS['DONE'] for row in rows.values())
    _check_stored(tmp_path, rows)
    assert os.listdir(tmp_path / 'ecs') == ['couplings_000.msv']


def _wait_for_waiters(admission, n):
    import time
    deadline = time.monotonic() + 5
    while len(admission._waiting) < n and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(admission._waiting) == n


def test_gpu_admission_lets_a_fitting_pair_past_one_waiting_for_memory(couple):
    import threading
    admission = couple.GpuAdmission([100], sessions=4)
    assert admission.acquire(60) == 0
    big = threading.Thread(target=admission.acquire, args=(60,))
    big.start()
    _wait_for_waiters(admission, 1)

    assert admission.acquire(10) == 0       # fits now: the GPU need not idle while 60 drains

    admission.release(0, 60)
    big.join(5)
    assert not big.is_alive()


def test_gpu_admission_stops_overtaking_a_pair_that_waited_too_long(couple):
    import threading
    admission = couple.GpuAdmission([100], sessions=4, patience=0)
    assert admission.acquire(60) == 0
    granted = []

    def take(nbytes):
        admission.acquire(nbytes)
        granted.append(nbytes)

    big = threading.Thread(target=take, args=(60,))
    big.start()
    _wait_for_waiters(admission, 1)
    small = threading.Thread(target=take, args=(10,))
    small.start()
    _wait_for_waiters(admission, 2)
    assert granted == []            # 10 would fit, but the 60 ahead of it has run out of patience

    admission.release(0, 60)
    big.join(5)
    small.join(5)
    assert granted == [60, 10]


def test_gpu_admission_needs_an_idle_session_too(couple):
    import threading
    admission = couple.GpuAdmission([100], sessions=1)
    assert admission.acquire(10) == 0
    second = threading.Thread(target=admission.acquire, args=(10,))
    second.start()
    _wait_for_waiters(admission, 1)         # memory to spare, but no session

    admission.release(0, 10)
    second.join(5)
    assert not second.is_alive()


def test_gpu_admission_picks_the_emptiest_gpu(couple):
    admission = couple.GpuAdmission([100, 50], sessions=4)
    assert admission.acquire(80) == 0
    assert admission.acquire(40) == 1       # 20 left on GPU 0, 50 on GPU 1
    with pytest.raises(ValueError):
        admission.acquire(101)              # more than any GPU will ever have


def test_stage_clock_reports_where_pairs_spend_their_time(couple, caplog):
    clock = couple.StageClock(sessions=2, every=0)
    with caplog.at_level(logging.INFO):
        clock.add(load=0.5, wait=0.25, gpu=1.0, analysis=0.125)

    line = caplog.records[-1].getMessage()
    assert '1 pairs' in line
    assert 'load 0.500 s' in line and 'waiting for a GPU 0.250 s' in line
    assert 'on the GPU 1.000 s' in line and 'analysis 0.125 s' in line


def test_gpu_gaussdca_keeps_pairs_within_the_device_budget(tmp_path, couple, monkeypatch):
    """Two sessions, but room on the device for one pair at a time."""
    import threading
    import time
    cfg = _couple_project(tmp_path, n_pairs=6, gpu_sessions=2, gpu_workers=4,
                          gpu_memory_limit=1.0)
    lock = threading.Lock()
    running, peak = [0], [0]

    def score(MSA):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        return _symmetric(MSA.shape[0], seed=int(MSA[0, 0])).astype(np.float32)

    _fake_gauss_cuda(couple, monkeypatch, score=score, free_mb=100,
                     pair_bytes=lambda N, B, q=21: 60 * 2**20)
    executor = _fake_gpu_executor(couple, monkeypatch, cfg)

    executor.run()

    assert all(row['status'] == STATUS['DONE'] for row in _couplings(tmp_path).values())
    assert peak[0] == 1


def test_gpu_gaussdca_fails_a_pair_too_big_for_any_gpu(tmp_path, couple, monkeypatch):
    cfg = _couple_project(tmp_path, n_pairs=3, gpu_sessions=1, gpu_workers=2)
    scored = []

    def score(MSA):
        scored.append(int(MSA[0, 0]))
        return _symmetric(MSA.shape[0], seed=int(MSA[0, 0])).astype(np.float32)

    _fake_gauss_cuda(couple, monkeypatch, score=score, free_mb=1024,
                     pair_bytes=lambda N, B, q=21: B * 2**20)
    executor = _fake_gpu_executor(couple, monkeypatch, cfg,
                                  depth=lambda pair: 2000 if pair['name'] == 'A1-B1' else 4)

    executor.run()

    rows = _couplings(tmp_path)
    assert rows['A1-B1']['status'] == STATUS['FAILED']
    assert rows['A0-B0']['status'] == rows['A2-B2']['status'] == STATUS['DONE']
    assert rows['A1-B1']['id'] not in scored        # never reached a session


def test_gpu_gaussdca_loads_and_analyses_in_host_processes(tmp_path, couple, monkeypatch):
    """The real load_pair_arrays and collect_score, run in spawned processes."""
    from massivora.db import get_hashed_file_path
    cfg = _couple_project(tmp_path, n_pairs=4, gpu_sessions=1, gpu_workers=4,
                          gpu_host_processes=2)
    saved, lengthes = _PAIR_MASK
    for i in range(4):
        align = BinaryAlignment()
        B = 50
        align.matrix = np.random.default_rng(i).integers(0, 21, (B, sum(lengthes)),
                                                         dtype=np.int8)
        align.matrix[0, 0] = i + 1          # the fake session seeds its score with this (= id)
        align.species_list = ['Query'] + [f's{k}' for k in range(1, B)]
        align.species_index_map = list(range(B))
        align.saved_columns, align.lengthes = saved, lengthes
        align.Reweight_Sequence(0.8)
        align.To_Zarr(get_hashed_file_path(f'A{i}-B{i}', None, str(tmp_path / 'ecs'))[0])
    _fake_gauss_cuda(couple, monkeypatch)
    executor = couple.GaussCouplingExecutorGPU(cfg)
    assert executor.gpu_host_processes == 2

    executor.run()

    rows = _couplings(tmp_path)
    assert all(row['status'] == STATUS['DONE'] for row in rows.values())
    assert all(row['number'] == 50 and row['pcontact'] is not None for row in rows.values())
    _check_stored(tmp_path, rows)


def test_gpu_gaussdca_reweights_on_the_gpu_and_stores_the_weights(tmp_path, couple,
                                                                  monkeypatch):
    """An alignment stored without weights is reweighted by its session, not by
    the host processes, and the store ends up as the CPU path would leave it."""
    from massivora.db import get_hashed_file_path
    cfg = _couple_project(tmp_path, n_pairs=3, gpu_sessions=1, gpu_workers=3,
                          gpu_host_processes=2)
    saved, lengthes = _PAIR_MASK
    want, files = {}, {}
    for i in range(3):
        B = 60
        matrix = np.random.default_rng(i).integers(0, 21, (B, sum(lengthes)), dtype=np.int8)
        matrix[0, 0] = i + 1                    # the fake session seeds its score with this (= id)
        matrix[B // 2:] = matrix[:B // 2]       # duplicates, so the weights are not all 1
        ref = BinaryAlignment()
        ref.matrix = matrix.copy()
        ref.Reweight_Sequence(0.8)
        want[f'A{i}-B{i}'] = (ref.weights, ref.Beff)
        align = BinaryAlignment()
        align.matrix = matrix
        align.species_list = ['Query'] + [f's{k}' for k in range(1, B)]
        align.species_index_map = list(range(B))
        align.saved_columns, align.lengthes = saved, lengthes
        files[f'A{i}-B{i}'] = get_hashed_file_path(f'A{i}-B{i}', None, str(tmp_path / 'ecs'))[0]
        align.To_Zarr(files[f'A{i}-B{i}'])     # no weights
    fake = _fake_gauss_cuda(couple, monkeypatch)
    executor = couple.GaussCouplingExecutorGPU(cfg)

    executor.run()

    rows = _couplings(tmp_path)
    assert sorted(fake.Session.reweighted) == [1, 2, 3]
    for name, (weights, beff) in want.items():
        assert rows[name]['status'] == STATUS['DONE']
        assert rows[name]['effnumber'] == beff
        stored = BinaryAlignment(files[name])
        assert stored.weights == weights and stored.Beff == beff
    _check_stored(tmp_path, rows)


def test_gpu_gaussdca_hands_back_unrun_pairs_when_the_device_is_lost(tmp_path, couple,
                                                                     monkeypatch):
    """After a sticky CUDA error every later pair would fail; they go back unspent instead."""
    cfg = _couple_project(tmp_path, n_pairs=5, gpu_sessions=1, gpu_workers=1)
    fake = {}

    def score(MSA):
        if MSA[0, 0] == rows_before['A1-B1']['id']:
            raise fake['cuda'].DeviceLostError('an illegal memory access was encountered')
        return _symmetric(MSA.shape[0], seed=int(MSA[0, 0])).astype(np.float32)

    rows_before = _couplings(tmp_path)
    fake['cuda'] = _fake_gauss_cuda(couple, monkeypatch, score=score)
    executor = _fake_gpu_executor(couple, monkeypatch, cfg)

    with pytest.raises(RuntimeError, match='CUDA context was lost'):
        executor.run()

    rows = _couplings(tmp_path)
    assert rows['A0-B0']['status'] == STATUS['DONE']
    assert rows['A1-B1']['status'] == STATUS['PENDING']     # in flight: one attempt spent
    for name in ('A2-B2', 'A3-B3', 'A4-B4'):                 # never run: as if never claimed
        assert rows[name]['status'] == STATUS['NOOPT']
        assert rows[name]['job_id'] is None and rows[name]['claimed_at'] is None


def test_gpu_gaussdca_stops_cleanly_on_sigterm(tmp_path, couple, monkeypatch, caplog):
    """`kill <pid>` finishes the pair on the GPU and hands the rest back, as the CPU coordinator does."""
    import signal
    import threading
    import time
    cfg = _couple_project(tmp_path, n_pairs=5, gpu_sessions=1, gpu_workers=1)
    rows_before = _couplings(tmp_path)
    original = signal.getsignal(signal.SIGTERM)

    def score(MSA):
        if MSA[0, 0] == rows_before['A1-B1']['id']:
            signal.pthread_kill(threading.main_thread().ident, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while signal.getsignal(signal.SIGTERM) is not original:     # handled
                assert time.monotonic() < deadline
                time.sleep(0.01)
        return _symmetric(MSA.shape[0], seed=int(MSA[0, 0])).astype(np.float32)

    _fake_gauss_cuda(couple, monkeypatch, score=score)
    executor = _fake_gpu_executor(couple, monkeypatch, cfg)

    with caplog.at_level(logging.WARNING):
        executor.run()

    rows = _couplings(tmp_path)
    assert rows['A0-B0']['status'] == rows['A1-B1']['status'] == STATUS['DONE']
    for name in ('A2-B2', 'A3-B3', 'A4-B4'):                 # never run: as if never claimed
        assert rows[name]['status'] == STATUS['NOOPT']
        assert rows[name]['job_id'] is None and rows[name]['claimed_at'] is None
    assert 'stopped on a signal' in caplog.text
    assert signal.getsignal(signal.SIGTERM) is original


# ---------------------------------------------------------------------------
# massiveilance: recovering a dead job from the shard it already wrote
# ---------------------------------------------------------------------------

_RECOVERY_PAIR = ([list(range(40)), list(range(30))], [40, 30])
_DEAD_JOB = '999999'


@pytest.fixture
def massiveilance():
    pytest.importorskip('ost', exc_type=ImportError)   # recovery calibrates through the analyzer
    import importlib
    import sys
    importlib.import_module('massivora.massiveilance')
    return sys.modules['massivora.massiveilance']      # the package shadows the module name


def _store_score_record(tmp_path, name, seed=0):
    """Put one pair's score in a shard, the way the coordinator would have."""
    saved, lengthes = _RECOVERY_PAIR
    score = _symmetric(sum(lengthes), seed=seed)
    with ShardWriter(str(tmp_path / 'ecs'), 'couplings') as writer:
        location = writer.append(name, score, saved_columns_to_bitmask(saved, lengthes))
    return score, location


def _claimed_by_a_dead_job(tmp_path, name, status=None, effnumber=812.5):
    """A couplings row left behind by a job that is no longer queued."""
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.executemany('INSERT INTO alignments (pid, length) VALUES (?, ?)',
                     [(f'{name}_A', 40), (f'{name}_B', 30)])
    conn.execute(
        'INSERT INTO couplings (name, pid1, pid2, status, job_id, claimed_at, effnumber) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (name, f'{name}_A', f'{name}_B', STATUS['RUNNING'] if status is None else status,
         _DEAD_JOB, 1758000000.0, effnumber))
    conn.commit()
    conn.close()


def _expected_calibration(score, calibrator_name, n_eff=812.5):
    """What the executor itself would have written for this score."""
    import massivora.analyzer as analyzer_module
    saved, lengthes = _RECOVERY_PAIR
    analyzer = analyzer_module.ComplexAnalyzer(score, saved_columns=saved,
                                               chainA_length=lengthes[0], EC_threshold=0)
    return getattr(analyzer_module, calibrator_name)().Calibrate(analyzer, n_eff)


def test_massiveilance_recovers_a_dead_job_from_its_shard_record(tmp_path, massiveilance,
                                                                 monkeypatch):
    cfg = _couple_project(tmp_path, n_pairs=0)
    score, location = _store_score_record(tmp_path, 'A0-B0')
    _claimed_by_a_dead_job(tmp_path, 'A0-B0')
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [])

    assert massiveilance.fix_zombie_tasks(cfg, 'couple') == 1

    row = _couplings(tmp_path)['A0-B0']
    expected = _expected_calibration(score, 'BuiltinPLMCalibrator')
    assert row['status'] == STATUS['DONE']
    assert (row['score_file_id'], row['score_offset'], row['score_len']) == location
    assert row['length'] == score.shape[0]
    assert row['pcontact'] == pytest.approx(expected['pcontact'])
    assert row['neffoverL'] == pytest.approx(expected['neffoverL'])
    assert row['log_lr_raw'] == pytest.approx(expected['log_lr_raw'])
    assert (row['job_id'], row['claimed_at']) == (None, None)


@pytest.mark.parametrize('status, requeued', [
    (STATUS['RUNNING'], STATUS['NOOPT']),
    (STATUS['RETRYING'], STATUS['PENDING']),
])
def test_massiveilance_requeues_a_dead_job_with_no_record(tmp_path, massiveilance, monkeypatch,
                                                          status, requeued):
    cfg = _couple_project(tmp_path, n_pairs=0)
    _claimed_by_a_dead_job(tmp_path, 'A0-B0', status=status)
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [])

    assert massiveilance.fix_zombie_tasks(cfg, 'couple') == 1

    row = _couplings(tmp_path)['A0-B0']
    assert row['status'] == requeued
    assert (row['score_file_id'], row['pcontact']) == (None, None)


@pytest.mark.parametrize('method, calibrator', [('plm', 'BuiltinPLMCalibrator'),
                                                ('gauss', 'BuiltinGaussCalibrator')])
def test_massiveilance_calibrates_with_the_method_the_run_used(tmp_path, massiveilance, monkeypatch,
                                                               method, calibrator):
    """plmDCA and GaussDCA carry different logistic models, so the config has to decide."""
    cfg = _couple_project(tmp_path, n_pairs=0, method=method)
    score, _ = _store_score_record(tmp_path, 'A0-B0')
    _claimed_by_a_dead_job(tmp_path, 'A0-B0')
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [])

    massiveilance.fix_zombie_tasks(cfg, 'couple')

    row = _couplings(tmp_path)['A0-B0']
    expected = _expected_calibration(score, calibrator)
    other = _expected_calibration(score, 'BuiltinGaussCalibrator' if method == 'plm'
                                  else 'BuiltinPLMCalibrator')
    assert row['pcontact'] == pytest.approx(expected['pcontact'])
    assert expected['pcontact'] != pytest.approx(other['pcontact'])   # the choice really matters


def test_massiveilance_takes_the_last_record_of_a_pair_computed_twice(tmp_path, massiveilance,
                                                                     monkeypatch):
    """A re-queued pair leaves its earlier record behind, so the newer one must win."""
    cfg = _couple_project(tmp_path, n_pairs=0)
    _store_score_record(tmp_path, 'A0-B0', seed=1)
    score, location = _store_score_record(tmp_path, 'A0-B0', seed=2)
    _claimed_by_a_dead_job(tmp_path, 'A0-B0')
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [])

    massiveilance.fix_zombie_tasks(cfg, 'couple')

    row = _couplings(tmp_path)['A0-B0']
    expected = _expected_calibration(score, 'BuiltinPLMCalibrator')
    assert (row['score_file_id'], row['score_offset'], row['score_len']) == location
    assert row['pcontact'] == pytest.approx(expected['pcontact'])


def test_massiveilance_requeues_a_score_it_cannot_calibrate(tmp_path, massiveilance, monkeypatch):
    """A DONE row without pcontact would drop out of every analysis query."""
    cfg = _couple_project(tmp_path, n_pairs=0)
    _store_score_record(tmp_path, 'A0-B0')
    # No effnumber on the row and no concatenated alignment to read Beff from.
    _claimed_by_a_dead_job(tmp_path, 'A0-B0', effnumber=None)
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [])

    massiveilance.fix_zombie_tasks(cfg, 'couple')

    row = _couplings(tmp_path)['A0-B0']
    assert row['status'] == STATUS['NOOPT']
    assert (row['score_file_id'], row['pcontact']) == (None, None)


def test_massiveilance_leaves_pairs_of_live_jobs_alone(tmp_path, massiveilance, monkeypatch):
    cfg = _couple_project(tmp_path, n_pairs=0)
    _store_score_record(tmp_path, 'A0-B0')
    _claimed_by_a_dead_job(tmp_path, 'A0-B0')
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [_DEAD_JOB])

    assert massiveilance.fix_zombie_tasks(cfg, 'couple') == 0

    row = _couplings(tmp_path)['A0-B0']
    assert row['status'] == STATUS['RUNNING'] and row['job_id'] == _DEAD_JOB
