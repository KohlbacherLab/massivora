"""Fused screen mode: the coupling stage concatenates pairs itself and, with
couple.keep_concatenated false, couples them from memory without storing them."""
import types

import pytest

from massivora.executors.couple import BaseCouplingExecutor


class _Align:
    def __init__(self):
        self.stored = []

    def To_Zarr(self, filename, overwrite=False):
        self.stored.append(filename)


@pytest.mark.parametrize('keep', [True, False])
def test_missing_pair_is_concatenated_and_stored_only_when_kept(tmp_path, keep):
    align, calls = _Align(), []
    executor = types.SimpleNamespace(
        keep_concatenated=keep,
        concatenate=lambda pair, alignment_dir, downsample, downsample_to:
            calls.append((pair['name'], alignment_dir, downsample, downsample_to)) or align)
    filename = str(tmp_path / 'A-B')
    got = BaseCouplingExecutor._load_or_concatenate(
        executor, {'id': 1, 'name': 'A-B'}, filename,
        {'alignment_dir': 'msa', 'downsample': True, 'downsample_to': 10000})
    assert got is align
    assert calls == [('A-B', 'msa', True, 10000)]
    assert align.stored == ([filename] if keep else [])
