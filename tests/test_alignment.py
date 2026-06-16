import os
from massivora import BinaryAlignment, TextAlignment

TEST1_A2M = os.path.join(os.path.dirname(__file__), "test1.a2m")
TEST2_A2M = os.path.join(os.path.dirname(__file__), "test2.a2m")

TEST1_ZARR = os.path.join(os.path.dirname(__file__), "test1.zarr")
TEST2_ZARR = os.path.join(os.path.dirname(__file__), "test2.zarr")

def test_alignment_process():
    alignment1 = TextAlignment(TEST1_A2M)
    alignment1.Filtering_MSA_Gap(50, 50)
    assert alignment1.matrix.shape == (65082, 708), f"Align1 gap filtering failed, got shape {alignment1.matrix.shape}"
    alignment1.Filtering_MSA_Invalid()
    assert alignment1.matrix.shape == (64682, 708), f"Align1 gap filtering failed, got shape {alignment1.matrix.shape}"
    alignment1.Best_Reciprocal_Hit()
    assert alignment1.matrix.shape == (594, 708), f"Align1 gap filtering failed, got shape {alignment1.matrix.shape}"
    alignment1.To_Zarr(TEST1_ZARR, overwrite=True)
    alignment2 = TextAlignment(TEST2_A2M)
    alignment2.Filtering_MSA_Gap(50, 50)
    assert alignment2.matrix.shape == (6232, 454), f"Align2 gap filtering failed, got shape {alignment2.matrix.shape}"
    alignment2.Filtering_MSA_Invalid()
    assert alignment2.matrix.shape == (5940, 454), f"Align2 gap filtering failed, got shape {alignment2.matrix.shape}"
    alignment2.Best_Reciprocal_Hit()
    assert alignment2.matrix.shape == (448, 454), f"Align2 gap filtering failed, got shape {alignment2.matrix.shape}"
    alignment2.To_Zarr(TEST2_ZARR, overwrite=True)
    concat = alignment1 + alignment2
    assert concat.matrix.shape == (318, 1162), f"Concatenation failed with shape {concat.matrix.shape}"

def test_alignment_concatenation():
    zarr1 = BinaryAlignment(TEST1_ZARR)
    assert zarr1.matrix.shape == (594, 708), f"Align1 Zarr loading failed, got shape {zarr1.matrix.shape}"
    zarr2 = BinaryAlignment(TEST2_ZARR)
    assert zarr2.matrix.shape == (448, 454), f"Align2 Zarr loading failed, got shape {zarr2.matrix.shape}"
    merged_alignment = zarr1 + zarr2
    assert merged_alignment.matrix.shape == (318, 1162), f"Concatenation failed with shape {merged_alignment.matrix.shape}"
    merged_alignment.Reweight_Sequence(x=0.8)
    assert round(merged_alignment.Beff, 2) == 41.12, f"Reweighting failed, expected Beff=41.12, got {merged_alignment.Beff}"
    merged_alignment.To_Zarr(os.path.join(os.path.dirname(__file__), "merged_test.zarr"), overwrite=True)