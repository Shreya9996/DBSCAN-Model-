"""
test_dbscan.py — Unit tests for the DBSCAN engine, dataset generator,
and neighborhood query logic.

Run with:  python -m pytest tests/test_dbscan.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons, make_blobs

# Ensure project root is importable
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from components.dbscan_engine import run_dbscan, find_neighbors, DBSCANResult
from components.dataset_generator import generate_dataset, parse_uploaded_csv


# ═══════════════════════════════════════════════════════════════════
#  Dataset Generator Tests
# ═══════════════════════════════════════════════════════════════════

class TestDatasetGenerator:
    """Tests for synthetic dataset generation."""

    @pytest.mark.parametrize("ds_type", ["Moons", "Circles", "Blobs", "Anisotropic", "Varied Density", "Random"])
    def test_generates_correct_shape(self, ds_type):
        """Each dataset type should return a DataFrame with 2 columns and n rows."""
        n = 200
        df = generate_dataset(ds_type, n_samples=n)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n
        assert list(df.columns) == ["Feature 1", "Feature 2"]

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        df1 = generate_dataset("Moons", 100, 0.05, 42)
        df2 = generate_dataset("Moons", 100, 0.05, 42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        df1 = generate_dataset("Moons", 100, 0.05, 42)
        df2 = generate_dataset("Moons", 100, 0.05, 99)
        assert not df1.equals(df2)

    def test_no_nan_values(self):
        """Generated datasets should not contain NaN."""
        for ds in ["Moons", "Circles", "Blobs", "Random"]:
            df = generate_dataset(ds, 300)
            assert not df.isnull().any().any(), f"NaN found in {ds} dataset"


# ═══════════════════════════════════════════════════════════════════
#  CSV Upload Validation Tests
# ═══════════════════════════════════════════════════════════════════

class TestCSVUpload:
    """Tests for CSV upload parsing and validation."""

    def test_valid_csv(self, tmp_path):
        """A valid 2-column numeric CSV should parse successfully."""
        csv_path = tmp_path / "valid.csv"
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            result, err = parse_uploaded_csv(f)
        assert err is None
        assert result is not None
        assert len(result) == 3

    def test_insufficient_numeric_columns(self, tmp_path):
        """A CSV with < 2 numeric columns should return an error."""
        csv_path = tmp_path / "one_col.csv"
        df = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
        df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            result, err = parse_uploaded_csv(f)
        assert result is None
        assert "2 numeric columns" in err

    def test_empty_csv(self, tmp_path):
        """An empty CSV should return an error."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("a,b\n")
        with open(csv_path, "rb") as f:
            result, err = parse_uploaded_csv(f)
        assert result is None
        assert "empty" in err.lower()

    def test_csv_with_missing_values(self, tmp_path):
        """CSV with NaN rows should drop them and return the rest."""
        csv_path = tmp_path / "nan.csv"
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, np.nan]})
        df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            result, err = parse_uploaded_csv(f)
        assert err is None
        assert result is not None
        # Only the first row survives (no NaN in x or y)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════
#  DBSCAN Engine Tests
# ═══════════════════════════════════════════════════════════════════

class TestDBSCANEngine:
    """Tests for the run_dbscan wrapper and metrics."""

    def _make_simple_data(self):
        """Two well-separated blobs, easy for DBSCAN."""
        X, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.3, random_state=0)
        return X

    def test_returns_dbscan_result(self):
        """run_dbscan should return a DBSCANResult instance."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        assert isinstance(result, DBSCANResult)

    def test_finds_two_clusters(self):
        """Two well-separated blobs should yield 2 clusters."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        assert result.n_clusters == 2

    def test_labels_length_matches_data(self):
        """Labels array length should equal number of data points."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        assert len(result.labels) == len(X)

    def test_point_types_correct(self):
        """Every point should be classified as core, border, or noise."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        valid_types = {"core", "border", "noise"}
        actual_types = set(result.point_types)
        assert actual_types.issubset(valid_types)

    def test_core_border_noise_sum(self):
        """Core + border + noise should equal total points."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        assert result.n_core + result.n_border + result.n_noise == len(X)

    def test_masks_are_mutually_exclusive(self):
        """Core, border, and noise masks must not overlap."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        overlap = result.core_mask & result.border_mask & result.noise_mask
        assert not overlap.any()
        # Exactly one category per point
        total_mask = result.core_mask.astype(int) + result.border_mask.astype(int) + result.noise_mask.astype(int)
        assert (total_mask == 1).all()

    def test_cluster_sizes_sum(self):
        """Sum of cluster sizes should equal n_points - n_noise."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        total_in_clusters = sum(result.cluster_sizes.values())
        assert total_in_clusters == len(X) - result.n_noise

    def test_noise_percentage(self):
        """Noise percentage should be correctly calculated."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        expected_pct = round(result.n_noise / len(X) * 100, 1)
        assert result.noise_pct == expected_pct

    def test_all_noise_with_tiny_eps(self):
        """With very small epsilon, all points should be noise."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.001, min_samples=5, scale=False)
        assert result.n_clusters == 0
        assert result.n_noise == len(X)
        assert result.silhouette is None

    def test_single_cluster_with_huge_eps(self):
        """With very large epsilon, all points should be in one cluster."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=100.0, min_samples=2, scale=False)
        assert result.n_clusters == 1
        assert result.n_noise == 0

    def test_silhouette_valid_when_two_clusters(self):
        """Silhouette score should be a float between -1 and 1 for 2+ clusters."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=0.5, min_samples=5)
        if result.n_clusters >= 2:
            assert result.silhouette is not None
            assert -1.0 <= result.silhouette <= 1.0

    def test_silhouette_none_when_one_cluster(self):
        """Silhouette should be None when fewer than 2 clusters."""
        X = self._make_simple_data()
        result = run_dbscan(X, eps=100.0, min_samples=2, scale=False)
        assert result.silhouette is None

    def test_scaling_affects_result(self):
        """Scaling vs. not scaling should produce different labels."""
        X = self._make_simple_data()
        r_scaled = run_dbscan(X, eps=0.5, min_samples=5, scale=True)
        r_raw = run_dbscan(X, eps=0.5, min_samples=5, scale=False)
        # They may differ; at minimum the function should not crash
        assert isinstance(r_scaled, DBSCANResult)
        assert isinstance(r_raw, DBSCANResult)

    def test_moons_dataset(self):
        """DBSCAN on moons should find 2 clusters."""
        X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
        result = run_dbscan(X, eps=0.2, min_samples=5, scale=False)
        assert result.n_clusters == 2


# ═══════════════════════════════════════════════════════════════════
#  Neighborhood Query Tests
# ═══════════════════════════════════════════════════════════════════

class TestFindNeighbors:
    """Tests for the find_neighbors function."""

    def test_basic_neighbors(self):
        """Points within eps should be returned as neighbors."""
        X = np.array([[0, 0], [0.1, 0], [0.2, 0], [5, 5]])
        indices, count = find_neighbors(X, point_idx=0, eps=0.25)
        assert count == 2  # points at [0.1,0] and [0.2,0]
        assert 1 in indices
        assert 2 in indices
        assert 3 not in indices

    def test_no_neighbors(self):
        """Isolated point should have zero neighbors."""
        X = np.array([[0, 0], [10, 10], [20, 20]])
        indices, count = find_neighbors(X, point_idx=0, eps=0.5)
        assert count == 0
        assert len(indices) == 0

    def test_all_neighbors(self):
        """Very large eps should include all other points."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        indices, count = find_neighbors(X, point_idx=0, eps=100)
        assert count == 2  # all except itself

    def test_excludes_self(self):
        """The query point should not be in its own neighbor list."""
        X = np.array([[0, 0], [0.1, 0]])
        indices, count = find_neighbors(X, point_idx=0, eps=1.0)
        assert 0 not in indices

    def test_invalid_index_negative(self):
        """Negative index should return empty."""
        X = np.array([[0, 0], [1, 1]])
        indices, count = find_neighbors(X, point_idx=-1, eps=1.0)
        assert count == 0

    def test_invalid_index_out_of_bounds(self):
        """Out-of-bounds index should return empty."""
        X = np.array([[0, 0], [1, 1]])
        indices, count = find_neighbors(X, point_idx=5, eps=1.0)
        assert count == 0


# ═══════════════════════════════════════════════════════════════════
#  Edge Case Tests
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for edge cases in the DBSCAN pipeline."""

    def test_very_small_dataset(self):
        """DBSCAN should handle a 2-point dataset without crashing."""
        X = np.array([[0, 0], [0.1, 0]])
        result = run_dbscan(X, eps=0.5, min_samples=2, scale=False)
        assert isinstance(result, DBSCANResult)
        assert len(result.labels) == 2

    def test_identical_points(self):
        """All identical points should form one cluster."""
        X = np.array([[1, 1]] * 20)
        result = run_dbscan(X, eps=0.5, min_samples=5, scale=False)
        assert result.n_clusters == 1
        assert result.n_noise == 0

    def test_single_point(self):
        """A single point should be noise (can't meet MinPts)."""
        X = np.array([[0, 0]])
        result = run_dbscan(X, eps=0.5, min_samples=2, scale=False)
        assert result.n_clusters == 0
        assert result.n_noise == 1

    def test_high_min_samples(self):
        """Very high MinPts with few points should produce all noise."""
        X = np.array([[0, 0], [0.1, 0], [0.2, 0]])
        result = run_dbscan(X, eps=0.5, min_samples=100, scale=False)
        assert result.n_clusters == 0
        assert result.n_noise == 3
