import numpy as np
import pandas as pd

from prepare_data import clean
from run_al import append_checkpoint, entropy, entropy_diversity, initial_and_random_order, mean_pairwise_similarity, seeded_rank, select_seeds
from study import canonicalize, fingerprint, rdkit2d


def test_fingerprints_are_deterministic_binary_1024():
    first = fingerprint("C[C@H](O)Cl")
    second = fingerprint("C[C@H](O)Cl")
    assert first.dtype == np.uint8
    assert first.shape == (1024,)
    assert set(np.unique(first)) <= {0, 1}
    np.testing.assert_array_equal(first, second)


def test_rdkit2d_is_deterministic_and_finite():
    first, names = rdkit2d("C[C@H](O)Cl")
    second, second_names = rdkit2d("C[C@H](O)Cl")
    assert len(first) == len(names) > 200
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(names, second_names)
    np.testing.assert_array_equal(first, second)


def test_cleaning_invalid_identical_and_conflicting_duplicates():
    raw = pd.DataFrame({
        "Drug": ["CCO", "OCC", "CCN", "NCC", "C", "not smiles", "Cl"],
        "Y": [1, 1, 0, 1, 0, 0, 1],
    })
    cleaned, stats = clean(raw)
    assert len(cleaned) == 3
    assert stats == {
        "raw_rows": 7, "invalid_rows_removed": 1,
        "identical_duplicate_rows_collapsed": 1,
        "conflicting_duplicate_groups": 1,
        "conflicting_duplicate_rows_removed": 2,
    }
    assert cleaned["canonical_smiles"].is_unique
    assert canonicalize("CCN") not in set(cleaned["canonical_smiles"])


def test_initial_set_is_reproducible_and_has_both_classes():
    labels = np.array([0] * 50 + [1] * 50)
    a, random_a = initial_and_random_order(labels, 4, 32)
    b, random_b = initial_and_random_order(labels, 4, 32)
    assert set(labels[a]) == {0, 1}
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(random_a, random_b)


def test_entropy_selects_analytically_most_uncertain():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    ranked = seeded_rank(np.arange(5), entropy(p), seed=0)
    assert ranked[0] == 2
    assert set(ranked[1:3]) == {1, 3}


def test_hybrid_stays_in_shortlist_and_improves_toy_diversity():
    # Labeled point and candidates are binary fingerprints with deliberately
    # redundant high-entropy candidates and two diverse shortlist members.
    fps = np.array([
        [1, 0, 0, 0],  # labeled
        [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0],
    ], dtype=np.uint8)
    unlabeled = np.arange(1, 7)
    probabilities = np.array([0.50, 0.50, 0.50, 0.49, 0.48, 0.01])
    shortlist = set(seeded_rank(unlabeled, entropy(probabilities), 7)[:5])
    chosen, _, _ = entropy_diversity(
        unlabeled, probabilities, fps, np.array([0]), batch_size=3,
        shortlist_size=5, seed=7,
    )
    entropy_batch = seeded_rank(unlabeled, entropy(probabilities), 7)[:3]
    assert set(chosen) <= shortlist
    assert len(set(chosen)) == 3
    assert mean_pairwise_similarity(fps[chosen]) < mean_pairwise_similarity(fps[entropy_batch])


def test_checkpoint_replaces_a_retried_trajectory(tmp_path):
    path = tmp_path / "metrics.csv"
    key = {"dataset": "DILI", "seed": 0, "model": "rf", "acquisition": "random"}
    append_checkpoint(path, pd.DataFrame([key | {"budget": 32, "score": 0.1}]))
    append_checkpoint(path, pd.DataFrame([key | {"budget": 32, "score": 0.2}]))
    saved = pd.read_csv(path)
    assert len(saved) == 1
    assert saved.iloc[0]["score"] == 0.2


def test_smoke_defaults_to_one_seed_but_honors_explicit_seeds():
    configured = [0, 1, 2]
    assert select_seeds(None, configured, smoke=True) == [0]
    assert select_seeds(None, configured, smoke=False) == configured
    assert select_seeds([2], configured, smoke=True) == [2]
