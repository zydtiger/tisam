"""Raw-label validity and remapping regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from tisam.data.label_semantics import normalize_policy_raw_mask, raw_label_validity

# Representative supported raw-label ignore policies.
CONFIGURED_IGNORE_LISTS: tuple[list[int], ...] = (
    [],
    [0],
    [0, 23],
    [0, 43],
    [0, 7, 15, 21],
)


@pytest.mark.parametrize("ignored", CONFIGURED_IGNORE_LISTS)
def test_matches_isin_over_the_full_uint8_domain(ignored: list[int]) -> None:
    """Every configured policy must reproduce the former `~np.isin` result exactly."""
    labels = np.arange(256, dtype=np.uint8).reshape(16, 16)

    expected = ~np.isin(labels.astype(np.int64), ignored)

    np.testing.assert_array_equal(raw_label_validity(labels, ignored), expected)


def test_matches_isin_for_signed_labels_including_negative_values() -> None:
    """Signed label planes remain supported because tile readers accept them."""
    labels = np.array([[-5, -1, 0, 1], [7, 15, 21, 100]], dtype=np.int16)
    ignored = [0, 7, 15, 21]

    expected = ~np.isin(labels.astype(np.int64), ignored)

    np.testing.assert_array_equal(raw_label_validity(labels, ignored), expected)


def test_matches_isin_for_ignore_values_outside_the_label_dtype_range() -> None:
    """`tisam_core.metrics` bounds its ignore entries only from below, not above."""
    labels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    ignored = [0, 300, 70000]

    expected = ~np.isin(labels, ignored)

    np.testing.assert_array_equal(raw_label_validity(labels, ignored), expected)


@pytest.mark.parametrize("ignored", CONFIGURED_IGNORE_LISTS)
def test_returns_an_independent_writable_boolean_array(ignored: list[int]) -> None:
    """Callers hand results to separate consumers, so no result may share a buffer."""
    labels = np.arange(256, dtype=np.uint8).reshape(16, 16)

    first = raw_label_validity(labels, ignored)
    second = raw_label_validity(labels, ignored)

    assert first.dtype == np.bool_
    assert first.flags.writeable
    assert first.flags.owndata
    assert first is not second
    assert not np.shares_memory(first, second)

    first[0, 0] = not first[0, 0]
    assert first[0, 0] != second[0, 0]


@pytest.mark.parametrize("ignored", CONFIGURED_IGNORE_LISTS)
def test_does_not_modify_or_widen_the_input(ignored: list[int]) -> None:
    """The scan must run at the caller's stored width and leave the input intact."""
    labels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    original = labels.copy()

    raw_label_validity(labels, ignored)

    assert labels.dtype == np.uint8
    np.testing.assert_array_equal(labels, original)


def test_empty_ignore_list_marks_every_pixel_valid() -> None:
    """An empty policy short-circuits without any per-pixel comparison."""
    labels = np.arange(256, dtype=np.uint8).reshape(16, 16)

    valid = raw_label_validity(labels, [])

    assert valid.shape == labels.shape
    assert bool(valid.all())


# The fixture label space used by the normalization cases below.
NORMALIZE_RAW_LABEL_COUNT = 4


def test_normalize_preserves_dtype_and_returns_uncopied_in_range_masks() -> None:
    """The policy mask must stay at its stored width instead of widening to int64."""
    in_range = np.array([[0, 1, 2, 3]], dtype=np.uint8)

    normalized = normalize_policy_raw_mask(in_range, NORMALIZE_RAW_LABEL_COUNT)

    assert normalized.dtype == np.uint8
    assert normalized is in_range


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
def test_normalize_maps_out_of_range_and_negative_ids_to_raw_label_zero(
    dtype: type[np.signedinteger] | type[np.unsignedinteger],
) -> None:
    """Out-of-range IDs must inherit raw label 0, matching the remapping helper."""
    raw = np.array([[0, 1, 3, 4, 200]], dtype=dtype)
    if np.issubdtype(dtype, np.signedinteger):
        raw = np.concatenate([raw, np.array([[-1, -7, 0, 1, 2]], dtype=dtype)])

    normalized = normalize_policy_raw_mask(raw, NORMALIZE_RAW_LABEL_COUNT)

    assert normalized.dtype == dtype
    expected = raw.astype(np.int64)
    expected[(expected < 0) | (expected >= NORMALIZE_RAW_LABEL_COUNT)] = 0
    np.testing.assert_array_equal(normalized.astype(np.int64), expected)
