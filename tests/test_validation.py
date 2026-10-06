import pandas as pd
from demandcast.validation import TEST_FOLDS, VAL_FOLDS, fold_masks, make_folds, time_split


def test_folds_are_chronological_and_disjoint():
    dates = pd.Series(pd.date_range("2023-01-01", "2024-12-31"))
    for folds in (VAL_FOLDS, TEST_FOLDS):
        prev_end = None
        for f, tr, te in fold_masks(dates, folds):
            assert dates[tr].max() < dates[te].min()      # train strictly before test
            assert not (tr & te).any()
            if prev_end is not None:
                assert f.test_start == prev_end            # consecutive blocks
            prev_end = f.test_end


def test_validation_precedes_test():
    assert VAL_FOLDS[-1].test_end <= TEST_FOLDS[0].test_start
    dates = pd.Series(pd.date_range("2023-01-01", "2024-12-31"))
    tr, va, te = time_split(dates)
    assert dates[tr].max() < dates[va].min() and dates[va].max() < dates[te].min()


def test_make_folds_count():
    assert len(make_folds("2024-01-01", 30, 6)) == 6
