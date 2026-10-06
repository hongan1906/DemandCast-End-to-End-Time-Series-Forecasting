"""Stage 7 model zoo for the tabular (feature) pipeline: Ridge -> XGBoost -> CatBoost."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["day_of_week", "month", "product_id"]


def make_ridge(feature_cols, alpha: float = 10.0):
    cats = [c for c in CATEGORICAL if c in feature_cols]
    nums = [c for c in feature_cols if c not in cats]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cats), ("num", StandardScaler(), nums)]
    )
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def make_xgboost(feature_cols=None, n_estimators: int = 300, seed: int = 0):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, n_jobs=4, verbosity=0,
    )


def make_catboost(feature_cols=None, seed: int = 0):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, random_seed=seed, verbose=0)


def available_models(feature_cols) -> dict:
    """name -> zero-arg factory. CatBoost is skipped if not installed."""
    zoo = {
        "ridge": lambda: make_ridge(feature_cols),
        "xgboost": lambda: make_xgboost(feature_cols),
    }
    try:
        import catboost  # noqa: F401

        zoo["catboost"] = lambda: make_catboost(feature_cols)
    except ImportError:
        pass
    return zoo
