from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb


class SklearnCompatibleXGBRegressor(xgb.XGBRegressor, BaseEstimator, RegressorMixin):
    def __sklearn_tags__(self):
        # Include any specific tags required for scikit-learn compatibility
        return {
            "requires_positive_y": False,
            "poor_score": False,
            "multioutput": False,
        }
