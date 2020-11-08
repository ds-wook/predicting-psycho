from bayes_opt import BayesianOptimization
from typing import Any, Dict, Tuple


def lgbm_parameter(
        func: Any,
        params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    lgbm_bo = BayesianOptimization(f=func, pbounds=params,
                                   verbose=2, random_state=91)
    lgbm_bo.maximize(init_points=5, n_iter=25, acq='ei', xi=0.01)

    return lgbm_bo.max['params']


def xgb_parameter(
        func: Any,
        params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    xgb_bo = BayesianOptimization(f=func, pbounds=params,
                                  verbose=2, random_state=91)
    xgb_bo.maximize(init_points=5, n_iter=5, acq='ei', xi=0.01)

    return xgb_bo.max['params']


def rf_parameter(
        func: Any,
        params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    rf_bo = BayesianOptimization(f=func, pbounds=params,
                                 verbose=2, random_state=91)
    rf_bo.maximize(init_points=5, n_iter=10, acq='ei', xi=0.01)
    return rf_bo.max['params']
