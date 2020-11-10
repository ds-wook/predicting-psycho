from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from optim.bayesian_test import xgb_cv
from optim.bayesian_optim import xgb_parameter
from optim.bayesian_test import lgbm_cv
from optim.bayesian_optim import lgbm_parameter
from model.kfold_model import stratified_kfold_model


if __name__ == "__main__":
    # 데이터 불러오기
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test_x.csv')
    submission = pd.read_csv('../../data/sample_submission.csv')

    drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
                 'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
                 'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
                 'QpE', 'QqE', 'QrE', 'QsE', 'QtE'] + ['index', 'hand']

    replace_dict = {'education': str, 'engnat': str,
                    'married': str, 'urban': str}
    train_y = train['voted']
    wf_list = [f'wf_0{i}' for i in range(1, 4)]
    wr_list =\
        [f'wr_0{i}' if i in range(1, 10) else f'wr_{i}' for i in range(1, 14)]
    train['wf_total'] = train[wf_list].sum(axis=1)
    train['wr_total'] = train[wr_list].sum(axis=1)
    test['wf_total'] = test[wf_list].sum(axis=1)
    test['wr_total'] = test[wr_list].sum(axis=1)
    train = train.astype(replace_dict)
    test = test.astype(replace_dict)
    train_x = train.drop(drop_list + ['voted'], axis=1)
    test_x = test.drop(drop_list, axis=1)
    train_ohe = pd.get_dummies(train_x)
    test_ohe = pd.get_dummies(test_x)
    print(f'After One Hot Test: {train_ohe.shape}')
    print(f'After One Hot Test: {test_ohe.shape}')

    lgb_param_bounds = {
        'max_depth': (4, 12),
        'num_leaves': (24, 100),
        'min_child_samples': (10, 200),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'max_bin': (10, 500),
        'reg_lambda': (0, 0.5),
        'reg_alpha': (0, 0.5)
    }

    bo_lgb = lgbm_parameter(lgbm_cv, lgb_param_bounds)

    # lgbm 분류기
    lgb_clf = LGBMClassifier(
                verbose=400,
                n_estimators=1000,
                learning_rate=0.02,
                random_state=91,
                max_depth=int(round(bo_lgb['max_depth'])),
                num_leaves=int(round(bo_lgb['num_leaves'])),
                min_child_samples=int(round(bo_lgb['min_child_samples'])),
                subsample=max(min(bo_lgb['subsample'], 1), 0),
                colsample_bytree=max(min(bo_lgb['colsample_bytree'], 1), 0),
                max_bin=max(int(round(bo_lgb['max_bin'])), 10),
                reg_lambda=max(bo_lgb['reg_lambda'], 0),
                reg_alpha=max(bo_lgb['reg_alpha'], 0)
            )
    lgb_preds =\
        stratified_kfold_model(lgb_clf, 5, train_ohe, train_y, test_ohe)

    # xgb 분류기
    xgb_param_bounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (100, 1000),
        'max_depth': (3, 8),
        'subsample': (0.4, 1.0),
        'gamma': (0, 3)}
    bo_xgb = xgb_parameter(xgb_cv, xgb_param_bounds)

    xgb_clf = XGBClassifier(
                objective='binary:logistic',
                random_state=91,
                learning_rate=bo_xgb['learning_rate'],
                n_estimators=int(round(bo_xgb['n_estimators'])),
                max_depth=int(round(bo_xgb['max_depth'])),
                subsample=bo_xgb['subsample'],
                gamma=bo_xgb['gamma']
            )
    xgb_preds =\
        stratified_kfold_model(xgb_clf, 5, train_ohe, train_y, test_ohe)
    y_preds = 0.6 * lgb_preds + 0.4 * xgb_preds

    y_preds += 1.0
    submission['voted'] = y_preds
    submission['voted'] = submission['voted'].astype(np.float32)
    submission.to_csv('../../res/bayesian_ensemble.csv', index=False)
