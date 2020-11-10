from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import pandas as pd

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


def lgbm_cv(
        num_leaves: int,
        max_depth: int,
        min_child_samples: int,
        subsample: float,
        colsample_bytree: float,
        max_bin: float,
        reg_alpha: float,
        reg_lambda: float) -> float:

    model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.02,
                num_leaves=int(round(num_leaves)),
                max_depth=int(round(max_depth)),
                min_child_samples=int(round(min_child_samples)),
                subsample=max(min(subsample, 1), 0),
                colsample_bytree=max(min(colsample_bytree, 1), 0),
                max_bin=max(int(round(max_bin)), 10),
                reg_alpha=max(reg_alpha, 0),
                reg_lambda=max(reg_lambda, 0),
                random_state=94
            )

    scoring = {'auc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, train_ohe, train_y, cv=5, scoring=scoring)
    accuracy = result['test_auc_score'].mean()
    return accuracy


def xgb_cv(
        learning_rate: float,
        n_estimators: int,
        max_depth: int,
        subsample: float,
        gamma: float) -> float:

    model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=int(round(n_estimators)),
            max_depth=int(round(max_depth)),
            subsample=subsample,
            gamma=gamma,
            random_state=94)

    scoring = {'auc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, train_ohe, train_y, cv=5, scoring=scoring)
    accuracy = result['test_auc_score'].mean()
    return accuracy


def rf_cv(
        n_estimators: int,
        max_depth: int,
        min_samples_split: int) -> float:
    model = RandomForestClassifier(
                   n_estimators=int(max(n_estimators, 0)),
                   max_depth=int(max(max_depth, 1)),
                   min_samples_split=int(max(min_samples_split, 2)),
                   n_jobs=-1,
                   random_state=42,
                   class_weight="balanced")
    scoring = {'auc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, train_ohe, train_y, cv=5, scoring=scoring)
    accuracy = result['test_auc_score'].mean()
    return accuracy
