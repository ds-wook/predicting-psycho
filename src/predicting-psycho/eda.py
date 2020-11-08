# %%[markdown]
'''
## 라이브러리 불러오기
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# %% [markdown]
'''
## 데이터 불러오기
'''

train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')
submission = pd.read_csv('../../input/sample_submission.csv')


# %% [markdown]
'''
## 데이터 전처리
'''

country_map = {c: i for i, c in enumerate(train['country'].unique())}
major_map = {c: i for i, c in enumerate(train['major'].unique())}

train['country'] = train['country'].map(country_map)
test['country'] = test['country'].map(country_map)

train['major'] = train['major'].map(major_map)
test['major'] = test['major'].map(major_map)

train.isna().sum()[train.isna().sum() != 0]
test.isna().sum()[test.isna().sum() != 0]

test = test.fillna(0)
test.isna().sum()[test.isna().sum() != 0]

x_train = train.loc[:, 'Q1':'ASD']
y_train = train['nerdiness']

x_test = test.loc[:, 'Q1':'ASD']

X_train, X_test, y_train, y_test =\
    train_test_split(x_train, y_train, test_size=0.3, random_state=91)
# %% [markdown]
'''
## 베이지안 최적화를 위한 모델
'''


def rf_cv(n_estimators, max_depth, min_samples_split):

    model = RandomForestClassifier(
                   n_estimators=int(max(n_estimators, 0)),
                   max_depth=int(max(max_depth, 1)),
                   min_samples_split=int(max(min_samples_split, 2)),
                   n_jobs=-1,
                   random_state=42,
                   class_weight="balanced")
    scoring = {'acc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    accuracy = result['test_acc_score'].mean()
    return accuracy


def rf_parameter(func, params):
    rf_bo = BayesianOptimization(f=func, pbounds=params,
                                 verbose=2, random_state=91)
    rf_bo.maximize(init_points=5, n_iter=15, acq='ei', xi=0.01)
    return rf_bo.max['params']


# %% [markdown]
'''
## 최적화 시작
'''
parameters = {"n_estimators": (300, 1000),
              "max_depth": (20, 100),
              "min_samples_split": (2, 10)}


params = rf_parameter(rf_cv, parameters)


# %% [markdown]
'''
## RandomForest 학습
'''
model = RandomForestClassifier(
             n_estimators=int(max(params["n_estimators"], 0)),
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)),
             n_jobs=-1,
             random_state=42,
             class_weight="balanced")
model.fit(X_train, y_train)
y_preds = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_preds)

# %%[markdown]
'''
## 제출
'''

