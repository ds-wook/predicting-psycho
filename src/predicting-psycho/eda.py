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


# %%[markdown]
'''
## EDA -> 시각화 및 데이터 분석은 알아서 해보시길...
'''

# %%
'''
## 데이터 전처리
'''


train = train.fillna(0)
test = test.fillna(0)
train_x = train.drop(['index', 'nerdiness'], axis=1)
train_y = train['nerdiness']
test_x = test.drop(['index'], axis=1)

train_ohe = pd.get_dummies(train_x)
test_ohe = pd.get_dummies(test_x)

print(f'After One Hot Test: {train_ohe.shape}')
print(f'After One Hot Test: {test_ohe.shape}')
X_train, X_test, y_train, y_test =\
    train_test_split(train_ohe, train_y, test_size=0.3, random_state=91)


# %% [markdown]
'''
## 베이지안 최적화를 위한 모델
'''


def rf_cv(max_depth, min_samples_split):
    model = RandomForestClassifier(
                   max_depth=int(max(max_depth, 1)),
                   min_samples_split=int(max(min_samples_split, 2)),
                   n_jobs=-1,
                   random_state=91)
    scoring = {'acc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    accuracy = result['test_acc_score'].mean()
    return accuracy


def rf_parameter(func, params):
    rf_bo = BayesianOptimization(f=func, pbounds=params,
                                 verbose=2, random_state=91)
    rf_bo.maximize(init_points=5, n_iter=5, acq='ei', xi=0.01)
    return rf_bo.max['params']


# %% [markdown]
'''
## 최적화 시작
'''
parameters = {"max_depth": (20, 100),
              "min_samples_split": (2, 10)}

params = rf_parameter(rf_cv, parameters)
# %% [markdown]
'''
## RandomForest 학습
'''
model = RandomForestClassifier(
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)),
             n_jobs=-1,
             random_state=91)
model.fit(X_train, y_train)
y_preds = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_preds)

# %%[markdown]
'''
## 베이지안 최적화 재학습
'''


def rf_cv(max_depth, min_samples_split):
    model = RandomForestClassifier(
                   max_depth=int(max(max_depth, 1)),
                   min_samples_split=int(max(min_samples_split, 2)),
                   n_jobs=-1,
                   random_state=42,
                   class_weight="balanced")
    scoring = {'acc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, train_ohe, train_y, cv=5, scoring=scoring)
    accuracy = result['test_acc_score'].mean()
    return accuracy


def rf_parameter(func, params):
    rf_bo = BayesianOptimization(f=func, pbounds=params,
                                 verbose=2, random_state=91)
    rf_bo.maximize(init_points=5, n_iter=5, acq='ei', xi=0.01)
    return rf_bo.max['params']


parameters = {"max_depth": (20, 100),
              "min_samples_split": (2, 10)}

params = rf_parameter(rf_cv, parameters)
# %% [markdown]
'''
# 모델 재학습 및 제출
'''
model = RandomForestClassifier(
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)),
             n_jobs=-1,
             random_state=91)
model.fit(train_ohe, train_y)
y_preds = model.predict_proba(test_ohe)[:, 1]
y_preds = y_preds.astype(np.float32)
submission['nerdiness'] = y_preds
submission.to_csv('../../res/bayesian_rf.csv', index=False)
# %%
train_ohe
# %%
test_ohe
# %%
