from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 데이터 불러오기
train = pd.read_csv('../../data/train.csv', index_col=0)
test = pd.read_csv('../../data/test_x.csv', index_col=0)
submission = pd.read_csv('../../data/sample_submission.csv', index_col=0)

X = train.drop(['voted'], axis=1)
y = train['voted']
X = pd.get_dummies(X)
test = pd.get_dummies(test)

X = X.fillna(X.mean())
X.drop_duplicates(keep='first', inplace=True)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
test = scaler.transform(test)
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=91)


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
                random_state=91
            )

    scoring = {'f1_score': make_scorer(accuracy_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    accuracy = result['test_f1_score'].mean()
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
            random_state=91)

    scoring = {'f1_score': make_scorer(accuracy_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    accuracy = result['test_f1_score'].mean()
    return accuracy
