from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 불러오기
train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test_x.csv')
submission = pd.read_csv('../../data/sample_submission.csv')

drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
             'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
             'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
             'QpE', 'QqE', 'QrE', 'QsE', 'QtE'] + ['index', 'hand']

replace_dict = {'education': str, 'engnat': str, 'married': str, 'urban': str}
train_y = train['voted']
train_x = train.drop(drop_list + ['voted'], axis=1)
test_x = test.drop(drop_list, axis=1)
train_x = train_x.astype(replace_dict)
test_x = test_x.astype(replace_dict)
train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)
train_y = 2 - train_y
train_x.iloc[:, :20] = (train_x.iloc[:, :20] - 3.) / 2.
test_x.iloc[:, :20] = (test_x.iloc[:, :20] - 3.) / 2
train_x.iloc[:, 20] = (train_x.iloc[:, 20] - 4.) / 4.
test_x.iloc[:, 20] = (test_x.iloc[:, 20] - 4.) / 4.
train_x.iloc[:, 21:31] = (train_x.iloc[:, 21:31] - 3.5) / 3.5
test_x.iloc[:, 21:31] = (test_x.iloc[:, 21:31] - 3.5) / 3.5
X_train, X_valid, y_train, y_valid =\
        train_test_split(train_x, train_y, test_size=0.2, random_state=94)


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

    scoring = {'acc_score': make_scorer(accuracy_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    accuracy = result['test_acc_score'].mean()
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

    scoring = {'acc_score': make_scorer(accuracy_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    accuracy = result['test_acc_score'].mean()
    return accuracy
