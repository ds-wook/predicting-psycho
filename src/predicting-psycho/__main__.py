from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from optim.bayesian_train import lgbm_cv
from optim.bayesian_optim import lgbm_parameter
from sklearn.preprocessing import MinMaxScaler
from model.kfold_model import stratified_kfold_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
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

    lgb_param_bounds = {
        'max_depth': (6, 16),
        'num_leaves': (24, 64),
        'min_child_samples': (10, 200),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'max_bin': (10, 500),
        'reg_lambda': (0.001, 10),
        'reg_alpha': (0.01, 50)
    }

    bo_lgb = lgbm_parameter(lgbm_cv, lgb_param_bounds)

    # lgbm 분류기
    lgb_clf = LGBMClassifier(
                objective='binary',
                verbose=400,
                random_state=94,
                n_estimators=500,
                learning_rate=0.02,
                max_depth=int(round(bo_lgb['max_depth'])),
                num_leaves=int(round(bo_lgb['num_leaves'])),
                min_child_samples=int(round(bo_lgb['min_child_samples'])),
                subsample=max(min(bo_lgb['subsample'], 1), 0),
                colsample_bytree=max(min(bo_lgb['colsample_bytree'], 1), 0),
                max_bin=max(int(round(bo_lgb['max_bin'])), 10),
                reg_lambda=max(bo_lgb['reg_lambda'], 0),
                reg_alpha=max(bo_lgb['reg_alpha'], 0)
            )
    X_train, X_valid, y_train, y_valid =\
        train_test_split(X, y, test_size=0.2, random_state=94)
    test_preds = stratified_kfold_model(lgb_clf, 5, X_train, y_train, X_valid)
    print(f'Accuracy Score: {accuracy_score(y_valid, test_preds):.5f}')

    y_preds = stratified_kfold_model(lgb_clf, 5, X, y, test)

    submission['voted'] = y_preds
    submission['voted'] = submission['voted'].astype(np.int64)

    submission.to_csv('../../res/bayesian_model.csv')
