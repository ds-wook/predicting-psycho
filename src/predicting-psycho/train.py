from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from optim.bayesian_train import xgb_cv
from optim.bayesian_optim import xgb_parameter
from optim.bayesian_train import lgbm_cv
from optim.bayesian_optim import lgbm_parameter
from model.kfold_model import stratified_kfold_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    train_x = train.drop(drop_list + ['voted'], axis=1)
    test_x = test.drop(drop_list, axis=1)
    train_x = train_x.astype(replace_dict)
    test_x = test_x.astype(replace_dict)
    le_encoder = OrdinalEncoder(train_x.columns)
    train_le = le_encoder.fit_transform(train_x, train_y)
    test_le = le_encoder.transform(test_x)

    X_train, X_valid, y_train, y_valid =\
        train_test_split(train_le, train_y, test_size=0.2, random_state=91)

    lgb_param_bounds = {
        'max_depth': (6, 16),
        'num_leaves': (24, 1024),
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
                verbose=400,
                n_estimators=500,
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
                gamma=bo_xgb['gamma'])
    lgb_preds = stratified_kfold_model(lgb_clf, 5, X_train, y_train, X_valid)
    xgb_preds = stratified_kfold_model(xgb_clf, 5, X_train, y_train, X_valid)

    print(f'LGBM AUC Score: {roc_auc_score(y_valid, lgb_preds):.5f}')
    print(f'XGB AUC Score: {roc_auc_score(y_valid, xgb_preds):.5f}')
