from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from model.kfold_model import stratified_kfold_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    train_x = pd.get_dummies(train_x)
    test_x = pd.get_dummies(test_x)
    train_y = 2 - train_y
    train_x.iloc[:, :20] = (train_x.iloc[:, :20] - 3.) / 2.
    test_x.iloc[:, :20] = (test_x.iloc[:, :20] - 3.) / 2
    train_x.iloc[:, 20] = (train_x.iloc[:, 20] - 4.) / 4.
    test_x.iloc[:, 20] = (test_x.iloc[:, 20] - 4.) / 4.
    train_x.iloc[:, 21:31] = (train_x.iloc[:, 21:31] - 3.5) / 3.5
    test_x.iloc[:, 21:31] = (test_x.iloc[:, 21:31] - 3.5) / 3.5

    # lgbm 분류기
    lgb_clf = LGBMClassifier(
                verbose=400,
                n_estimators=500,
                learning_rate=0.02,
                random_state=91)
    X_train, X_valid, y_train, y_valid =\
        train_test_split(train_x, train_y, test_size=0.2, random_state=91)

    test_preds = stratified_kfold_model(lgb_clf, 5, X_train, y_train, X_valid)
    test_preds = np.array([1 if prob > 0.5 else 0 for prob in test_preds])
    test_preds = test_preds.reshape(-1, 1)
    print(f'Accuracy Score: {accuracy_score(y_valid, test_preds):.5f}')
