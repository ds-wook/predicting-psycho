import pandas as pd
from typing import Tuple


def machiavellism_test_score(
        data: pd.DataFrame) -> pd.DataFrame:
    answers = ['QaA', 'QbA', 'QcA', 'QdA', 'QeA',
               'QfA', 'QgA', 'QhA', 'QiA', 'QjA',
               'QkA', 'QlA', 'QmA', 'QnA', 'QoA',
               'QpA', 'QqA', 'QrA', 'QsA', 'QtA']
    flipping_column = ["QeA", "QfA", "QkA", "QqA", "QrA"]
    data[flipping_column] = data[flipping_column].apply(lambda x: 6 - x)
    data['math_score'] = data[answers].mean(axis=1)
    return data


def mean_feature(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replace_dict = {'education': str, 'engnat': str,
                    'married': str, 'urban': str}
    train = train.astype(replace_dict)
    test = test.astype(replace_dict)
    features = ['age_group', 'education', 'engnat',
                'urban', 'race', 'religion', 'gender']
    for feature in features:
        map_dic =\
            train[[feature, 'voted']].groupby(feature).mean()
        map_dic = map_dic.to_dict()['voted']

        train[feature + '_target_enc'] =\
            train[feature].apply(lambda x: map_dic.get(x))
        test[feature + '_target_enc'] =\
            test[feature].apply(lambda x: map_dic.get(x))

    return train, test


def fea_eng_encoding(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
                 'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
                 'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
                 'QpE', 'QqE', 'QrE', 'QsE', 'QtE'] + ['index', 'hand']

    replace_dict = {'education': str, 'engnat': str,
                    'married': str, 'urban': str}
    train, test = mean_feature(train, test)
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
    train_ohe = machiavellism_test_score(train_ohe)
    test_ohe = machiavellism_test_score(test_ohe)

    return train_ohe, test_ohe, train_y
