# %%[markdown]
'''
## 라이브러리 불러오기
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# %% [markdown]
'''
## 데이터 불러오기
'''

train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test_x.csv')
submission = pd.read_csv('../../input/sample_submission.csv')


# %%[markdown]
'''
## EDA -> 시각화 및 데이터 분석은 알아서 해보시길...
'''

print(f'train\'s shapes: {train.shape}')
train.head()
# %%
train.info()
# %%

wf_list = [f'wf_0{i}' for i in range(1, 4)]
wr_list =\
    [f'wr_0{i}' if i in range(1, 10) else f'wr_{i}' for i in range(1, 14)]

train['wf_total'] = train[wf_list].sum(axis=1)
train['wr_total'] = train[wr_list].sum(axis=1)
# %%
