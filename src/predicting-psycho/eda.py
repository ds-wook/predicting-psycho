# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# %%


train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test_x.csv')


# %%


train.head()


# %%


train.info()


# %%


submission = pd.read_csv('../../data/sample_submission.csv')
submission.head()


# %%
