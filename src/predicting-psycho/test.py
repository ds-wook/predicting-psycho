import pandas as pd

lgb_model = pd.read_csv('../../res/bayesian_lgbm.csv')
deep_model = pd.read_csv('../../res/deeplearning.csv')
submission = pd.read_csv('../../data/sample_submission.csv')

submission['voted'] = 0.5 * lgb_model['voted'] + 0.5 * deep_model['voted']
submission.to_csv('../../res/deep_lgbm.csv', index=False)
