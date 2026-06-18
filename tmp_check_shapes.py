import os
import pandas as pd
from src.utils.config import get_train_config

cfg = get_train_config()
proc_dir = cfg['paths']['proc_dir']
X = pd.read_csv(os.path.join(proc_dir, 'X_train.csv'))
y = pd.read_csv(os.path.join(proc_dir, 'y_train.csv'), header=None, names=['log_HER']).squeeze()
print('X shape:', X.shape)
print('y shape:', y.shape)
print('X rows index length:', len(X))
print('y length:', len(y))
print('Columns count:', len(X.columns))
