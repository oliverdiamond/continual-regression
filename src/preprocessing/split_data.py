'''
Train/Test/Val split for Nov-Dec data
Before running, download month_all.hdf5 from
____
and place in continual-regression/data/
'''

import h5py
from pathlib import Path

# Load data
data_dir = Path(__file__).parents[2] / 'data'
month_all =  h5py.File(data_dir / 'month_all.hdf5', 'r')

# Get indicies for split
train_end = month_all['file_indices'][33][1]+1
val_end = month_all['file_indices'][43][1]+1
test_end = month_all['file_indices'][54][1]+1

# train/val/test split
filenames = ['month_train.hdf5', 'month_val.hdf5', 'month_test.hdf5']
ranges = [
    (0, train_end),
    (train_end, val_end),
    (val_end, test_end),
    ]

for i, (start, end) in enumerate(ranges):
    with h5py.File(data_dir / filenames[i], 'w') as out:
        out.create_dataset('features', data=month_all['features'][start:end])
        out.attrs['labels'] = month_all.attrs['labels']

month_all.close()


