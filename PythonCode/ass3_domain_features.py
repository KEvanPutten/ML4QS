from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
import copy
import pandas as pd

DataViz = VisualizeDataset()

dataset_path = './intermediate_datafiles_ass3/'
try:
    dataset = pd.read_csv(dataset_path + 'imputation_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to outlier and imputation scripts first!')
    raise e

dataset.index = dataset.index.to_datetime()

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

# ------------------------------------------------------------------------------------
# TIME DOMAIN

NumAbs = NumericalAbstraction()
# Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 1 minute
# This part is for generating plots, it plots for all current non-label features
window_sizes = [int(float(4*5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(1*60000)/milliseconds_per_instance)]

cols = [c for c in dataset.columns if not 'label' in c]
for c in cols:
    dataset_copy = copy.deepcopy(dataset)
    for ws in window_sizes:
        dataset_copy = NumAbs.abstract_numerical(dataset_copy, [c], ws, 'mean')
        dataset_copy = NumAbs.abstract_numerical(dataset_copy, [c], ws, 'std')
        dataset_copy = NumAbs.abstract_numerical(dataset_copy, [c], ws, 'min')
        dataset_copy = NumAbs.abstract_numerical(dataset_copy, [c], ws, 'MAD')
        dataset_copy = NumAbs.abstract_numerical(dataset_copy, [c], ws, 'kurtosis')
        dataset_copy = NumAbs.abstract_numerical(dataset_copy, [c], ws, 'slope')
    DataViz.plot_dataset(dataset_copy,
                         [c, c+'_temp_min', c+'_temp_mean', c+'_temp_std', c+'_temp_MAD', c+'_temp_kurtosis', c+'_temp_slope', 'label'],
                         ['exact', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

# Kurtosis appears to spike when changing activities; not usefor for activity classification
# Min also does not really appear useful since most features have a frequency
# Slope is useless
# MAD is more robust and less susceptible to outliers than deviation, so we feel it might be usefull instead of std?

# Compute the time domain metric and add them to the dataset
# We select a window size of 20 seconds
ws = int(float(20000)/milliseconds_per_instance)
selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'MAD')

# Categorical abstraction is not usefult since we're trying to predict the activities

# ------------------------------------------------------------------------------------
# FREQUENCY DOMAIN

# ------------------------------------------------------------------------------------
# REDUCE OVERLAP

# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]

dataset.to_csv(dataset_path + 'domain_features_result.csv')

DataViz.plot_dataset(dataset,
                     ['acc_x', 'gyr_x', 'lin_acc_x', 'light_illuminance', 'mag_x', 'loc_height', 'pca_1', 'label'],
                     ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
                     ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])
