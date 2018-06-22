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
print 'starting time domain computations.'

NumAbs = NumericalAbstraction()
"""
# Set the window sizes to the number of instances representing 20 seconds, 30 seconds and 40 seconds
# This part is for generating plots, it plots for all current non-label features
window_sizes = [int(float(4*5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(40000)/milliseconds_per_instance)]

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
"""

# Kurtosis might be useful, not sure
# Slope appears to be useless
# Min also does not really appear useful since most features have a frequency
# MAD is more robust and less susceptible to outliers than deviation, so we feel it might be useful

# Compute the time domain metric and add them to the dataset
# We select a window size of 30 seconds
ws = int(float(30000)/milliseconds_per_instance)
print ws
selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'MAD')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'kurtosis')

# Categorical abstraction is not usefult since we're trying to predict the activities

# ------------------------------------------------------------------------------------
# FREQUENCY DOMAIN
print 'starting frequency domain computations.'

FreqAbs = FourierTransformation()
fs = float(1000)/milliseconds_per_instance

periodic_predictor_cols = ['acc_x','acc_y','acc_z','gyr_x','gyr_y', 'gyr_z',
                           'mag_x','mag_y','mag_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']

# Spectral analysis of ALL features (generates 18 plots)
"""
for c in periodic_predictor_cols:
    data_table = FreqAbs.abstract_frequency(copy.deepcopy(dataset), [c],
                                            int(float(10000) / milliseconds_per_instance), fs)
    DataViz.plot_dataset(data_table, [c+'_max_freq', c+'_freq_weighted', c+'_pse', c+'_freq_skewness', c+'_freq_kurtosis', 'label'],
                         ['like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'points'])
"""

# Compute and add frequency domain features to dataset
dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)

# ------------------------------------------------------------------------------------
# REDUCE OVERLAP
print 'reducing overlap.'

# The percentage of overlap we allow
window_overlap = 0.95
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]

dataset.to_csv(dataset_path + 'domain_features_result_95.csv')

DataViz.plot_dataset(dataset,
                     ['acc_x', 'gyr_x', 'lin_acc_x', 'light_illuminance', 'mag_x', 'loc_height', 'pca_1', 'label'],
                     ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
                     ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])
