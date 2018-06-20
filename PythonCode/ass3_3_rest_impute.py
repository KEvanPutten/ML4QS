from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles_ass3/'
dataset = pd.read_csv(dataset_path + 'outliers_result.csv', index_col=0)
dataset.index = dataset.index.to_datetime()

# Computer the number of milliseconds covered by an instane based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

# Step 2: Let us impute the missing values.
MisVal = ImputationMissingValues()
# And we impute for all columns except for the label in the selected way (interpolation)
for col in [c for c in dataset.columns if not 'label' in c]:
    dataset = MisVal.impute_interpolate(dataset, col)

# Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz
LowPass = LowPassFilter()

# Determine the sampling frequency.
fs = float(1000)/milliseconds_per_instance
cutoff = 1.5

# Let us study acc_x:
new_dataset = LowPass.low_pass_filter(copy.deepcopy(dataset), 'acc_x', fs, cutoff, order=10)
DataViz.plot_dataset(new_dataset.ix[int(0.4*len(new_dataset.index)):int(0.43*len(new_dataset.index)), :], ['acc_x', 'acc_x_lowpass'], ['exact','exact'], ['line', 'line'])

# And not let us include all measurements that have a form of periodicity (and filter them):
periodic_measurements = ['acc_x', 'acc_y', 'acc_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_y', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']

for col in periodic_measurements:
    dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
    dataset[col] = dataset[col + '_lowpass']
    del dataset[col + '_lowpass']


# Determine the PC's for all but our target columns (the labels)
PCA = PrincipalComponentAnalysis()
selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c))]
pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

# Plot the variance explained.
plot.plot(range(1, len(selected_predictor_cols)+1), pc_values, 'b-')
plot.xlabel('principal component number')
plot.ylabel('explained variance')
plot.show(block=False)

# We select 9 as the best number of PC's as this explains most of the variance
n_pcs = 9

dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

#And we visualize the result of the PC's
DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])

"""
# And the overall final dataset:
DataViz.plot_dataset(dataset, ['acc_', 'lin_', 'gyr_', 'light_', 'loc_', 'mag_', 'press_', 'pca_', 'label'],
                              ['like', 'like', 'like', 'like', 'like', 'like', 'like','like', 'like'],
                              ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])
"""

# Store the outcome.
dataset.to_csv(dataset_path + 'imputation_result.csv')
