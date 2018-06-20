##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################


dataset_path = './ass3_newdata/parsed/'
result_dataset_path = './intermediate_datafiles_ass3/'

# Import the relevant classes.

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
import copy
import os


if not os.path.exists(result_dataset_path):
    print('Creating result directory: ' + result_dataset_path)
    os.makedirs(result_dataset_path)

# Chapter 2: Initial exploration of the dataset.

# Set a granularity (i.e. how big are our discrete time steps). We start very
# coarse grained, namely one measurement per minute, and secondly use four measurements
# per second

granularities = [60000, 250]
datasets = []

for milliseconds_per_instance in granularities:

    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(dataset_path, milliseconds_per_instance)

    # Add the selected measurements to it.

    # Add numerical measurements
    DataSet.add_numerical_dataset('accelerometer.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_')

    DataSet.add_numerical_dataset('linear_acceleration.csv', 'timestamps', ['x','y','z'], 'avg', 'lin_acc_')

    DataSet.add_numerical_dataset('magnetometer.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_')

    DataSet.add_numerical_dataset('Gyroscope.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyr_')

    DataSet.add_numerical_dataset('light.csv', 'timestamps', ['illuminance'], 'avg', 'light_')

    DataSet.add_numerical_dataset('location.csv', 'timestamps', ['latitude','height','velocity'], 'avg', 'loc_')


    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).

    DataSet.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')


    # Get the resulting pandas data table

    dataset = DataSet.data_table

    # Plot the data

    DataViz = VisualizeDataset()

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['acc_x','acc_y','acc_z'])
    DataViz.plot_dataset_boxplot(dataset, ['gyr_x', 'gyr_y', 'gyr_z'])

    # Plot all data
    DataViz.plot_dataset(dataset,
                         ['acc_', 'mag_', 'gyr_', 'light_', 'loc_', 'lin_acc_', 'label'],
                         ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'line', 'points'])

    # print a summary of the dataset
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

# And print the table that has been included in the book
# util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we have generated (250 ms).
dataset.to_csv(result_dataset_path + 'aggregation_result.csv')

