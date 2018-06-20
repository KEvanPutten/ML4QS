##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################


dataset_path = './ass3_rawdata/'
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

granularities = [60000, 2500]
datasets = []

for milliseconds_per_instance in granularities:

    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(dataset_path, milliseconds_per_instance)

    # Add the selected measurements to it.

    # Add numerical measurements
    DataSet.add_numerical_dataset('A01_parsed_raw_data.csv', 'timestamps',
                                  ['ankle_l_x', 'ankle_l_y', 'ankle_l_z', 'ankle_r_x', 'ankle_r_y', 'ankle_r_z',
                                   'belt_x', 'belt_y', 'belt_z', 'chest_x', 'chest_y', 'chest_z'], 'avg', '')

    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    DataSet.add_binary_labels_dataset('A01_parsed_raw_data.csv', 'timestamps',
                                      ['labelWalking', 'labelFalling', 'labelLyingDown', 'labelLying',
                                       'labelSittingDown', 'labelSitting', 'labelStandingFromLying', 'labelOnAllFours',
                                       'labelSittingOnTheGround', 'labelStandingFromSitting',
                                       'labelStandingFromSittingOnTheGround'], 'max', '')

    # Get the resulting pandas data table

    dataset = DataSet.data_table

    # Plot the data

    DataViz = VisualizeDataset()

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['ankle_l_x', 'ankle_l_y', 'ankle_l_z', 'ankle_r_x', 'ankle_r_y', 'ankle_r_z',
                                           'belt_x', 'belt_y', 'belt_z', 'chest_x', 'chest_y', 'chest_z'])

    # Plot all data
    DataViz.plot_dataset(dataset, ['ankle_l_', 'ankle_r_', 'belt_', 'chest_', 'label'], ['like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'])

    # And print a summary of the dataset

    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

# And print the table that has been included in the book

util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we have generated (250 ms).
dataset.to_csv(result_dataset_path + 'chapter2_result.csv')
