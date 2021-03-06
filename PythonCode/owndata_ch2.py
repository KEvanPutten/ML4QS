dataset_path = '../assignment1/axel-data-parsed/'
result_dataset_path = './intermediate_datafiles/'

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
import copy
import os


if not os.path.exists(result_dataset_path):
    print('Creating result directory: ' + result_dataset_path)
    os.makedirs(result_dataset_path)

# Set a granularity (i.e. how big are our discrete time steps). We start very
# coarse grained, namely one measurement per minute, and secondly use four measurements
# per second

granularities = [60000, 250]
datasets = []

for milliseconds_per_instance in granularities:

    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(dataset_path, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer and magnetometer data (continuous numerical measurements) of the phone
    # and aggregate the values per timestep by averaging the values/
    DataSet.add_numerical_dataset('accelerometer.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_phone_')

    DataSet.add_numerical_dataset('magnetometer.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_phone_')

    DataSet.add_numerical_dataset('gyroscope.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyro_phone_')

    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    DataSet.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')

    # Get the resulting pandas data table

    dataset = DataSet.data_table

    # Plot the data

    DataViz = VisualizeDataset()

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x','acc_phone_y','acc_phone_z'])

    # Plot all data
    DataViz.plot_dataset(dataset, ['acc_', 'mag_', 'gyro_', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])

    # And print a summary of the dataset

    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

# And print the table that has been included in the book

util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we have generated (250 ms).
dataset.to_csv(result_dataset_path + 'owndata_chapter2_result.csv')
