import numpy as np
import pandas as pd

dataset_path = './intermediate_datafiles_ass3/'

try:
    dataset = pd.read_csv(dataset_path + 'ConfLongDemo_JSI.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

# create unique index
dataset.reset_index(inplace=True)

# add new columns to data
columns = ['ankle_l_x', 'ankle_l_y', 'ankle_l_z', 'ankle_r_x', 'ankle_r_y', 'ankle_r_z',
           'belt_x', 'belt_y', 'belt_z', 'chest_x', 'chest_y', 'chest_z']
for c in columns:
    dataset[c] = np.nan

# copy data to appropriate columns
dataset_copy = dataset.copy()

dataset_copy.loc[dataset['sensor_id'] == '010-000-024-033', 'ankle_l_x'] = dataset['x']
dataset_copy.loc[dataset['sensor_id'] == '010-000-024-033', 'ankle_l_y'] = dataset['y']
dataset_copy.loc[dataset['sensor_id'] == '010-000-024-033', 'ankle_l_z'] = dataset['z']

dataset_copy.loc[dataset['sensor_id'] == '010-000-030-096', 'ankle_r_x'] = dataset['x']
dataset_copy.loc[dataset['sensor_id'] == '010-000-030-096', 'ankle_r_y'] = dataset['y']
dataset_copy.loc[dataset['sensor_id'] == '010-000-030-096', 'ankle_r_z'] = dataset['z']

dataset_copy.loc[dataset['sensor_id'] == '020-000-033-111', 'chest_x'] = dataset['x']
dataset_copy.loc[dataset['sensor_id'] == '020-000-033-111', 'chest_y'] = dataset['y']
dataset_copy.loc[dataset['sensor_id'] == '020-000-033-111', 'chest_z'] = dataset['z']

dataset_copy.loc[dataset['sensor_id'] == '020-000-032-221', 'belt_x'] = dataset['x']
dataset_copy.loc[dataset['sensor_id'] == '020-000-032-221', 'belt_y'] = dataset['y']
dataset_copy.loc[dataset['sensor_id'] == '020-000-032-221', 'belt_z'] = dataset['z']


# And store it all!
dataset_copy.to_csv(dataset_path + 'parsed_raw_data.csv')

data_A01 = dataset_copy.loc[dataset_copy.id == 'A01']
data_A02 = dataset_copy.loc[dataset_copy.id == 'A02']
data_A03 = dataset_copy.loc[dataset_copy.id == 'A03']
data_A04 = dataset_copy.loc[dataset_copy.id == 'A04']
data_A05 = dataset_copy.loc[dataset_copy.id == 'A05']
data_A01.to_csv(dataset_path + 'A01_parsed_raw_data.csv')
data_A02.to_csv(dataset_path + 'A02_parsed_raw_data.csv')
data_A03.to_csv(dataset_path + 'A03_parsed_raw_data.csv')
data_A04.to_csv(dataset_path + 'A04_parsed_raw_data.csv')
data_A05.to_csv(dataset_path + 'A05_parsed_raw_data.csv')

data_B01 = dataset_copy.loc[dataset_copy.id == 'B01']
data_B02 = dataset_copy.loc[dataset_copy.id == 'B02']
data_B03 = dataset_copy.loc[dataset_copy.id == 'B03']
data_B04 = dataset_copy.loc[dataset_copy.id == 'B04']
data_B05 = dataset_copy.loc[dataset_copy.id == 'B05']
data_B01.to_csv(dataset_path + 'B01_parsed_raw_data.csv')
data_B02.to_csv(dataset_path + 'B02_parsed_raw_data.csv')
data_B03.to_csv(dataset_path + 'B03_parsed_raw_data.csv')
data_B04.to_csv(dataset_path + 'B04_parsed_raw_data.csv')
data_B05.to_csv(dataset_path + 'B05_parsed_raw_data.csv')

data_C01 = dataset_copy.loc[dataset_copy.id == 'C01']
data_C02 = dataset_copy.loc[dataset_copy.id == 'C02']
data_C03 = dataset_copy.loc[dataset_copy.id == 'C03']
data_C04 = dataset_copy.loc[dataset_copy.id == 'C04']
data_C05 = dataset_copy.loc[dataset_copy.id == 'C05']
data_C01.to_csv(dataset_path + 'C01_parsed_raw_data.csv')
data_C02.to_csv(dataset_path + 'C02_parsed_raw_data.csv')
data_C03.to_csv(dataset_path + 'C03_parsed_raw_data.csv')
data_C04.to_csv(dataset_path + 'C04_parsed_raw_data.csv')
data_C05.to_csv(dataset_path + 'C05_parsed_raw_data.csv')

data_D01 = dataset_copy.loc[dataset_copy.id == 'D01']
data_D02 = dataset_copy.loc[dataset_copy.id == 'D02']
data_D03 = dataset_copy.loc[dataset_copy.id == 'D03']
data_D04 = dataset_copy.loc[dataset_copy.id == 'D04']
data_D05 = dataset_copy.loc[dataset_copy.id == 'D05']
data_D01.to_csv(dataset_path + 'D01_parsed_raw_data.csv')
data_D02.to_csv(dataset_path + 'D02_parsed_raw_data.csv')
data_D03.to_csv(dataset_path + 'D03_parsed_raw_data.csv')
data_D04.to_csv(dataset_path + 'D04_parsed_raw_data.csv')
data_D05.to_csv(dataset_path + 'D05_parsed_raw_data.csv')

data_E01 = dataset_copy.loc[dataset_copy.id == 'E01']
data_E02 = dataset_copy.loc[dataset_copy.id == 'E02']
data_E03 = dataset_copy.loc[dataset_copy.id == 'E03']
data_E04 = dataset_copy.loc[dataset_copy.id == 'E04']
data_E05 = dataset_copy.loc[dataset_copy.id == 'E05']
data_E01.to_csv(dataset_path + 'E01_parsed_raw_data.csv')
data_E02.to_csv(dataset_path + 'E02_parsed_raw_data.csv')
data_E03.to_csv(dataset_path + 'E03_parsed_raw_data.csv')
data_E04.to_csv(dataset_path + 'E04_parsed_raw_data.csv')
data_E05.to_csv(dataset_path + 'E05_parsed_raw_data.csv')
