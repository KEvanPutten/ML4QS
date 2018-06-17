from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot

import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter5_result_git_mod.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = dataset.index.to_datetime()

# experimental crap for deriving the activity level

# activity_washingHands = dataset['labelWashingHands']
# activity_eating = dataset['labelEating']*2
# activity_standing = dataset['labelStanding']*1.5
# activity_driving = dataset['labelDriving']*2
# activity_walking = dataset['labelWalking']*6
# activity_running = dataset['labelRunning']*10
#
# activity_acc = abs(dataset['acc_phone_x_temp_mean_ws_120']) + abs(dataset['acc_phone_y_temp_mean_ws_120']) + abs(dataset['acc_phone_z_temp_mean_ws_120'])
# #activity_acc_freq = dataset['test']
#
# activity_acc_watch = abs(dataset['acc_watch_x_temp_mean_ws_120']) + abs(dataset['acc_watch_y_temp_mean_ws_120']) + abs(dataset['acc_watch_z_temp_mean_ws_120'])
#
#
# activity = activity_driving+activity_eating+activity_running+activity_standing+activity_walking+activity_washingHands
# dataset['activity_level'] = activity
# dataset['activity_acc'] = activity_acc_watch
#
# plot.plot(dataset.index, dataset['hr_watch_rate']/200, 'ro-')
# plot.plot(dataset.index, dataset['activity_acc']/25, 'bo:')
#
# # plot.ylim([0, 1])
# plot.xlabel('time')
# plot.ylabel('value')
# plot.legend(['$hr_watch_rate$'], loc=4, fontsize='small', numpoints=1)
# plot.hold(False)
# plot.show()


# ---------------------------------------------------------

prepare = PrepareDatasetForLearning()

# Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate', '2016-02-08 18:29:56',
                                                                                   '2016-02-08 19:34:07', '2016-02-08 20:07:50')

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

basic_features = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y', 'acc_watch_z',
                  'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z',
                  'labelOnTable', 'labelSitting', 'labelWashingHands', 'labelWalking', 'labelStanding', 'labelDriving',
                  'labelEating', 'labelRunning',
                  'light_phone_lux', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'mag_watch_x', 'mag_watch_y',
                  'mag_watch_z', 'press_phone_pressure']
pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7']
time_features = [name for name in dataset.columns if ('temp_' in name and not 'hr_watch' in name)]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print '#basic features: ', len(basic_features)
print '#PCA features: ', len(pca_features)
print '#time features: ', len(time_features)
print '#frequency features: ', len(freq_features)
cluster_features = ['cluster']
print '#cluster features: ', len(cluster_features)
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(
    set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

selected_features = ['temp_pattern_labelOnTable', 'labelOnTable', 'temp_pattern_labelOnTable(b)labelOnTable', 'cluster',
                     'pca_1_temp_mean_ws_120', 'pca_2_temp_mean_ws_120', 'pca_2', 'acc_watch_y_temp_mean_ws_120',
                     'gyr_watch_y_pse',
                     'gyr_watch_x_pse']
possible_feature_sets = [features_after_chapter_5]
#[basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['Chapter 5']
#['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

# We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.

repeats = 1

# we set a washout time to give the NN's the time to stabilize. We do not compute the error during the washout time.

washout_time = 10
scores_over_all_algs = []

# Now let us focus on the learning part.

learner = TemporalRegressionAlgorithms()
eval = RegressionEvaluation()

columns = ['self.acc_watch_x', 'self.acc_watch_y', 'self.hr_watch_rate']
equations = ['self.hr_watch_rate * self.acc_watch_x','self.a * self.hr_watch_rate', 'self.b * self.acc_watch_y']
targets = ['self.hr_watch_rate']
parameters = ['self.a', 'self.b']
#parameters = ['self.y1', 'self.y2', 'self.y3', 'self.y4', 'self.y5', 'self.y6', 'self.y7', 'self.b']

print 'training'
#regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(train_X[features_after_chapter_5],
#                                                               train_y.to_frame(name='hr_watch_rate'), test_X[features_after_chapter_5],
#                                                               test_y.to_frame(name='hr_watch_rate'), columns, equations, targets, parameters)

print 'creating visuals'
#DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'], test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')

# Code below is based on what was provided via email

for i in range(0, len(possible_feature_sets)):
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # a list of columns the model addresses (i.e. the states), the string should be preceded by 'self.' in order for the approach to work.
    # a list of equations to derive the specified states, again using 'self.' preceding all parameters and columns names.
    # a list of targets (a subset of the columns) (again with 'self.')
    # a list of parameters in the equations (again with 'self.')

    performance_tr_dyn = 0
    performance_tr_dyn_std = 0
    performance_te_dyn = 0
    performance_te_dyn_std = 0

    for repeat in range(0, repeats):
        print '----', repeat
        regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(selected_train_X,
                                                                           train_y.to_frame(name='hr_watch_rate'),
                                                                           selected_test_X,
                                                                           test_y.to_frame(name='hr_watch_rate'),
                                                                           columns, equations, targets, parameters)
        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:, ],
                                                           regr_train_y.ix[washout_time:, ])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:, ], regr_test_y.ix[washout_time:, ])

        performance_tr_dyn += mean_tr
        performance_tr_dyn_std += std_tr
        performance_te_dyn += mean_te
        performance_te_dyn_std += std_te

    overall_performance_tr_dyn = performance_tr_dyn/repeats
    overall_performance_tr_dyn_std = performance_tr_dyn_std/repeats
    overall_performance_te_dyn = performance_te_dyn/repeats
    overall_performance_te_dyn_std = performance_te_dyn_std/repeats

    scores_with_sd = [(overall_performance_tr_dyn, overall_performance_tr_dyn_std, overall_performance_te_dyn, overall_performance_te_dyn_std)]
    print scores_with_sd
    util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), scores_with_sd)
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_regression(['Dynamical'], feature_names, scores_over_all_algs)