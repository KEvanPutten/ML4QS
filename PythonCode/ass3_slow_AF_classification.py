from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
import os


# Of course we repeat some stuff from Chapter 3, namely to load the dataset
DataViz = VisualizeDataset()
# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles_ass3/'
export_tree_path = 'Example_graphs/ass3/'
try:
    dataset = pd.read_csv(dataset_path + 'clustering_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous scripts first!')
    raise e

if not os.path.exists(export_tree_path):
    os.makedirs(export_tree_path)

dataset.index = dataset.index.to_datetime()


# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)
#train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

basic_features = ['acc_x','acc_y','acc_z','lin_acc_x','lin_acc_y','lin_acc_z','mag_x','mag_y','mag_z','gyr_x','gyr_y',
                  'gyr_z','light_illuminance','loc_latitude','loc_height','loc_velocity']
pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9']
time_features = [name for name in dataset.columns if '_temp_' in name]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print '#basic features: ', len(basic_features)
print '#PCA features: ', len(pca_features)
print '#time features: ', len(time_features)
print '#frequency features: ', len(freq_features)
cluster_features = ['cluster']
print '#cluster features: ', len(cluster_features)
features_after_outliers_and_imputation = list(set().union(basic_features, pca_features))
features_after_domain_features = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_cluster_features = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))


# Based on the plot we select the top 10 features.
selected_features_with_NB = ['acc_x_temp_MAD_ws_120', 'mag_z_freq_0.0_Hz_ws_40', 'mag_y_freq_0.0_Hz_ws_40',
                             'lin_acc_y_freq_0.0_Hz_ws_40', 'pca_7_temp_mean_ws_120', 'acc_z_temp_std_ws_120', 'acc_y',
                             'pca_9_temp_MAD_ws_120', 'mag_x_freq_0.1_Hz_ws_40', 'mag_z_freq_1.1_Hz_ws_40']

selected_features_with_KNN = ['mag_y_freq_0.8_Hz_ws_40', 'pca_7_temp_std_ws_120', 'mag_x_max_freq',
                              'gyr_z_freq_2.0_Hz_ws_40', 'gyr_y_freq_0.0_Hz_ws_40', 'mag_z_freq_1.5_Hz_ws_40',
                              'acc_z_temp_MAD_ws_120', 'acc_y_temp_kurtosis_ws_120', 'mag_x_freq_1.2_Hz_ws_40',
                              'lin_acc_y_freq_1.8_Hz_ws_40']

selected_features_with_DT = ['acc_z_freq_0.0_Hz_ws_40', 'loc_height_temp_mean_ws_120', 'pca_4_temp_kurtosis_ws_120',
                             'lin_acc_y_temp_kurtosis_ws_120', 'pca_1_temp_kurtosis_ws_120', 'acc_z_temp_MAD_ws_120',
                             'mag_x_freq_1.2_Hz_ws_40', 'gyr_z_freq_2.0_Hz_ws_40', 'acc_y_temp_kurtosis_ws_120',
                             'lin_acc_y_freq_0.6_Hz_ws_40']

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

possible_feature_sets = [basic_features, features_after_outliers_and_imputation, features_after_domain_features,
                         features_after_cluster_features, selected_features_with_DT, selected_features_with_NB]
feature_names = ['initial set', 'After imputation', 'With Domain features', 'With cluster features', 'Selected features DT', 'Selected features NB']

#possible_feature_sets = [basic_features, selected_features_with_DT, selected_features_with_NB, selected_features_with_KNN]
#feature_names = ['initial set', 'Selected features DT', 'Selected features NB', 'Selected features KNN']
repeats = 4

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):
    print "working on feature set", feature_names[i]
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_nn = 0
    performance_tr_rf = 0
    performance_tr_svm = 0
    performance_te_nn = 0
    performance_te_rf = 0
    performance_te_svm = 0

    for repeat in range(0, repeats):
        print repeat
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(selected_train_X, train_y, selected_test_X, gridsearch=True, alpha=10)
        performance_tr_nn += eval.accuracy(train_y, class_train_y)
        performance_te_nn += eval.accuracy(test_y, class_test_y)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(selected_train_X, train_y, selected_test_X, gridsearch=True)
        performance_tr_rf += eval.accuracy(train_y, class_train_y)
        performance_te_rf += eval.accuracy(test_y, class_test_y)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(selected_train_X, train_y, selected_test_X, gridsearch=True)
        performance_tr_svm += eval.accuracy(train_y, class_train_y)
        performance_te_svm += eval.accuracy(test_y, class_test_y)

    overall_performance_tr_nn = performance_tr_nn/repeats
    overall_performance_te_nn = performance_te_nn/repeats
    overall_performance_tr_rf = performance_tr_rf/repeats
    overall_performance_te_rf = performance_te_rf/repeats
    overall_performance_tr_svm = performance_tr_svm/repeats
    overall_performance_te_svm = performance_te_svm/repeats

    # And we run our deterministic classifiers:
    print "now running deterministic classifiers"

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(selected_train_X, train_y, selected_test_X, gridsearch=True)
    performance_tr_knn = eval.accuracy(train_y, class_train_y)
    performance_te_knn = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(selected_train_X, train_y, selected_test_X, gridsearch=True)
    performance_tr_dt = eval.accuracy(train_y, class_train_y)
    performance_te_dt = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(selected_train_X, train_y, selected_test_X)
    performance_tr_nb = eval.accuracy(train_y, class_train_y)
    performance_te_nb = eval.accuracy(test_y, class_test_y)

    scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index),
                                                       len(selected_test_X.index),
                                                       [(overall_performance_tr_nn, overall_performance_te_nn),
                                                        (overall_performance_tr_rf, overall_performance_te_rf),
                                                        (overall_performance_tr_svm, overall_performance_te_svm),
                                                        (performance_tr_knn, performance_te_knn),
                                                        (performance_tr_dt, performance_te_dt),
                                                        (performance_tr_nb, performance_te_nb)])
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_classification(['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)

exit(0)