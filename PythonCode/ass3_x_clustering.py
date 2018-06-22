from util.VisualizeDataset import VisualizeDataset
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
import copy
import pandas as pd
import matplotlib.pyplot as plot
import util.util as util


# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles_ass3/'

try:
    dataset = pd.read_csv(dataset_path + 'domain_features_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e
dataset.index = dataset.index.to_datetime()

# First let us use non hierarchical clustering.

clusteringNH = NonHierarchicalClustering()

#
## Do some initial runs to determine the right number for k
#

"""
print '===== k medoids clustering ====='
for f in features:
    print 'now doing feature: ', f
    best_k = 0
    best_silhouette = -1000
    for k in k_values:
        print 'k = ', k
        dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), [f+'x', f+'y', f+'z'], k, 'default', 20, n_inits=10)
        silhouette_score = dataset_cluster['silhouette'].mean()
        print 'silhouette = ', silhouette_score
        silhouette_values.append(silhouette_score)
        if silhouette_score > best_silhouette:
            best_k = k
            best_silhouette = silhouette_score

    # plot.plot(k_values, silhouette_values, 'b-')
    # plot.ylim([0,1])
    # plot.xlabel('k')
    # plot.ylabel('silhouette score')
    # plot.show()

    # And run k medoids with the highest silhouette score

    if f == 'mag_phone_':
        k = 8
    else:
        k = best_k

    dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), [f+'x', f+'y', f+'z'], k, 'default', 20, n_inits=50)
    DataViz.plot_clusters_3d(dataset_kmed, [f+'x', f+'y', f+'z'], 'cluster', ['label'])
    DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
    util.print_latex_statistics_clusters(dataset_kmed, 'cluster', [f+'x', f+'y', f+'z'], 'label')
"""

features = ['acc_', 'lin_acc_', 'gyro_', 'mag_']

# Let us look at k-means first.

k_values = range(2, 10)
silhouette_values = []

print '===== kmeans clustering ====='
for f in features:
    print 'now doing feature: ', f
    best_k = 0
    best_silhouette = -1000

    for k in k_values:
        print 'k = ', k
        dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), [f+'x', f+'y', f+'z'], k, 'default', 20, 10)
        silhouette_score = dataset_cluster['silhouette'].mean()
        print 'silhouette = ', silhouette_score
        silhouette_values.append(silhouette_score)

        if silhouette_score > best_silhouette:
            best_k = k
            best_silhouette = silhouette_score

        dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), [f + 'x', f + 'y', f + 'z'], k,
                                                             'default', 20, n_inits=50)
        DataViz.plot_clusters_3d(dataset_kmed, [f + 'x', f + 'y', f + 'z'], 'cluster', ['label'])
        DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
        util.print_latex_statistics_clusters(dataset_kmed, 'cluster', [f + 'x', f + 'y', f + 'z'], 'label')

# sil score > 0.7
# sil resemble eachother

exit(0)
# And run the knn with the highest silhouette score

k = 4

dataset_knn = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), ['acc_x', 'acc_y', 'acc_z'], k, 'default', 50, 50)
DataViz.plot_clusters_3d(dataset_knn, ['acc_x', 'acc_y', 'acc_z'], 'cluster', ['label'])
DataViz.plot_silhouette(dataset_knn, 'cluster', 'silhouette')
util.print_latex_statistics_clusters(dataset_knn, 'cluster', ['acc_x', 'acc_y', 'acc_z'], 'label')
del dataset_knn['silhouette']


k_values = range(2, 10)
silhouette_values = []


exit(0)

# Do some initial runs to determine the right number for k

print '===== k medoids clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k, 'default', 20, n_inits=10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)

plot.plot(k_values, silhouette_values, 'b-')
plot.ylim([0,1])
plot.xlabel('k')
plot.ylabel('silhouette score')
plot.show()

# And run k medoids with the highest silhouette score

k = 6

dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k, 'default', 20, n_inits=50)
DataViz.plot_clusters_3d(dataset_kmed, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], 'cluster', ['label'])
DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
util.print_latex_statistics_clusters(dataset_kmed, 'cluster', ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], 'label')

# And the hierarchical clustering is the last one we try

clusteringH = HierarchicalClustering()

k_values = range(2, 10)
silhouette_values = []

# Do some initial runs to determine the right number for the maximum number of clusters.

print '===== agglomaritive clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster, l = clusteringH.agglomerative_over_instances(copy.deepcopy(dataset), ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k, 'euclidean', use_prev_linkage=True, link_function='ward')
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)
    if k == k_values[0]:
        DataViz.plot_dendrogram(dataset_cluster, l)

plot.plot(k_values, silhouette_values, 'b-')
plot.ylim([0,1])
plot.xlabel('max number of clusters')
plot.ylabel('silhouette score')
plot.show()

# And we select the outcome dataset of the knn clustering....

dataset_knn.to_csv(dataset_path + 'chapter5_result.csv')
