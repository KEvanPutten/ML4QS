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
dataset_path = './intermediate_datafiles/'

try:
    dataset = pd.read_csv(dataset_path + 'owndata_chapter4_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e
dataset.index = dataset.index.to_datetime()

# First let us use non hierarchical clustering.

clusteringNH = NonHierarchicalClustering()

# Let us look at k-means first.

k_values = range(2, 10)
silhouette_values = []

features = ['mag_phone_', 'gyro_phone_', 'acc_phone_', 'mag_phone_']

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
