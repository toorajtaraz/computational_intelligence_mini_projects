from util import _x, _y_int
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import accuracy_score, adjusted_rand_score
import numpy as np
from tabulate import tabulate
import math
import random as rand
#global vars

clustering = None

def classify_clusters(l1, l2):
    ref_labels = {}
    for i in range(len(np.unique(l1))):
        index = np.where(l1 == i,1,0)
        temp = np.bincount(l2[index==1]).argmax()
        ref_labels[i] = temp
    decimal_labels = np.zeros(len(l1))
    for i in range(len(l1)):
        decimal_labels[i] = ref_labels[l1[i]]
    return decimal_labels

def init_clustring_scikit(bandwidth=2, slice_size=6000):
    global clustering
    indexes = np.random.choice(len(_x), size=slice_size, replace=False)
    clustering = MeanShift(bandwidth=bandwidth, n_jobs=12)
    clustering.fit(_x[indexes])
    return _y_int[indexes]

def test_accuracy_scikit(labels):
    global clustering
    decimal_labels = classify_clusters(clustering.labels_, labels)
    print("NUMBER OF CLUSTERS:", len(np.unique(clustering.labels_)))
    print("predicted labels:\t", decimal_labels[:16].astype('int'))
    print("true labels:\t\t", labels[:16])
    print(60 * '_')
    AP = accuracy_score(decimal_labels,labels)
    RI = adjusted_rand_score(decimal_labels,labels)
    print("Accuracy (PURITY):" , AP)
    print("Accuracy (RAND INDEX):" , RI)
    return AP, RI, len(np.unique(clustering.labels_))

def pipeline(bandwidth_max=100, coefficient=2):
    bandwidth = 2
    result = []
    AP = None
    RI = None
    while bandwidth <= bandwidth_max:
        print(10 * "*" + "TRYING WITH " + str(bandwidth) + 10 * "*")
        labels = init_clustring_scikit(bandwidth)
        AP, RI, n= test_accuracy_scikit(labels)
        result.append([bandwidth, AP, RI, n])
        bandwidth *= coefficient
        bandwidth = math.ceil(bandwidth)
    print(tabulate(result, headers=['BandWidth', 'AP', 'RI', 'Cluster Count']))

pipeline(bandwidth_max=100, coefficient=1.2)
