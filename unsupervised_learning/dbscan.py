from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from util import _x, _y_int
import math
import numpy as np
import random as rand
#global vars

clustering = None
label_count = len(np.unique(_y_int))
slice_size = 6000
def classify_clusters(l1, l2):
    ref_labels = {}
    m = np.unique(l1).max() + 1
    for i in range(l1.size):
        if l1[i] == -1:
            l1[i] = m
    for i in range(len(np.unique(l1))):
        index = np.where(l1 == i,1,0)
        temp = np.bincount(l2[index==1])
        ref_labels[i] = temp.argmax()
    decimal_labels = np.zeros(len(l1))
    for i in range(len(l1)):
        decimal_labels[i] = ref_labels[l1[i]]
    return decimal_labels

def init_clustring_scikit(epsilon=2, min_samples=2):
    global clustering, slice_size
    indexes = np.random.choice(len(_x), size=slice_size, replace=False)
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(_x[indexes])
    clustering.fit(x_train)
    print(clustering.labels_)
    return _y_int[indexes]

def test_accuracy_scikit(labels):
    global clustering
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    decimal_labels = classify_clusters(clustering.labels_, labels)
    print("predicted labels:\t", decimal_labels[:16].astype('int'))
    print("true labels:\t\t", labels[:16])
    print(60 * '_')
    AP = accuracy_score(decimal_labels,labels)
    RI = adjusted_rand_score(decimal_labels,labels)
    print("Accuracy (PURITY):" , AP)
    print("Accuracy (RAND INDEX):" , RI)
    return AP, RI, len(np.unique(clustering.labels_))

def pipeline(epsilon_max=50, min_samples_max=50, coefficient=2):
    epsilon = 1
    min_samples = 1
    result = []
    AP = None
    RI = None
    while epsilon <= epsilon_max:
        while min_samples <= min_samples_max:
            print(10 * "*" + "TRYING WITH " + str(epsilon) + " " + str(min_samples) + 10 * "*")
            labels = init_clustring_scikit(epsilon, min_samples)
            AP, RI, n= test_accuracy_scikit(labels)
            result.append([epsilon, min_samples, AP, RI, n])
            min_samples *= coefficient
            min_samples = math.ceil(min_samples)
        min_samples = 1
        epsilon *= coefficient
        epsilon = math.ceil(epsilon)
    print(tabulate(result, headers=['epsilon', 'min_samples', 'AP', 'RI', 'Cluster Count']))

pipeline(coefficient=1.2)
