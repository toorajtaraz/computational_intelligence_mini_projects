from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score
from tabulate import tabulate
from mnist_utils.util import _x, _y_int
import math
import numpy as np
import random as rand

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

def init_clustring_scikit(linkage, slice_size=10000):
    global clustering
    indexes = np.random.choice(len(_x), size=slice_size, replace=False)
    clustering = AgglomerativeClustering(n_clusters=10, affinity="euclidean", compute_full_tree=True, linkage=linkage)
    clustering.fit(_x[indexes])
    return _y_int[indexes]

def test_accuracy_scikit(labels):
    global clustering
    decimal_labels = classify_clusters(clustering.labels_, labels)
    print("predicted labels:\t", decimal_labels[:16].astype('int'))
    print("true labels:\t\t", labels[:16])
    print(60 * '_')
    AP = accuracy_score(decimal_labels,labels)
    RI = adjusted_rand_score(decimal_labels,labels)
    print("Accuracy (PURITY):" , AP)
    print("Accuracy (RAND INDEX):" , RI)
    return AP, RI

def pipeline(linkage=["ward", "single", "average", "complete"]):
    result = []
    AP = None
    RI = None
    for x in linkage:
        print(10 * "*" + "TRYING WITH " + x + 10 * "*")
        labels = init_clustring_scikit(x)
        AP, RI = test_accuracy_scikit(labels)
        result.append([x, AP, RI])
    print(tabulate(result, headers=['linkage', 'AP', 'RI']))

pipeline()
