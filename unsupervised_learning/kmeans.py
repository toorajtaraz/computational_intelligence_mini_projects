from mnist_utils.util import _x, _y_int
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
import numpy as np
from fast_pytorch_kmeans import KMeans
import torch
from tabulate import tabulate
#global vars

kmeans_main = None
cluster_ids_x = None

def classify_clusters(l1, l2):
    ref_labels = {}
    for i in range(len(np.unique(l1))):
        index = np.where(l1 == i,1,0)
        ref_labels[i] = np.bincount(l2[index==1]).argmax()
    decimal_labels = np.zeros(len(l1))
    for i in range(len(l1)):
        decimal_labels[i] = ref_labels[l1[i]]
    return decimal_labels

def init_clustring_scikit(cluster_count=10):
    global kmeans_main
    kmeans_main = MiniBatchKMeans(n_clusters=cluster_count, verbose=False)
    kmeans_main.fit(_x)

def test_accuracy_scikit():
    global kmeans_main
    decimal_labels = classify_clusters(kmeans_main.labels_, _y_int)
    print("predicted labels:\t", decimal_labels[:16].astype('int'))
    print("true labels:\t\t",_y_int[:16])
    print(60 * '_')
    AP = accuracy_score(decimal_labels,_y_int)
    RI = adjusted_rand_score(decimal_labels,_y_int)
    print("Accuracy (PURITY):" , AP)
    print("Accuracy (RAND INDEX):" , RI)
    return AP, RI


def init_clustring_torch(cluster_count=10):
    global clusters_from_label, cluster_ids_x
    _kmeans = KMeans(n_clusters=cluster_count, mode='euclidean', verbose=1)
    x = torch.from_numpy(_x)
    cluster_ids_x = _kmeans.fit_predict(x)

def test_accuracy_torch():
    global cluster_ids_x 
    decimal_labels = classify_clusters(cluster_ids_x.cpu().detach().numpy(), _y_int)
    print("predicted labels:\t", decimal_labels[:16].astype('int'))
    print("true labels:\t\t",_y_int[:16])
    print(60 * '_')
    AP = accuracy_score(decimal_labels,_y_int)
    RI = adjusted_rand_score(decimal_labels,_y_int)
    print("Accuracy (PURITY):" , AP)
    print("Accuracy (RAND INDEX):" , RI)
    return AP, RI

def pipeline(lib="torch", cluster_count_max=300, coefficient=2):
    cluster_count = len(np.unique(_y_int))
    result = []
    if lib == "torch":
        while cluster_count <= cluster_count_max:
            print(10 * "*" + "TRYING WITH " + str(cluster_count) + 10 * "*")
            init_clustring_torch(cluster_count)
            AP, RI = test_accuracy_torch() 
            result.append([cluster_count, AP, RI])
            cluster_count *= coefficient
            cluster_count = int(cluster_count)
    elif lib == "scikit":
        while cluster_count <= cluster_count_max:
            print(10 * "*" + "TRYING WITH " + str(cluster_count) + 10 * "*")
            init_clustring_scikit(cluster_count)
            AP, RI = test_accuracy_scikit() 
            result.append([cluster_count, AP, RI])
            cluster_count *= coefficient
            cluster_count = int(cluster_count)
    else:
        print("LIB NOT SUPPORTED")

    print(tabulate(result, headers=['K', 'AP', 'RI']))

pipeline(cluster_count_max=200, coefficient=3, lib="scikit")
