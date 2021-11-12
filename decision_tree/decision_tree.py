from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from mnist_utils.util import _x, _y, _y_int, _a, _b_int
from graphviz import Source
from sklearn import tree
import numpy as np
import random as rn
from sklearn.metrics import accuracy_score, adjusted_rand_score
import time
import math
from tabulate import tabulate
def learn(seed=0, criterion="entropy", splitter="best", min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, max_depth=None):
    dtc = tree.DecisionTreeClassifier(random_state=seed, criterion=criterion, splitter=splitter,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      min_impurity_decrease=min_impurity_decrease, max_depth=max_depth)
    dtc = dtc.fit(_x, _y_int)
    return dtc

def test_accuracy(trained_tree):
    test1 = accuracy_score(_b_int, trained_tree.predict(_a))
    train1 = accuracy_score(_y_int, trained_tree.predict(_x))

    test2 = adjusted_rand_score(_b_int, trained_tree.predict(_a))
    train2 = adjusted_rand_score(_y_int, trained_tree.predict(_x))
    return test1, train1, test2, train2

def pipeline(mss_rate=10, msl_rate=10, mid_rate=0.002, md_rate=10, mss_max=100, msl_max=100, mid_max=1, md_max=100, criterion="entropy", splitter="best"):
    seed = math.ceil(time.time_ns() / 10000000000)
    np.random.seed(seed)
    rn.seed(seed)
    min_samples_split = 2
    min_samples_leaf = 1
    min_impurity_decrease = 0.0
    max_depth = 1
    result = []
    while max_depth is None or max_depth <= md_max:
        while min_impurity_decrease <= mid_max:
            while min_samples_leaf <= msl_max:
                while min_samples_split <= mss_max:
                    print("Trying with MSS = ", min_samples_split, " MSL = ", min_samples_leaf, " MID = ", min_impurity_decrease, " MD = ", max_depth)
                    dtc = learn(seed=seed, criterion=criterion, splitter=splitter,
                                min_impurity_decrease=min_impurity_decrease, min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split, max_depth=max_depth)
                    test1, train1, test2, train2 = test_accuracy(dtc)
                    result.append([min_samples_split, min_samples_leaf, min_impurity_decrease, max_depth, test1, train1, test2, train2])
                    min_samples_split = math.ceil(min_samples_split * mss_rate)
                min_samples_leaf = math.ceil(min_samples_leaf * msl_rate)
                min_samples_split = 2
            min_impurity_decrease += mid_rate
            min_samples_split = 2
            min_samples_leaf = 1
        if max_depth is None:
            break
        max_depth = math.ceil(max_depth * md_rate)
        if max_depth > md_max:
            max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        min_impurity_decrease = 0.0

    print(tabulate(result, headers=['MSS', 'MSL', 'MID', 'MD', "TEST_P", "TRAIN_P", "TEST_RI", "TRAIN_RI"]))
pipeline(1.8, 1.8, 0.5, 8, 100, 100, 2, 100)
#pipeline(15, 15, 1.5, 15, 100, 100, 3, 100)
