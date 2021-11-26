#!/usr/bin/env python3
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from mnist_utils.util import _x, _y, _y_int, _a, _b_int
from sklearn import svm
from sklearn.metrics import accuracy_score, adjusted_rand_score
from tabulate import tabulate
import math
import numpy as np
import random as rn
import time

def learn(decision_function_shape="ovo", kernel="linear", max_iter=1):
    svm_classifier = svm.SVC(decision_function_shape=decision_function_shape, kernel=kernel, max_iter=max_iter)
    svm_classifier.fit(_x[:10000], _y_int[:10000])
    return svm_classifier

def test_accuracy(trained_svm):
    test1 = accuracy_score(_b_int, trained_svm.predict(_a))
    train1 = accuracy_score(_y_int, trained_svm.predict(_x))

    test2 = adjusted_rand_score(_b_int, trained_svm.predict(_a))
    train2 = adjusted_rand_score(_y_int, trained_svm.predict(_x))
    return test1, train1, test2, train2

def pipeline(max_iter_max=100, max_iter_coe=3):
    max_iter = -1
    kernels = ["linear", "rbf", "poly"]
    decision_function_shapes = ["ovo", "ovr"]
    result = []
    while True:
        for k in kernels:
            for d in decision_function_shapes:
                print("Trying with max_iter = ", max_iter, " kernel = ", k, " decision_function_shape = ", d)
                svc = learn(d, k, max_iter)
                test1, train1, test2, train2 = test_accuracy(svc)
                result.append([max_iter, k, d, test1, train1, test2, train2])

        max_iter *= max_iter_coe
        if max_iter >= max_iter_max:
            max_iter = -1
        if max_iter == -1:
            break

    print(tabulate(result, headers=['max_iter', 'kernel', 'shape', "TEST_P", "TRAIN_P", "TEST_RI", "TRAIN_RI"]))

pipeline()
