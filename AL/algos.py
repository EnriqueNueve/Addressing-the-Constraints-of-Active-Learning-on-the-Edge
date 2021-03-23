# Import modules
import abc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor
from itertools import chain

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn import mixture
from skimage.feature import hog
from skimage.color import rgb2gray
from scipy.cluster.vq import vq

import matplotlib.pyplot as plt

#######################################################

class alAlgo(metaclass=abc.ABCMeta):
    """
    alAlgo() Documentation:
    --------------------------

    Purpose
    ----------
    Parent class that will be used for making new Active Learning algo classes.
    Currently, the class is very sparse. Will make adjustments as the project continues.

    Attributes
    ----------
    algo_name : str
        used to keep track of name of algo in engine.log

    sample_log : dict
        tracks what samples are chosen each round, places sample ids in list within dict

    round : int
        tracks what round algo is on

    predict_to_sample : bool
        bool that determines whether or not the algo needs the predictions of the model to choose which samples to label

    Methods
    -------
    @classmethod
    __subclasshook__(cls, subclass):
        Used to check if custom child class of alAlgo is properly made

    reset(self):
        set round=0 and sample_log={}

    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self, algo_name="NA"):
        self.algo_name = algo_name
        self.round = 0
        self.sample_log = {}
        self.predict_to_sample = False

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    def reset(self):
        self.round = 0
        self.sample_log = {}

    @abc.abstractmethod
    def __call__(self, cache: list, n: int, yh):
        """ Selects which samples to get labels for """
        raise NotImplementedError


#######################################################

class marginConfidence(alAlgo):
    """
    marginConfidence(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Score samples by predictions through formula MC(x)=(1-(P(y1*|x)-P(y2*|x)))

    Attributes
    ----------
    predict_to_sample : bool
        Determines if algo needs models prediction on cache to determine what samples from the cache to return

    Methods
    -------
    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self):
        super().__init__(algo_name="Margin Confidence")
        self.predict_to_sample = True
        self.feature_set = False
        self.single_output = False

    def __call__(self, cache: list, n: int, yh) -> list:

        # Check if embedded cache, then cache is available for the round
        if any(isinstance(i, list) for i in cache):
            try:
                cache = cache[self.round]
            except:
                raise ValueError("Active Learning Algo has iterated through each round\'s unlabled cache.")

        # Check if sample size is to large for cache
        if len(cache) < n:
            raise ValueError("Sample size n is larger than length of round's cache")

        # Calculate MC(x) values
        yh_vals = yh.iloc[:, 1:].values
        MC_vals = []
        for i in range(yh_vals.shape[0]):
            sample = yh_vals[i, :]
            sample[::-1].sort()
            y1, y2 = sample[0], sample[1]
            mc_val = 1 - (y1 - y2)
            MC_vals.append(mc_val)

        target_col_names = ["y" + str(i) for i in range(yh_vals.shape[1])]
        yh_col_names = ["MC", "ID"] + target_col_names
        yh = pd.concat([pd.DataFrame(MC_vals), yh], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_largest = yh.nlargest(n, 'MC')
        batch = n_largest["ID"].to_list()

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch

#######################################################

class uniformSample(alAlgo):
    """
    uniformSample(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Randomly samples over a uniform distribution of passed cache of data ids.
    Use as a baseline to compare the performance of your active learning algorithms.

    Attributes
    ----------
    predict_to_sample : bool
        Determines if algo needs models prediction on cache to determine what samples from the cache to return

    Methods
    -------
    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self):
        super().__init__(algo_name="Passive")
        self.predict_to_sample = False
        self.feature_set = False
        self.single_output = False

    def __call__(self, cache: list, n: int, yh=None) -> list:
        # Check if embedded cache, then cache is available for the round
        if any(isinstance(i, list) for i in cache):
            try:
                cache = cache[self.round]
            except:
                raise ValueError("Active Learning Algo has iterated through each round\'s unlabled cache.")

        # Check if sample size is to large for cache
        if len(cache) < n:
            raise ValueError("Sample size n is larger than length of round's cache")

        # Select from uniform distributions data ID's from given cache
        idx = random.sample(range(0, len(cache)), n)
        batch = [cache[i] for i in idx]

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("Selected samples: ")
        print(idx)
        print("\n")

        # Increment round
        self.round += 1

        return batch

#######################################################

class clusterMargin(alAlgo):
    """
    clusterMargin(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

    Attributes
    ----------
    predict_to_sample : bool
        Determines if algo needs models prediction on cache to determine what samples from the cache to return

    Methods
    -------
    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self, budget = 100 ,n_cluster=10, p = 0.5, sub_sample=500 , config = {"vector": "HOG", "cluster": "Kmeans",\
                                                                            "outlier": "IsoForest", "AdjP": 'linear'}):
        super().__init__(algo_name="clusterMargin")
        self.predict_to_sample = False
        self.feature_set = False
        self.single_output = False

        self.config = config
        self.budget = budget
        self.n_cluster = n_cluster
        self.p = p
        self.sub_sample = sub_sample


    def adjustP(self, mean):
        if self.config["AdjP"] == 'linear':
            self.p = mean
            return self.p

    def getCentroids(self, data):
        if self.config["cluster"] == "Kmeans":
            k_means = MiniBatchKMeans(init='k-means++', n_clusters=self.n_cluster, n_init=10)
            k_means.fit(data)
            return k_means.cluster_centers_
        elif self.config["cluster"] == "GM":
            clf = mixture.GaussianMixture(n_components=self.n_cluster, covariance_type='diag')
            clf.fit(data)
            means = clf.means_
            return means

    def getOutliers(self, data):
        if self.config["outlier"] == "IsoForest":
            clf = IsolationForest(max_samples=100)
            clf.fit(data)
            y_pred_train = clf.predict(data)
            y_outlier_id = np.where(y_pred_train == -1)
            return y_outlier_id[0].tolist()
        elif self.config["outlier"] == "OneClassSVM":
            clf = OneClassSVM(gamma='auto').fit(data)
            pred =  clf.score_samples(data)
            ids = np.argsort(pred)[:self.budget+1]
            return ids

    def getVectors(self, data):
        if self.config["vector"] == "HOG":
            data_feature = []
            for i in range(data.shape[0]):
                if data[i, :, :, :].shape[-1] > 1:
                    img_gray = rgb2gray(data[i, :, :, :])
                else:
                    img_gray = data[i, :, :, :]
                fd = hog(img_gray, visualize=False)
                data_feature.append(fd)
            data = np.stack(data_feature, axis=0)
            return data
        if self.config["vector"] == "FLAT":
            data_feature = []
            for i in range(data.shape[0]):
                if data[i, :, :, :].shape[-1] > 1:
                    img_gray = rgb2gray(data[i, :, :, :])
                else:
                    img_gray = data[i, :, :, :]
                img_gray = img_gray.flatten()
                data_feature.append(img_gray)
            data = np.stack(data_feature, axis=0)
            return data

    def cluster(self, data, cache, s=1):

        # Get data in vector format
        data = self.getVectors(data)

        # Cluster data
        centroids = self.getCentroids(data)
        model = MiniBatchKMeans(init='k-means++', n_clusters=self.n_cluster, n_init=10)
        model.cluster_centers_ = centroids

        # Select samples clostest to each cluster
        if self.n_cluster < self.budget:
            closest_points = []
            samples_per_cluster = int(self.budget / self.n_cluster)
            for i in range(self.n_cluster):
                dist = model.transform(data)[:, i]
                ind = np.argsort(dist)[::-1][:samples_per_cluster]
                closest_points.append(ind)
            closest = list(chain(*closest_points))
            closest_points = [cache[index] for index in closest]
            closest_points.sort()
        elif self.n_cluster == self.budget:
            closest, distances = vq(centroids, data)
            closest_points = [cache[index] for index in closest]
            closest_points.sort()

        # Check for duplicates
        s = set()
        duplicates = set(x for x in closest_points if x in s or s.add(x))
        print(duplicates)
        closest_points_filtered = list(set(closest_points))
        filtered_cache = list(set(cache) - set(closest_points))

        # If duplicates, select random points to make difference
        if len(closest_points) > len(closest_points_filtered ):
            r_points = random.sample(range(len(filtered_cache)),len(closest_points) - len(closest_points_filtered ) )
            r_points_picked = [filtered_cache[index] for index in r_points]
            closest_points_filtered = closest_points_filtered + r_points_picked

        print(len(closest_points_filtered),self.p)

        if s == 0:
            # Select outliers
            outliers = self.getOutliers(data)
            outliers = list(set(outliers) - set(closest_points))
            outliers = random.sample(outliers,self.budget)
            ids = outliers+closest_points
            return random.sample(ids, self.budget)
        else:
            return closest_points_filtered

    def __call__(self, batch) -> list:

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1
