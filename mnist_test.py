from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo

import tensorflow as tf

#######################################################

NTEST = 2

def testPassive():
    for i in range(0,NTEST):
        np.random.seed(i+1)
        # DataManager parameters
        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False
        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        algo = algos.uniformSample()
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/mnist/passive/"+str(i)+"_log.csv")

def testMargin():
    for i in range(0,NTEST):
        np.random.seed(i+1)
        # DataManager parameters
        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        algo = algos.marginConfidence()
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/mnist/margin/"+str(i)+"_log.csv")

def testA():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "GM", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/1/"+str(i)+"_log.csv")

def testB():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "GM", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/2/"+str(i)+"_log.csv")

def testC():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "Kmeans", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/3/"+str(i)+"_log.csv")

def testD():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "Kmeans", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/4/"+str(i)+"_log.csv")

def testE():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "GM", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/5/"+str(i)+"_log.csv")

def testF():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "GM", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/6/"+str(i)+"_log.csv")

def testG():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "Kmeans", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/7/"+str(i)+"_log.csv")

def testH():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.00144, .2, .79856)  # (train, val, unlabeled)
        # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "Kmeans", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.8, sub_sample=500, config = c)
        algo.reset()

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("mnist_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/cm/8/"+str(i)+"_log.csv")

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('results/mnist'):
        os.makedirs('results/mnist')

    testPassive()
    testMargin()
    testA()
    testB()
    testC()
    testD()
    testE()
    testF()
    testG()
    testH()

main()
