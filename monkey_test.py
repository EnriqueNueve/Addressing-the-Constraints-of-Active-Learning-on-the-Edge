from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo

import tensorflow as tf

#######################################################

NTEST = 3

def testPassive():
    for i in range(0,NTEST):
        np.random.seed(i+1)
        # DataManager parameters
        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False
        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        algo = algos.uniformSample()
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/passive/"+str(i)+"_log.csv")

def testMargin():
    for i in range(0,NTEST):
        np.random.seed(i+1)
        # DataManager parameters
        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        algo = algos.marginConfidence()
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/margin/"+str(i)+"_log.csv")

def testA():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "GM", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/1/"+str(i)+"_log.csv")

def testB():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "GM", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/2/"+str(i)+"_log.csv")

def testC():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "Kmeans", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/3/"+str(i)+"_log.csv")

def testD():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "HOG", "cluster": "Kmeans", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/4/"+str(i)+"_log.csv")

def testE():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "GM", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/5/"+str(i)+"_log.csv")

def testF():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265) # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "GM", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/6/"+str(i)+"_log.csv")

def testG():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265) # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "Kmeans", "outlier": "IsoForest","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/7/"+str(i)+"_log.csv")

def testH():
    for i in range(0,NTEST):
        np.random.seed(i+1)

        split = (.0735, .2, .7265)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = monkeyLoader(bins, keep_bins)   # Declare data manager class
        dataClass.loadCaches()

        c = {"vector": "FLAT", "cluster": "Kmeans", "outlier": "OneClassSVM","AdjP": 'linear'}
        algo = algos.clusterMargin(budget= 100 ,n_cluster=30, p=.5, sub_sample=500, config = c)
        algo.reset()

        modelName = "monkeyCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)
        engine.loadModel("monkey_model")
        engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=False)
        engine.saveLog(path="results/monkey/cm/8/"+str(i)+"_log.csv")

def main():
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
