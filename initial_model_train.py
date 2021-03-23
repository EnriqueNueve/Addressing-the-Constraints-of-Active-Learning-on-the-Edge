"""
Purpose: Create initial models and split data.
"""

from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo

import os
import tensorflow as tf

#######################################################

def make_MNIST_model():
    # | ----------------------------
    # | 1. Select data
    # | ---------------------------

    # DataManager parameters
    split = (.00144, .2, .79856)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = mnistLoader(bins, keep_bins)  # Declare data manager class
    dataClass.parseData(split, bins, keep_bins)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    algo = algos.uniformSample()
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "mnistCNN"  # Pick pre-made model
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
    zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 100
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    engine.initialTrain(epochs=30, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=True)

    engine.saveModel("Models/mnist_model")

#######################################################

def make_Bee_model():
    # | ----------------------------
    # | 1. Select data
    # | ---------------------------

    # DataManager parameters
    split = (.0212, .2, .7788)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = beeLoader(bins, keep_bins)  # Declare data manager class
    dataClass.parseData(split, bins, keep_bins)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    algo = algos.uniformSample()
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "beeCNN"  # Pick pre-made model
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
    zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 100
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    engine.initialTrain(epochs=30, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=True)

    engine.saveModel("Models/bee_model")

#######################################################

def make_Monkey_model():
    # | ----------------------------
    # | 1. Select data
    # | ---------------------------

    # DataManager parameters
    split = (.0735, .2, .7265)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = monkeyLoader(bins, keep_bins)  # Declare data manager class
    dataClass.parseData(split, bins, keep_bins)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    algo = algos.uniformSample()
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "monkeyCNN"  # Pick pre-made model
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
    zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 50
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    engine.initialTrain(epochs=30, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=True)

    engine.saveModel("Models/monkey_model")

#######################################################

if __name__ == "__main__":
    # Clean Model folder
    if not os.path.exists('Models'):
        os.makedirs('Models')
    if os.path.isfile('Models/mnist_model.h5'):
        os.remove('Models/mnist_model.h5')
    if os.path.isfile('Models/bee_model.h5'):
        os.remove('Models/bee_model.h5')
    if os.path.isfile('Models/monkey_model.h5'):
        os.remove('Models/monkey_model.h5')

    # Clean MNIST data folder
    if os.path.isfile('Data/DataSets/MNIST/ul_cache.csv'):
        os.remove('Data/DataSets/MNIST/ul_cache.csv')
    if os.path.isfile('Data/DataSets/MNIST/train_cache_.csv'):
        os.remove('Data/DataSets/MNIST/train_cache_.csv')
    if os.path.isfile('Data/DataSets/MNIST/val_cache_.csv'):
        os.remove('Data/DataSets/MNIST/val_cache_.csv')

    # Clean Bees data folder
    if os.path.isfile('Data/DataSets/Bees/ul_cache.csv'):
        os.remove('Data/DataSets/Bees/ul_cache.csv')
    if os.path.isfile('Data/DataSets/Bees/train_cache_.csv'):
        os.remove('Data/DataSets/Bees/train_cache_.csv')
    if os.path.isfile('Data/DataSets/Bees/val_cache_.csv'):
        os.remove('Data/DataSets/Bees/val_cache_.csv')

    # Clean Monkey data folder
    if os.path.isfile('Data/DataSets/Monkey/ul_cache.csv'):
        os.remove('Data/DataSets/Monkey/ul_cache.csv')
    if os.path.isfile('Data/DataSets/Monkey/train_cache_.csv'):
        os.remove('Data/DataSets/Monkey/train_cache_.csv')
    if os.path.isfile('Data/DataSets/Monkey/val_cache_.csv'):
        os.remove('Data/DataSets/Monkey/val_cache_.csv')

    # Train initial models
    make_MNIST_model()
    make_Bee_model()
    make_Monkey_model()
