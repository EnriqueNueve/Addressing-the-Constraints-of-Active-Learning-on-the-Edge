# Import modules
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from math import floor
from random import shuffle
from itertools import chain
from tqdm import trange
import matplotlib.pyplot as plt
import random
import copy

#######################################################

def testCallEngine():
    print("Called test function from ~/engine.py")

#######################################################

class Engine():
    """
    Engine() Documentation:
    --------------------------

    Purpose
    ----------
    Control experiment for active learning. Schedules when to use the data loader, the model
    and the active learning algo to sample.

    Attributes
    ----------
    algoClass : alAlgo()
        used to decide what samples to take each round
    dataClass : DataManager()
        used to fetch and pre-process data for each round or initial training
    zooKeeper: zooKeeper()
        used to store tensorflow model and methods for training and testing the model.

    Methods
    -------
    trainBatch(self, cycles, batch_size):
        Used to call zooKeeper to perform training of tf model

    getLog(self):
        Creates dictionary to keep track of performance metrics

    updateLog(self, round=None, time_round=None, batch_size=None, train_loss=None, val_loss=None,
                  train_metrics=None, val_metrics=None):
        Updates metrics at end of each iteration within initialTrain() and run()

    initialTrain(self, epochs, batch_size, val=True, plot=False:
        Used to train tf model from zooKeeper on initially provided training data

    valTest(self, batch_size):
        Used to check model's loss on validation data set

    plotLog(self, logs, xlabel, ylabel, title, labels):
        Plots tracked metrics. To modify plot, do so here.

    run(self, rounds, cycles, batch_size, val=True, plot=False):
        Used to track and update cache of each active learning round. Controls runCycle()
        method which is what calls for data, model update, and update cache. If you wanted to
        have very very very precise control of what happens between training rounds, modify code here.
        Warning it is very intricate and could easily break the code, do so at your own risk.

    runCycle(self, batch_size, cycles):
        Subroutine for run where model is trained on batch from algo, metrics logged, and caches updated

    trainBatch(self, cycles, batch_size):
        Used to train model to update weights.

    valTest(self, batch_size):
        Call model and get metrics of prediction.

    evalCache(self, cache, batch_size):
        Calls model and performs prediction and returns predictions.

    saveLog(self, path):
        Save log of type dict to a csv

    plotLog(self, logs, xlabel, ylabel, title, labels):
        Plots loss on validation data after train and active learning

    initialTrain(self, epochs, batch_size, val=True, plot=False):
        Trains model on data present in train cache

    """

    def __init__(self, algoClass, dataClass, zooKeeper, sample_size):
        self.intialTrain_metric_log = {}
        self.intialVal_metric_log = {}
        self.train_metric_log = {}
        self.val_metric_log = {}

        # Declare passed objects
        self.algoClass = algoClass  # Class for Active Learning algo with method to select batches
        self.dataClass = dataClass  # Data class that stores cache df and has method to load in data based off of algoClass batch
        self.modelManager = zooKeeper  # Model class that stores model and has train and predict methods

        self.round = 0  # Keep track of what round of training the system is on
        self.sample_size = sample_size

        self.val_best = 0

        self.val_track = None

        self.multipleBins = False
        if self.dataClass.bins > 1:
            self.multipleBins = True

        self.log = self.getLog()  # Make log to track active learning

        self.test_cache = []

    def getLog(self):
        """ Builds dictionary with proper keys matching to metrics being tracked """
        keys = ["round", "data_set", "algo", "model",
                "time_round", "batch_size", "n_train", "n_val"]

        # Add col names for unlabeled caches
        if self.dataClass.bins == 1:
            temp_col_names = ["u_cache"]
        else:
            temp_col_names = ["u_cache_" + str(i) for i in range(self.dataClass.bins)]
        keys.extend(temp_col_names)

        # Add col names for loss metric
        temp_col_names = ["loss_train_" + self.modelManager.modelObject.loss.__name__,
                          "loss_val_" + self.modelManager.modelObject.loss.__name__]
        keys.extend(temp_col_names)

        # Add col names for extra metrics
        temp_col_names = ["train_" + metric.name for i, metric in enumerate(self.modelManager.metrics)]
        keys.extend(temp_col_names)
        temp_col_names = ["val_" + metric.name for i, metric in enumerate(self.modelManager.metrics)]
        keys.extend(temp_col_names)

        log = {k: [] for k in keys}
        return log

    def updateLog(self, round=None, time_round=None, batch_size=None, train_loss=None, val_loss=None,
                  train_metrics=None, val_metrics=None):
        """ Updates metrics within log """
        # Add round to log
        if round == None:
            self.log["round"].append("NA")
        else:
            self.log["round"].append(round)

        # Add time to run round to log
        if time_round == None:
            self.log["time_round"].append("NA")
        else:
            self.log["time_round"].append(time_round)

        # Add name of data set to log
        self.log["data_set"].append(self.dataClass.dataset_name)

        # Add active learning algo name to log
        self.log["algo"].append(self.algoClass.algo_name)

        # Add model name to log
        self.log["model"].append(self.modelManager.modelName)

        # Add batch size to log
        if batch_size == None:
            self.log["batch_size"].append("NA")
        else:
            self.log["batch_size"].append(batch_size)

            # Add number of samples in train cache
        self.log["n_train"].append(len(self.dataClass.train_cache))

        # Add number of samples in val cache
        self.log["n_val"].append(len(self.dataClass.val_cache))

        # Add number of samples in unlabeled caches
        if self.dataClass.bins > 1:
            for i, sub_list in enumerate(self.dataClass.unlabeled_cache):
                self.log["u_cache_" + str(i)].append(len(sub_list))
        else:
            self.log["u_cache"].append(len(self.dataClass.unlabeled_cache))

        # Add loss of train to log
        if train_loss == None:
            self.log["loss_train_" + self.modelManager.modelObject.loss.__name__].append("NA")
        else:
            self.log["loss_train_" + self.modelManager.modelObject.loss.__name__].append(train_loss)

            # Add loss of val to log
        if val_loss == None:
            self.log["loss_val_" + self.modelManager.modelObject.loss.__name__].append("NA")
        else:
            self.log["loss_val_" + self.modelManager.modelObject.loss.__name__].append(val_loss)

        # Add train metrics
        if train_metrics == None:
            for i, metric in enumerate(self.modelManager.modelObject.metrics):
                self.log["train_" + metric.name].append('NA')
        else:
            for i, metric in enumerate(self.modelManager.modelObject.metrics):
                self.log["train_" + metric.name].append(train_metrics[i])

        # Add val metrics
        if val_metrics == None:
            for i, metric in enumerate(self.modelManager.modelObject.metrics):
                self.log["val_" + metric.name].append('NA')
        else:
            for i, metric in enumerate(self.modelManager.modelObject.metrics):
                self.log["val_" + metric.name].append(val_metrics[i])

    def trainBatch(self, cycles, batch_size):
        """ Calls model to predict, gets error, apply gradients to update weights """
        remainder_samples = len(
            self.dataClass.train_cache) % batch_size  # Calculate number of remainder samples from batches
        total_loss = []
        shuffle_cache = list(self.dataClass.train_cache)
        random.shuffle(shuffle_cache)

        # Run batches
        for i in range(cycles):
            for batch in trange(floor(len(self.dataClass.train_cache) / batch_size)):
                batch_ids = self.dataClass.train_cache[batch_size * batch:batch_size * (batch + 1)]
                X, y = self.dataClass.getBatch(batch_ids)

                # Try change y dtype
                y = y.astype('float32')

                loss = self.modelManager.modelObject.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                batch_ids = self.dataClass.train_cache[(-1) * remainder_samples:]
                X, y = self.dataClass.getBatch(batch_ids)

                # Try change y dtype
                y = y.astype('float32')

                loss = self.modelManager.modelObject.trainBatch(X, y)
                total_loss.append(loss)

            # Check if val accuracy up or down
            if self.val_track != None:
                val_total_loss, total_scores = self.valTest(batch_size)
                keys = list(total_scores[0].keys())
                val_metrics = []
                for i, k in enumerate(keys):
                    samples = []
                    for j, sample in enumerate(total_scores):
                        samples.append((sample[k]))
                    samples = np.mean(samples)
                    val_metrics.append(samples)

                val_track_key = keys.index(self.val_track)
                val_score = val_metrics[val_track_key]
                if val_score > self.val_best:
                    self.modelManager.modelObject.model.save_weights("tmp/val_best_weights.h5")
                    print("Val accuracy improved from {} to {}".format(self.val_best, val_score))
                    self.val_best = val_score


        return total_loss

    def valTest(self, batch_size):
        """ Call model, get metrics of prediction """
        remainder_samples = len(
            self.dataClass.val_cache) % batch_size  # Calculate number of remainder samples from batches
        total_loss = []
        total_scores = []

        # Run batches
        for batch in trange(floor(len(self.dataClass.val_cache) / batch_size)):
            batch_ids = self.dataClass.val_cache[batch_size * batch:batch_size * (batch + 1)]
            X, y = self.dataClass.getBatch(batch_ids)

            # Try change y dtype
            y = y.astype('float32')

            loss, scores, _ = self.modelManager.modelObject.eval(X, y)
            total_loss.append(loss)
            total_scores.append(scores)

        # Run remainders
        if remainder_samples > 0:
            batch_ids = self.dataClass.val_cache[(-1) * remainder_samples:]
            X, y = self.dataClass.getBatch(batch_ids)

            # Try change y dtype
            y = y.astype('float32')

            loss, scores, _ = self.modelManager.modelObject.eval(X, y)
            total_loss.append(loss)
            total_scores.append(scores)

        return total_loss, total_scores

    def saveLog(self, path):
        """ Save log of type dict to a csv """
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(path)

    def plotLog(self, logs, xlabel, ylabel, title, labels):
        """ Plots loss on validation data after train and active learning """
        fig, ax = plt.subplots()
        for i, log in enumerate(logs):
            ax.scatter(list(range(len(log))), list(log.values()), label=labels[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        ax.legend()
        ax.grid(True)

        plt.show()

    def evalCache(self, cache, batch_size, use_extractor=False):
        """ Calls model and performs prediction and returns predictions """
        remainder_samples = len(cache) % batch_size  # Calculate number of remainder samples from batches
        predictions = []

        # Run batches
        for batch in trange(floor(len(cache) / batch_size)):
            batch_ids = cache[batch_size * batch:batch_size * (batch + 1)]
            X, y = self.dataClass.getBatch(batch_ids)

            # Try change y dtype
            y = y.astype('float32')

            if use_extractor == False:
                _, _, yh = self.modelManager.modelObject.eval(X, y)
                predictions.append(yh)
            elif use_extractor == True:
                yh = self.extractor.predict(X)
                predictions.append(yh)

        # Run remainders
        if remainder_samples > 0:
            batch_ids = cache[(-1) * remainder_samples:]
            X, y = self.dataClass.getBatch(batch_ids)

            # Try change y dtype
            y = y.astype('float32')


            if use_extractor == False:
                _, _, yh = self.modelManager.modelObject.eval(X, y)
                predictions.append(yh)
            elif use_extractor == True:
                yh = self.extractor.predict(X)
                predictions.append(yh)

        if use_extractor == True:
            return np.concatenate(predictions, axis=0)

        if use_extractor == False:
            return np.concatenate(predictions, axis=0)

    def evalAlgoClassClassifier(self, cache, unlabeled_pred, batch_size):
        remainder_samples = len(cache) % batch_size  # Calculate number of remainder samples from batches
        predictions = []

        # Run batches
        for batch in trange(floor(len(cache) / batch_size)):
            X = unlabeled_pred[batch_size * batch:batch_size * (batch + 1), :]
            yh = self.algoClass.inferBinaryClassifier(X)
            predictions.append(yh)

        # Run remainders
        if remainder_samples > 0:
            X = unlabeled_pred[(-1) * remainder_samples:, :]
            yh = self.algoClass.inferBinaryClassifier(X)
            predictions.append(yh)

        return np.concatenate(predictions, axis=0)

    def evalAlgoClassClassifierDALOC(self, cache, unlabeled_pred, batch_size):
        remainder_samples = len(cache) % batch_size  # Calculate number of remainder samples from batches
        predictions = []

        # Run batches
        for batch in trange(floor(len(cache) / batch_size)):
            X = unlabeled_pred[batch_size * batch:batch_size * (batch + 1), :]
            yh = self.algoClass.inferBinaryClassifier(X)
            predictions.append(yh)

        # Run remainders
        if remainder_samples > 0:
            X = unlabeled_pred[(-1) * remainder_samples:, :]
            yh = self.algoClass.inferBinaryClassifier(X)
            predictions.append(yh)

        bc_preds = np.concatenate(predictions, axis=0)

        predictions = []
        # Run batches
        for batch in trange(floor(len(cache) / batch_size)):
            X = unlabeled_pred[batch_size * batch:batch_size * (batch + 1), :]
            yh = self.algoClass.inferOC(X)
            predictions.append(yh)

        # Run remainders
        if remainder_samples > 0:
            X = unlabeled_pred[(-1) * remainder_samples:, :]
            yh = self.algoClass.inferOC(X)
            predictions.append(yh)

        oc_preds = np.concatenate(predictions, axis=0)
        oc_preds = np.sum(oc_preds, axis=1)
        oc_preds = oc_preds.reshape(oc_preds.shape[0], 1)

        preds = np.hstack((bc_preds, oc_preds))
        return preds

    def trainAlgoClassClasssifier(self, cache_df, batch_size):
        print("Training {} classifier".format(self.algoClass.algo_name))
        # 1. feed all samples through model and get second to last layer output
        self.extractor = tf.keras.Model(inputs=self.modelManager.modelObject.model.inputs,
                                        outputs=self.modelManager.modelObject.model.layers[-2].output, name="Extractor")
        print(self.extractor.summary())

        # 1a. feed through labeled cache and add [1,0] as target
        labeled_pred = self.evalCache(self.dataClass.train_cache, batch_size, use_extractor=True)
        if self.algoClass.single_output == True:
            if self.algoClass.algo_name == "AADA":
                labeled_targets = -1 * np.ones((labeled_pred.shape[0], 1))
                labeled_targets = labeled_targets.reshape(labeled_targets.shape[0], 1)

                # try changing type
                labeled_targets = labeled_targets.astype('float32')

        elif self.algoClass.single_output == False:
            labeled_targets = np.hstack((np.ones((labeled_pred.shape[0], 1)), np.zeros((labeled_pred.shape[0], 1))))

            # try changing type
            labeled_targets = labeled_targets.astype('float32')

        # 1b. feed through unlabeled cache (or subset) and add [0,1]
        unlabeled_pred = self.evalCache(cache_df, batch_size, use_extractor=True)
        if self.algoClass.single_output == True:
            if self.algoClass.algo_name == "AADA":
                unlabeled_targets = np.ones((unlabeled_pred.shape[0], 1))
                unlabeled_targets = unlabeled_targets.reshape(unlabeled_targets.shape[0], 1)

                # try changing type
                unlabeled_targets = unlabeled_targets.astype('float32')
        elif self.algoClass.single_output == False:
            unlabeled_targets = np.hstack(
                (np.zeros((unlabeled_pred.shape[0], 1)), np.ones((unlabeled_pred.shape[0], 1))))

            # try changing type
            unlabeled_targets = unlabeled_targets.astype('float32')

            # 2. Stack data and shuffle rows
        a = np.concatenate((labeled_pred, labeled_targets), axis=1)
        b = np.concatenate((unlabeled_pred, unlabeled_targets), axis=1)
        temp_dataset = np.concatenate((a, b), axis=0)

        # 3. train binary classifier
        if self.algoClass.algo_name == "OC":
            self.algoClass.trainBinaryClassifier(labeled_pred, labeled_pred.shape[0])
        elif self.algoClass.algo_name == "MemAE_Binary":
            # First train MemAE on labeled data
            self.algoClass.trainVAE(labeled_pred, batch_size)

            # Make MemAE augmented dataset
            _, labeled_MemAE_data, _ = self.algoClass.model(labeled_pred)
            _, unlabeled_MemAE_data, _ = self.algoClass.model(unlabeled_pred)
            a = np.concatenate((labeled_MemAE_data, labeled_targets), axis=1)
            b = np.concatenate((unlabeled_MemAE_data, unlabeled_targets), axis=1)
            MemAE_dataset = np.concatenate((a, b), axis=0)
            np.random.shuffle(MemAE_dataset)

            # Train binary classifier
            self.algoClass.trainBC(MemAE_dataset, batch_size)
        elif self.algoClass.algo_name == "VAE" or self.algoClass.algo_name == "MemAE AL":
            self.algoClass.trainVAE(labeled_pred, batch_size)
        elif self.algoClass.algo_name == "DALOC":
            # Train Orthonormal Certificats
            self.algoClass.trainOC(labeled_pred, batch_size)
            # Train binary classifier
            self.algoClass.trainBinaryClassifier(temp_dataset, batch_size)
        elif self.algoClass.algo_name == "AADA":
            self.algoClass.trainBinaryClassifier(temp_dataset, batch_size)
        else:
            self.algoClass.trainBinaryClassifier(temp_dataset, batch_size)

        # 4. feed unlabeled cache through binary classifier and save prediction ids
        if self.algoClass.algo_name == "DALOC":
            yh = self.evalAlgoClassClassifierDALOC(cache_df, unlabeled_pred, batch_size)
        elif self.algoClass.algo_name == "clusterDAL":
            # Cluster feature extracts
            clusters = self.algoClass.cluster(unlabeled_pred, cache_df)
            yh = self.evalAlgoClassClassifier(cache_df, unlabeled_pred, batch_size)
        elif self.algoClass.algo_name == "MemAE_Binary":
            yh = self.evalAlgoClassClassifier(cache_df, unlabeled_MemAE_data, batch_size)
        else:
            yh = self.evalAlgoClassClassifier(cache_df, unlabeled_pred, batch_size)

        # Get ids
        if self.algoClass.algo_name == "clusterDAL":
            ids = self.algoClass(cache_df, self.sample_size, pd.DataFrame(yh), clusters)
        else:
            ids = self.algoClass(cache_df, self.sample_size, pd.DataFrame(yh))

        # 5. return back ids of top samples
        return ids

    def clusterMarginWork(self,cache_ids,batch_size):
        # Sample Bernouli to determine whether to explore or exploit
        print('p value: ',self.algoClass.p)
        s = np.random.binomial(1, self.algoClass.p, 1)

        print("Bernoulli Sample: ", s)

        # If s is 0 -> explore, else exploit
        if s == 0:
            X, _ = self.dataClass.getBatch(cache_ids)
            closest_points = self.algoClass.cluster(X, cache_ids,s)
        else:
            # Get model predictions over unlabeled
            yh = self.evalCache(cache_ids, batch_size)
            yh = pd.concat([pd.DataFrame(cache_ids), pd.DataFrame(yh)], axis=1)

            # Get margin of all unlabeled cache predictions
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
            n_largest = yh.nlargest(self.algoClass.sub_sample, 'MC')
            batch = n_largest["ID"].to_list()

            # Cluster based off of margin values
            X, _ = self.dataClass.getBatch(batch)
            closest_points = self.algoClass.cluster(X, batch)

        return closest_points

    def runCycle(self, batch_size, cycles):
        """ Subroutine for run where model is trained on batch from algo, metrics logged, and caches updated """
        # ------------------------------------
        # 1. Get data and update caches
        # ------------------------------------

        # Get round unlabeled cache ids
        # print("Active Learning training round: {}".format(self.round))

        if self.dataClass.bins > 1:
            if self.round >= self.dataClass.bins:
                raise ValueError(
                    'Error: Engine has been called to run rounds more times than number of unlabeled caches!')
            cache_df = self.dataClass.unlabeled_cache[self.round]
        else:
            cache_df = copy.deepcopy(self.dataClass.unlabeled_cache)

        # Get subset of cache ids based off of active learning algo
        if self.algoClass.predict_to_sample == True:
            # cache_df
            yh = self.evalCache(cache_df, batch_size)
            yh = pd.concat([pd.DataFrame(cache_df[:len(yh)]), pd.DataFrame(yh)], axis=1)
            ids = self.algoClass(cache_df, self.sample_size, yh)
        elif self.algoClass.feature_set == True:
            ids = self.trainAlgoClassClasssifier(cache_df, batch_size)
        else:
            if self.algoClass.algo_name == "clusterMargin" or self.algoClass.algo_name == "clusterMargin2":
                ids = self.clusterMarginWork(cache_df,batch_size)
                self.algoClass(ids)
            else:
                ids = self.algoClass(cache_df, self.sample_size)

        # Manage new labels within caches
        self.dataClass.train_cache.extend(ids)  # Add new ids to train cache

        # Remove new ids from unlabeled caches: (two cases) one or multiple unlabeled caches
        if self.dataClass.bins > 1:
            for i in range(self.round, self.dataClass.bins):
                for id in ids:
                    if id in self.dataClass.unlabeled_cache[i]:
                        self.dataClass.unlabeled_cache[i].remove(id)
        elif self.dataClass.bins == 1:
            for id in ids:
                if id in self.dataClass.unlabeled_cache:
                    self.dataClass.unlabeled_cache.remove(id)

        # ------------------------------------
        # 2. Train and log
        # ------------------------------------

        total_loss = self.trainBatch(cycles, batch_size)
        total_loss = list(chain(*total_loss))
        avg_loss = sum(total_loss) / len(total_loss)
        print("Training loss: {}".format(self.round, avg_loss))
        self.train_metric_log["Round_" + str(self.round)] = avg_loss.numpy()

    def run(self, rounds, cycles, batch_size, val=True, val_track=None, plot=False):
        """
        Purpose: Calls runCycle through for loop to perform active learning training
        :param rounds: int that determines number of active learning training rounds
        :return: None
        """

        self.val_track = val_track

        print("\n")
        print("-" * 20)
        print("Starting active learning")
        print("-" * 20)

        for n in range(rounds):
            start_time = time.time()
            print("\n")
            print("Round: {}".format(n))
            self.runCycle(batch_size, cycles)

            if val == True:
                total_loss, total_scores = self.valTest(batch_size)
                total_loss = list(chain(*total_loss))
                val_avg_loss = sum(total_loss) / len(total_loss)
                print("{}. Val loss: {}".format(self.round, val_avg_loss))
                self.val_metric_log["Round_" + str(self.round)] = val_avg_loss.numpy()

                keys = list(total_scores[0].keys())
                val_metrics = []
                for i, k in enumerate(keys):
                    samples = []
                    for j, sample in enumerate(total_scores):
                        samples.append((sample[k]))
                    samples = np.mean(samples)
                    val_metrics.append(samples)
                    if ((self.algoClass.algo_name == "clusterMargin") and (i == 0)):
                        self.algoClass.p = self.algoClass.adjustP(samples)

                    print("{}: {}".format(self.modelManager.modelObject.metrics[i].name, samples))

            time_round = time.time() - start_time
            self.updateLog(round="round_" + str(n), time_round=time_round, batch_size=batch_size,
                           val_loss=val_avg_loss.numpy(), val_metrics=val_metrics)

            self.round += 1

        # Load best weights
        self.modelManager.loadBestValWeights("tmp/val_best_weights.h5")

        # Log performance best weights
        total_loss, total_scores = self.valTest(batch_size)
        total_loss = list(chain(*total_loss))
        val_avg_loss = sum(total_loss) / len(total_loss)
        keys = list(total_scores[0].keys())
        val_metrics = []
        for i, k in enumerate(keys):
            samples = []
            for j, sample in enumerate(total_scores):
                samples.append((sample[k]))
            samples = np.mean(samples)
            val_metrics.append(samples)
            print("{}: {}".format(self.modelManager.modelObject.metrics[i].name, samples))

        time_round = time.time() - start_time
        self.updateLog(round="val_best", time_round=time_round, batch_size=batch_size,
                       train_loss=None, val_loss=val_avg_loss.numpy(), val_metrics=val_metrics)

        # Logic to plot loss from active learning round
        if plot == True:
            xlabel = "Round"
            ylabel = "Error"
            title = "Round vs Error on Active Learning"
            if val == True:
                labels = ["Train", "Val"]
                self.plotLog([self.train_metric_log, self.val_metric_log], xlabel, ylabel, title, labels)
            else:
                self.plotLog([self.train_metric_log], xlabel, ylabel, title, "Train")
            input('press return to continue' + "\n")

        print("\n" + "Active Learning done")

    def initialTrain(self, epochs, batch_size, val=True,val_track=None,plot=False):
        self.val_track = val_track

        """ Trains model on data present in train cache """
        print("\n")
        print("-" * 20)
        print("Training model on training cache")
        print("-" * 20)
        for epoch in range(epochs):
            start_time = time.time()

            total_loss = self.trainBatch(1, batch_size)
            total_loss = list(chain(*total_loss))
            train_avg_loss = sum(total_loss) / len(total_loss)
            print("Epoch {}. training average loss: {}".format(epoch, train_avg_loss))
            self.intialTrain_metric_log["Epoch_" + str(epoch)] = train_avg_loss.numpy()

            if val == True:
                total_loss, total_scores = self.valTest(batch_size)

                total_loss = list(chain(*total_loss))
                val_avg_loss = sum(total_loss) / len(total_loss)
                print("{}. Val loss: {}".format(self.round, val_avg_loss))
                self.intialVal_metric_log["Epoch_" + str(epoch)] = val_avg_loss.numpy()

                keys = list(total_scores[0].keys())
                val_metrics = []
                for i, k in enumerate(keys):
                    samples = []
                    for j, sample in enumerate(total_scores):
                        samples.append((sample[k]))
                    samples = np.mean(samples)
                    val_metrics.append(samples)
                    print("{}: {}".format(self.modelManager.modelObject.metrics[i].name, samples))

            time_round = time.time() - start_time
            self.updateLog(round="train_" + str(epoch), time_round=time_round, batch_size=batch_size,
                           train_loss=train_avg_loss.numpy(), val_loss=val_avg_loss.numpy(), val_metrics=val_metrics)

        print('Finished initial training of model.')

        # Load best weights
        self.modelManager.loadBestValWeights("tmp/val_best_weights.h5")

        # Log performance best weights
        total_loss, total_scores = self.valTest(batch_size)
        total_loss = list(chain(*total_loss))
        val_avg_loss = sum(total_loss) / len(total_loss)
        keys = list(total_scores[0].keys())
        val_metrics = []
        for i, k in enumerate(keys):
            samples = []
            for j, sample in enumerate(total_scores):
                samples.append((sample[k]))
            samples = np.mean(samples)
            val_metrics.append(samples)
            print("{}: {}".format(self.modelManager.modelObject.metrics[i].name, samples))

        time_round = time.time() - start_time
        self.updateLog(round="train_best", time_round=time_round, batch_size=batch_size,
                       train_loss=None, val_loss=val_avg_loss.numpy(), val_metrics=val_metrics)

        # Logic to plot loss from initial training
        if plot == True:
            xlabel = "Epoch"
            ylabel = "Error"
            title = "Epoch vs Error on Train"
            if val == True:
                labels = ["Train", "Val"]
                self.plotLog([self.intialTrain_metric_log, self.intialVal_metric_log], xlabel, ylabel, title, labels)
            else:
                self.plotLog([self.intialTrain_metric_log], xlabel, ylabel, title, "Train")
            input('press return to continue')

    def saveModel(self, model_name=None):
        if model_name == None:
            self.modelManager.modelObject.model.save_weights('model')
        else:
            self.modelManager.modelObject.model.save_weights(model_name+".h5")

    def loadModel(self, model_name):
        self.modelManager.modelObject.model.load_weights('Models/'+model_name+".h5")
        self.modelManager.modelObject.model.compile()
