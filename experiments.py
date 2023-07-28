import numpy as np
import tensorflow as tf
import transformers
import tensorflow_probability as tfp
from transformers import AutoTokenizer, TFAutoModel, AutoConfig, TFAutoModelForSequenceClassification
import sklearn
from sklearn.calibration import calibration_curve
import subprocess
import os
from threading import Thread , Timer
import sched, time
import pickle
import re
from pathlib import Path
from model_classes import MINION, READ, READ_GRU

"""
    Helper functions
"""

""" Defining naming conventions for experiments. """
def get_experiment_name(model_type, hidden_dimension, learning_rate, epochs, batchsize):
    return model_type + f"_hdim{hidden_dimension}_lr{learning_rate}_ep{epochs}_bs{batchsize}"

def get_gpu_hist_name(experiment_name):
    return "gpu_hist_" + experiment_name + ".pkl"

def get_training_hist_name(experiment_name):
    return "training_hist_" + experiment_name + ".pkl"


def track_gpu_mem(experiment_obj, interval=5.0):
    """
        This function calls itself every 5 secs and print the gpu_memory.
    """
    thread_gpu_tracker = Timer(interval, track_gpu_mem, [experiment_obj, interval])
    thread_gpu_tracker.start()

    out = subprocess.check_output("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv", shell=True).decode()
    float_pattern = r'\d+\.\d+|\d+'
    numbers = re.findall(float_pattern, out)

    experiment_obj.gpu_hist["counter"] += 1
    experiment_obj.gpu_hist["mib"].append(float(numbers[0]))
    experiment_obj.gpu_hist["percent"].append(float(numbers[1]))
    experiment_obj.gpu_hist["interval"] = interval

    return thread_gpu_tracker


"""
    Experiment class
"""
class Experiment:
    def __init__(self,
                 model_type,
                 hidden_dimension,
                 batchsize=256,
                 learning_rate=1e-3,
                 epochs=5,
                 tpu_strategy=None,
                 base_model_name="vinai/bertweet-base") -> None:
    
        self.model_type = model_type
        self.hidden_dimension = hidden_dimension
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tpu_strategy = tpu_strategy
        self.base_model_name = base_model_name

        self.experiment_name = get_experiment_name(model_type, hidden_dimension, learning_rate, epochs, batchsize)
        self.gpu_hist = None
    
    def compile(self):
        def initialize(model_type):
            if model_type == 'base':
                return TFAutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                            config=AutoConfig.from_pretrained(self.base_model_name))
            elif model_type == 'read':
                return READ(self.base_model_name, 2, hidden_dimension=self.hidden_dimension)
            elif model_type == 'read_gru':
                return READ_GRU(self.base_model_name, 2, hidden_dimension=self.hidden_dimension)
            elif model_type == 'minion':
                return MINION(self.base_model_name, 2, hidden_dimension=self.hidden_dimension)
            else:
                raise ValueError("{model_type} is not a valid model type. Use 'base', 'read', 'read_gru' or 'minion'.")

        if self.tpu_strategy is None:
            self.model = initialize(self.model_type)
            self.model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(self.model_type == 'base')),
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=False, clipnorm=None),
                metrics=[tf.metrics.SparseCategoricalAccuracy()]
            )
        else:
            with self.tpu_strategy.scope():
                self.model = initialize(self.model_type)
                self.model.compile(
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(self.model_type == 'base')),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=False, clipnorm=None),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()]
                )

    def train(self, train_ds, val_ds=None):
        """
            Trains self.model while tracking the memory and energy usage of the gpu.
            The model and all histories are saved.
        """
        self.gpu_hist = { "mib" : [], "percent" : [], "counter" : 0 }

        gpu_mem_proc = track_gpu_mem(self, 10.0) # start gpu tracking
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=self.epochs, verbose=1)
        gpu_mem_proc.join() # stop gpu tracking

        # store model and histories
        history_name = get_training_hist_name(self.experiment_name)
        gpu_hist_name = get_gpu_hist_name(self.experiment_name)
        self.model.save(self.experiment_name)
        with open(gpu_hist_name, 'wb') as f:
            pickle.dump(self.gpu_hist, f)
        with open(history_name, 'wb') as f:
            pickle.dump(self.history, f)
        subprocess.run("gsutil cp -r {self.experiment_name + \"/\"} \"gs://cil_2023/models/\"", shell=True)
        subprocess.run("gsutil cp {history_name} \"gs://cil_2023/models/\"", shell=True)
        subprocess.run("gsutil cp {gpu_hist_name} \"gs://cil_2023/models/\"", shell=True)

    def load_model(self):
        if not Path(self.experiment_name).exists():
            subprocess.run("gsutil cp -r {\"gs://cil_2023/models/\" + self.experiment_name} .", shell=True)
        self.model = tf.keras.models.load_model(self.experiment_name)

    def load_gpu_history(self):
        gpu_file_name = get_gpu_hist_name(self.experiment_name)
        
        if not Path(gpu_file_name).exists():
            subprocess.run("gsutil cp {\"gs://cil_2023/models/\" + gpu_file_name} .", shell=True)
        
        with open(gpu_file_name, 'rb') as file:
            unpickled_object = pickle.load(file)

        # remove trailing zeros (due to some bugs, this may occur, if gpu measurement does not automatically stop)
        unpickled_object["percent"] = np.trim_zeros(np.array(unpickled_object["percent"]), trim='b').tolist()

        return unpickled_object
    
    def predict(self, test_ds):
        probs = self.model.predict(test_ds)

        if self.model_type == 'base':
            probs = tf.nn.softmax(probs.logits).numpy() # base model outputs logits
            
        predictions = np.argmax(probs, axis=1)
        labels = np.concatenate([y for x, y in test_ds], axis=0)
        
        return predictions, probs, labels

    def get_scoring(self, predictions, probs, labels, nbins=20):
        """
            Returns dict containing accuracy, f1-score, ace, mce, calibration curve
        """
        scoring = {}
        scoring["accuracy"] = sklearn.metrics.accuracy_score(labels, predictions)
        scoring["f1"] = sklearn.metrics.f1_score(labels, predictions, average='macro')
        
        # compute adaptive calibration error (ace)
        prob_true_quantile, prob_pred_quantile = calibration_curve(y_true=labels, y_prob=probs[:, 1],
                                                                   n_bins=nbins, strategy='quantile')
        scoring["ace"] = np.mean(np.abs(prob_true_quantile - prob_pred_quantile))

        # compute maximum calibration error (mce)
        prob_true, prob_pred = calibration_curve(y_true=labels, y_prob=probs[:, 1],
                                                 n_bins=nbins, strategy='uniform')
        scoring["mce"] = np.max(np.abs(prob_true - prob_pred))

        # calibration curve
        scoring["cal_curve_x"] = prob_pred
        scoring["cal_curve_y"] = prob_true

        return scoring
    
    def evaluate(self, test_ds, nbins=20):
        predictions, probs, labels = self.predict(test_ds)
        return self.get_scoring(predictions, probs, labels, nbins)

    def get_duration(self):
        if self.gpu_hist is None:
            self.gpu_hist = self.load_gpu_history()
        percent_values = self.gpu_hist["percent"]
        interval = self.gpu_hist["interval"]
        total_time_seconds = len(percent_values) * interval

        # Convert total_time_seconds to hours and minutes
        total_time_minutes, seconds = divmod(total_time_seconds, 60)
        hours, minutes = divmod(total_time_minutes, 60)

        # Calculate average time per epoch
        if self.epochs > 0:
            average_time_per_epoch_seconds = total_time_seconds / self.epochs
            avg_time_minutes, avg_time_seconds = divmod(average_time_per_epoch_seconds, 60)
            avg_time_hours, avg_time_minutes = divmod(avg_time_minutes, 60)

        return total_time_seconds, hours, minutes, average_time_per_epoch_seconds

    def compute_energy_consumption(self, p0 = 0.4):
        # p0 = 0.4 is the value for gpu A100; see here for SXM: https://www.nvidia.com/en-us/data-center/a100/
        if self.gpu_hist is None:
            self.gpu_hist = self.load_gpu_history()
        utilization_percents = self.gpu_hist["percent"]
        interval = self.gpu_hist["interval"]

        # Calculate the number of entries per minute
        entries_per_minute = (int)(60 // interval)

        # Check if the utilization_percents length is a multiple of entries_per_minute
        # If not, it means the last minute is incomplete, and we should only consider complete minutes for the calculation
        if len(utilization_percents) % entries_per_minute != 0:
            drop_count = (int)(len(utilization_percents) % entries_per_minute)
            utilization_percents = utilization_percents[:-drop_count]

        # Reshape the utilization_percents to a 2D array with 'entries_per_minute' columns
        utilization_percents = np.array(utilization_percents).reshape(-1, entries_per_minute)

        # Create utilization_percents_per_minutes list by taking the average of each row
        utilization_percents_per_minutes = utilization_percents.mean(axis=1)


        # Compute the total energy consumption
        total_energy_consumption = np.sum(utilization_percents_per_minutes) * p0 / 6000

        # result is in kWH
        return total_energy_consumption

