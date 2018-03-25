# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '25.03.18' '20:34'

# imports
import os
import pandas as pd
import numpy as np
import cPickle as pickle
from constants import *

def create_onehot_label(x):
    label = np.zeros((1, NUM_LABELS), dtype=np.float32)
    label[:, int(x)] = 1
    return label

def read_data(data_dir):
    # lets open pickle file if exists
    pickle_file = os.path.join(data_dir, "EmoData.pickle")
    if not os.path.exists(pickle_file):
        # EmoData.pickle doesn't exist
        print "Reading train.csv ..."
        train_filename = os.path.join(data_dir, "train.csv")
        data_frame = pd.read_csv(train_filename)

        # formatting the data
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()

        # reshapping data
        train_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        train_labels = np.array([map(create_onehot_label, data_frame['Emotion'].values)]).reshape(-1, NUM_LABELS)

        # get 10% data as validation data
        permutations = np.random.permutation(train_images.shape[0])
        train_images = train_images[permutations]
        train_labels = train_labels[permutations]
        validation_percent = int(train_images.shape[0] * VALIDATION_PERCENT)
        validation_images = train_images[:validation_percent]
        validation_labels = train_labels[:validation_percent]
        train_images = train_images[validation_percent:]
        train_labels = train_labels[validation_percent:]

        print "Reading test.csv ..."
        test_filename = os.path.join(data_dir, "test.csv")
        data_frame = pd.read_csv(test_filename)

        # formatting data
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()

        # reshapping data
        test_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        # lets write EmoData.pickle file
        with open(pickle_file, "wb") as file:
            try:
                print 'Picking ...'
                save = {
                    "train_images": train_images,
                    "train_labels": train_labels,
                    "validation_images": validation_images,
                    "validation_labels": validation_labels,
                    "test_images": test_images,
                }
                pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)

            except:
                print("Unable to pickle file :/")

    with open(pickle_file, "rb") as file:
        save = pickle.load(file)
        train_images = save["train_images"]
        train_labels = save["train_labels"]
        validation_images = save["validation_images"]
        validation_labels = save["validation_labels"]
        test_images = save["test_images"]

    return train_images, train_labels, validation_images, validation_labels, test_images

def get_next_batch(images, labels, step):
    offset = (step * BATCH_SIZE) % (images.shape[0] - BATCH_SIZE)
    batch_images = images[offset: offset + BATCH_SIZE]
    batch_labels = labels[offset:offset + BATCH_SIZE]
    return batch_images, batch_labels