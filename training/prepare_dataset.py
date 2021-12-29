import tensorflow as tf
import numpy as np
import random


class FENDataset:
    def __init__(self, image_names, shuffle, batch_size):
        self.__image_names = image_names
        self.__shuffle = shuffle
        self.__batch_size = batch_size
        self.__ds_size = len(self.__image_names)

    def create_datasets(self, train_ratio):
        filelist_ds = tf.data.Dataset.from_tensor_slices(self.__image_names)

        ds_train = filelist_ds.take(int(self.__ds_size * train_ratio))
        ds_test = filelist_ds.skip(int(self.__ds_size * train_ratio))

        ds_train = ds_train.map(self._combine_positions_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.shuffle(int(self.__ds_size * train_ratio))
        ds_train = ds_train.batch(self.__batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.map(self._combine_positions_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(self.__batch_size)

        return ds_train, ds_test

    def _combine_positions_labels(self, file_path: tf.Tensor):
        x = self._generate_X(file_path)
        y = self._generate_y(file_path)
        return x, y

    def _generate_X(self, file):

        return np.array(encoded_positions)

    def _generate_y(self, file):

        return np.array(labels)
