import tensorflow as tf
import numpy as np
import os
import re
from data_preprocessing.utils import convert_fen_to_matrix


class FENDataset:
    def __init__(self, dataset_dir, shuffle_data, batch_size):
        self.__dataset_dir = dataset_dir
        self.__shuffle_data = shuffle_data
        self.__batch_size = batch_size

    def create_datasets(self, train_ratio):
        list_games_folders_with_paths = [f.path for f in os.scandir(self.__dataset_dir) if f.is_dir()]
        list_of_fen_pairs = []
        for game_folder in list_games_folders_with_paths:
            game_files = os.listdir(game_folder)
            actual_positions = [file for file in game_files if re.search(f"pos\\d+.txt$", file) is not None]
            for i in range(len(actual_positions) - 1):
                before = os.path.join(game_folder, f"pos{i}.txt")
                after = os.path.join(game_folder, f"pos{i + 1}.txt")
                list_of_fen_pairs.append((before, after))
                legal_moves = [file for file in game_files if re.search(f"pos{i}_legal\\d+.txt$", file) is not None]
                for legal_move in legal_moves:
                    list_of_fen_pairs.append((before, os.path.join(game_folder, legal_move)))

        list_of_fen_pairs_ds = tf.data.Dataset.from_tensor_slices(list_of_fen_pairs)
        ds_size = len(list_of_fen_pairs_ds)

        ds_train = list_of_fen_pairs_ds.take(int(ds_size * train_ratio))
        ds_test = list_of_fen_pairs_ds.skip(int(ds_size * train_ratio))

        ds_train = ds_train.shuffle(int(ds_size * train_ratio)) if self.__shuffle_data else ds_train
        ds_train = ds_train.map(self._combine_positions_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.batch(self.__batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.shuffle(int(ds_size * (1 - train_ratio))) if self.__shuffle_data else ds_test
        ds_test = ds_test.map(self._combine_positions_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(self.__batch_size)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_test

    def _tf_helper_func(self, files_tuple):
        file_before_name = str(files_tuple[0].numpy())[2:-1]
        file_after_name = str(files_tuple[1].numpy())[2:-1]
        with open(file_before_name, "r") as file_before:
            fen_before = file_before.readline()
        with open(file_after_name, "r") as file_after:
            fen_after = file_after.readline()
        encoded_position_before = convert_fen_to_matrix(fen_before)
        encoded_position_after = convert_fen_to_matrix(fen_after)
        label = 0.0 if "_legal" in file_after_name else 1.0
        return encoded_position_before, encoded_position_after, label

    def _combine_positions_labels(self, x):
        a, b, y = tf.py_function(self._tf_helper_func, [x],
                                 Tout=[tf.float32, tf.float32, tf.float32])
        return (a, b), y
