import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import random
import os
import chess
import chess.pgn
from utils import DATA_REAL_PATH
from data_preprocessing.utils import get_pgn_filepaths, convert_fen_to_matrix, check_elo
from models.squeezenet import squeezenet_chess
import argparse


def parse_file_helper(file):
    file_name = str(file.numpy())[2:-1]
    datapoints = []
    with open(file_name) as input_file:
        game = chess.pgn.read_game(input_file)
        fen_before = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = game.board()
        for actual_index, actual_move in enumerate(game.mainline_moves()):
            for legal_index, legal_move in enumerate(board.legal_moves):
                if legal_move == actual_move:
                    continue
                board.push(legal_move)  # make move
                datapoints.append((fen_before, board.fen(), 0))
                board.pop()  # undo move
            board.push(actual_move)  # make actual move
            datapoints.append((fen_before, board.fen(), 1))
            fen_before = board.fen()
    random.shuffle(datapoints)
    datapoints = np.array(datapoints).T
    encoded_position_before = [convert_fen_to_matrix(fen) for fen in datapoints[0]]
    encoded_position_after = [convert_fen_to_matrix(fen) for fen in datapoints[1]]
    label = datapoints[2]
    return [encoded_position_before, encoded_position_after, label]


def parse_file(x):
    a, b, c = tf.py_function(parse_file_helper, [x], Tout=[tf.float32, tf.float32, tf.float32])
    dataset = tf.data.Dataset.from_tensor_slices(tuple([a, b, c]))
    dataset = dataset.map(lambda in1, in2, out: ({'chessboard_before': in1, 'chessboard_after': in2},
                                                 {'target': out}))
    return dataset


def weighted_binary_crossentropy(pos_weight=1, neg_weight=1):
    def loss(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        logloss = -(y_true * K.log(y_pred) * pos_weight + (1 - y_true) * K.log(1 - y_pred) * neg_weight)
        return K.mean(logloss, axis=-1)

    return loss


def main():
    # Check if the GPU  is enabled
    print(tf.config.list_physical_devices("GPU"))
    # For reproducibility
    np.random.seed(42)
    random.seed(42)
    # For readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # Dataset and training configuration
    dataset_dir = os.path.join(DATA_REAL_PATH, 'datasets', "1001_1200")
    shuffle_data = True
    batch_size = 64
    input_shape = (8, 8, 18)
    train_ratio = 0.8
    epochs = 50

    # Create train and test sets
    list_of_filenames = get_pgn_filepaths(dataset_dir, check_elo, min_elo=1001, max_elo=1800)
    random.shuffle(list_of_filenames)
    dataset = tf.data.Dataset.from_tensor_slices(list_of_filenames)
    number_of_files = len(list_of_filenames)

    ds_train = dataset.take(int(number_of_files * train_ratio))
    ds_val = dataset.skip(int(number_of_files * train_ratio))

    ds_train = ds_train.shuffle(number_of_files)
    # ds_train = ds_train.flat_map(parse_csv)
    ds_train = ds_train.interleave(map_func=parse_file, cycle_length=4, block_length=16)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.shuffle(number_of_files)
    # ds_test = ds_test.flat_map(parse_csv)
    ds_val = ds_val.interleave(map_func=parse_file, cycle_length=4, block_length=16)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Iterate through the datasets once to get the total number of data points
    # Count the number of labels for each class to use it for balancing the loss during training
    train_samples = 0
    labels_counts = {}
    for item in ds_train.take(10000):
        targets = item[1].get('target')
        train_samples += int(tf.shape(targets)[0])
        bin_count_obj = tf.sparse.bincount(tf.cast(targets, tf.int64))
        for i in range(len(bin_count_obj.indices)):
            idx = int(bin_count_obj.indices[i])
            labels_counts[idx] = labels_counts.get(idx, 0) + int(bin_count_obj.values[i])

    test_samples = 0
    for item in ds_val:
        targets = item[1].get('target')
        test_samples += int(tf.shape(targets)[0])

    print(f"Number of train samples: {train_samples}")
    print(f"Number of test samples: {test_samples}")
    print(f"Labels counts: {labels_counts}")

    # Deal with imbalanced dataset by setting the weights to balance the loss
    n_classes = len(labels_counts)
    class_weight = {label: train_samples / (n_classes * count)
                    for (label, count) in labels_counts.items()}
    print("Weights for balancing loss during training: ", class_weight)

    # Model paths
    model_path = os.path.join(DATA_REAL_PATH, 'newest_model')
    checkpoint_path = os.path.join(model_path, 'checkpoint_dir', 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Model creation
    model = squeezenet_chess(image_shape=input_shape)

    # Load weights
    # model.load_weights(os.path.join(model_path, 'model_tf_format', 'model'))
    # model.load_weights(checkpoint_path)

    # Some useful callbacks

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', verbose=1, save_weights_only=True,
        save_freq='epoch', mode='auto', save_best_only=True)

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)

    callbacks_list = [early_stopping, checkpoint_callback, reduce_lr_on_plateau]

    # Compiling the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = weighted_binary_crossentropy(pos_weight=class_weight.get(1),
                                        neg_weight=class_weight.get(0))
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  run_eagerly=True)
    model.summary()

    model.fit(
        x=ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=callbacks_list
    )

    # Saving weights after training (tf format)
    model.save_weights(os.path.join(model_path, 'model_tf_format', 'model'), save_format='tf')

    # Saving weights after training (h5 format)
    model.save_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'), save_format='h5')


if __name__ == '__main__':
    main()
