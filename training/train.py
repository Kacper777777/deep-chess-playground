import io
import re
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import csv
import random
import os
import chess
import chess.pgn
from utils import DATA_REAL_PATH
from data_preprocessing.utils import convert_fen_to_matrix, encode_move_8x8x73
from models.squeezenet import chess_move_classifier, chess_position_evaluator, \
    chess_move_classifier_and_position_evaluator
import argparse

# TODO Use argparse Python utilities in order to pass arguments using command line or config file


min_elo, max_elo = 1001, 1400


def get_position_move_and_result_helper(file):
    file_name = str(file.numpy())[2:-1]
    df = pd.read_csv(file_name, sep='\t')
    df = df.sample(frac=1).reset_index(drop=True)
    df['WhiteElo'] = pd.to_numeric(df['WhiteElo'], errors='coerce')
    df['BlackElo'] = pd.to_numeric(df['BlackElo'], errors='coerce')
    df = df[df['WhiteElo'].notnull()]
    df = df[df['BlackElo'].notnull()]
    df['WhiteElo'] = df['WhiteElo'].between(1001, 1400, inclusive=True)
    df['BlackElo'] = df['BlackElo'].between(1001, 1400, inclusive=True)
    df_dict = df.to_dict('records')
    datapoints = []
    for row in df_dict:
        pgn = row["PGN"]
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        for actual_index, actual_move in enumerate(game.mainline_moves()):
            fen_before = board.fen()
            board.push(actual_move)
            datapoints.append(f"""{str(fen_before)};{str(actual_move)};{row["Result"]}""")
    random.shuffle(datapoints)
    return datapoints


def get_position_move_and_result(file):
    a = tf.py_function(get_position_move_and_result_helper, [file], Tout=[tf.string])
    dataset = tf.data.Dataset.from_tensor_slices(a)
    return dataset


def create_matrices_helper(fen_move_result):
    fen_move_result = str(fen_move_result.numpy())[2:-1]
    elements = fen_move_result.split(";")
    fen, move, str_result = elements[0], elements[1], elements[2]
    res_white = float(str_result[0])
    res_black = float(str_result[2])
    draw = float((res_white + res_black) == 0)
    if res_white:
        result = 0
    elif draw:
        result = 1
    else:
        result = 2
    encoded_position = convert_fen_to_matrix(fen)
    # move_probabilities = encode_move_8x8x73(move)
    evaluation = tf.one_hot(indices=result, depth=3)
    return encoded_position, evaluation


def create_matrices(x):
    a, b = tf.py_function(create_matrices_helper, [x], Tout=[tf.float32, tf.float32])
    return ({"chessboard": a}, {"evaluation": b})


def main():
    # Check if the GPU  is enabled
    print(tf.config.list_physical_devices("GPU"))
    # For reproducibility
    np.random.seed(42)
    random.seed(42)
    # For readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # Dataset and training configuration
    dataset_dir = os.path.join(DATA_REAL_PATH, 'datasets', "1001-1400")
    shuffle_data = True
    batch_size = 512
    input_shape = (8, 8, 18)
    train_ratio = 0.8
    epochs = 50

    # Create train and test sets
    list_of_filenames = [f.path for f in os.scandir(dataset_dir)]
    random.shuffle(list_of_filenames)
    dataset = tf.data.Dataset.from_tensor_slices(list_of_filenames)
    number_of_files = len(list_of_filenames)

    ds_train = dataset.take(int(number_of_files * train_ratio))
    ds_val = dataset.skip(int(number_of_files * train_ratio))

    ds_train = ds_train.shuffle(number_of_files)
    ds_train = ds_train.interleave(map_func=get_position_move_and_result,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(1000000)
    ds_train = ds_train.map(map_func=create_matrices,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.shuffle(number_of_files)
    ds_val = ds_val.interleave(map_func=get_position_move_and_result,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(1000000)
    ds_val = ds_val.map(map_func=create_matrices,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Model paths
    model_path = os.path.join(DATA_REAL_PATH, 'newest_model')
    checkpoint_path = os.path.join(model_path, 'checkpoint_dir', 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Model creation
    model = chess_position_evaluator(input_shape=input_shape)

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
    loss = 'categorical_crossentropy'
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


if __name__ == '__main__':
    main()
