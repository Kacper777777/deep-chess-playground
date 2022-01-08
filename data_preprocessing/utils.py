import csv
import re

import numpy as np
import os
import chess
import chess.pgn


def separate_pgns_into_different_files(pgn_file_path, destination_dir):
    """Get all the pgns from the single file and write each .pgn into separate file.
    All of the generated files will be created in the destination dir."""
    with open(pgn_file_path) as input_file:
        counter = 0
        while True:
            game = chess.pgn.read_game(input_file)
            if game is None:
                break
            with open(os.path.join(destination_dir, f"game{counter}.pgn"), 'w') as out:
                print(game, file=out)
            counter += 1


def get_pgn_filepaths(pgn_dir, cond_func, **kwargs):
    """Get all of the file paths to the pgn files that satisfy the condition."""
    filepaths = []
    for pgn_entry in os.scandir(pgn_dir):
        pgn_file_path = pgn_entry.path
        with open(pgn_file_path) as input_file:
            game = chess.pgn.read_game(input_file)
            if cond_func(game, **kwargs):
                filepaths.append(pgn_file_path)
    return filepaths


def convert_fen_to_matrix(fen):
    fen_elements = fen.split(' ')
    flatten_board = str(chess.Board(fen))
    flatten_board = flatten_board.replace(' ', '').replace('\n', '')
    on_move = 1 if fen_elements[1] == 'w' else 0
    white_kingside_castle = 1 if "K" in fen_elements[2] else 0
    white_queenside_castle = 1 if "Q" in fen_elements[2] else 0
    black_kingside_castle = 1 if "k" in fen_elements[2] else 0
    black_queenside_castle = 1 if "q" in fen_elements[2] else 0
    en_passant_index = (8 - int(fen_elements[3][1]), ord(fen_elements[3][0]) - ord('a')) if fen_elements[3] != '-' else None
    half_moves = int(fen_elements[4])
    full_moves = int(fen_elements[5])

    encoded_position = np.zeros((8, 8, 18), dtype=np.float32)

    piece_channel_dict = {
        'P': 0,
        'N': 1,
        'B': 2,
        'R': 3,
        'Q': 4,
        'K': 5,
        'p': 6,
        'n': 7,
        'b': 8,
        'r': 9,
        'q': 10,
        'k': 11}

    for i in range(len(flatten_board)):
        if flatten_board[i] != '.':
            encoded_position[int(i / 8), int(i % 8), piece_channel_dict[flatten_board[i]]] = 1

    encoded_position[:, :, 12] = on_move
    encoded_position[:, :, 13] = white_kingside_castle
    encoded_position[:, :, 14] = white_queenside_castle
    encoded_position[:, :, 15] = black_kingside_castle
    encoded_position[:, :, 16] = black_queenside_castle
    if en_passant_index is not None:
        encoded_position[en_passant_index[0], en_passant_index[1], 17] = 1

    return encoded_position


def check_elo(game, min_elo, max_elo):
    # TODO In the future use something more efficient than regex here
    if (re.search(f"^\\d{3,4}$", str(game.headers['WhiteElo'])) is None) or \
            (re.search(f"^\\d{3,4}$", str(game.headers['BlackElo'])) is None):
        return False
    if (min_elo <= int(game.headers['WhiteElo']) <= max_elo) and (min_elo <= int(game.headers['BlackElo']) <= max_elo):
        return True
    else:
        return False
