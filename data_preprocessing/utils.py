import csv
import re
import numpy as np
import os
import chess
import chess.pgn


def write_one_game_in_one_line(input_file_path, output_file_path):
    """Get all the pgns from the single file and write each .pgn into new file in one line."""
    with open(input_file_path, 'r') as input_file:
        with open(output_file_path, 'w') as output_file:
            while True:
                game = chess.pgn.read_game(input_file)
                if game is None:
                    break
                for key in game.headers:
                    print(f'[{key} "{game.headers[key]}"]', file=output_file, sep='', end=' | ')
                print('\n', file=output_file, end='')


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
    if (re.search("^\\d{3,4}$", str(game.headers['WhiteElo'])) is None) or \
            (re.search("^\\d{3,4}$", str(game.headers['BlackElo'])) is None):
        return False
    if (min_elo <= int(game.headers['WhiteElo']) <= max_elo) and (min_elo <= int(game.headers['BlackElo']) <= max_elo):
        return True
    else:
        return False
