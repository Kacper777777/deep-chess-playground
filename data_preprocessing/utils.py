import csv
import re
from threading import Thread
import numpy as np
import os
import chess
import chess.pgn


def make_csv_from_pgn(source_dir, destination_dir, num_games_per_file):
    """It takes all the .pgn files from the source_dir and it creates .csv files
    in the destination_dir. Each .csv file will have at most num_games_per_file records."""
    def save_games_to_csv(headers, games, filepath):
        if len(games) == 0:
            return
        print(f"Saving games to a file")
        with open(filepath, 'w', newline='') as output_file:
            tsv_writer = csv.writer(output_file, delimiter='\t')
            tsv_writer.writerow(headers)
            for item in games:
                tsv_writer.writerow(item)
        print("Games saved")

    list_of_dir_entries = [f for f in os.scandir(source_dir)]
    destination_file_counter = 0
    games = []
    headers = ['Event', 'Site', 'Date', 'Round', 'White', 'Black',
               'Result', 'BlackElo', 'BlackRatingDiff', 'ECO', 'Opening', 'Termination',
               'TimeControl', 'UTCDate', 'UTCTime', 'WhiteElo', 'WhiteRatingDiff', 'PGN']

    for entry in list_of_dir_entries:
        with open(entry.path, 'r') as input_file:
            print(f"Opened {entry.path} file")
            ctr = 0
            while True:
                print(f"Game nr {ctr}")
                ctr += 1
                game = chess.pgn.read_game(input_file)
                if game is None:
                    break
                game_info = [game.headers.get(key, '?') for key in headers[:-1]]
                mainline_moves = [str(move) for move in game.mainline_moves()]
                pgn = " ".join(mainline_moves)
                game_info.append(pgn)
                games.append(game_info)

                if len(games) == num_games_per_file:
                    Thread(target=save_games_to_csv, args=(headers,
                                                           list(games[:num_games_per_file]),
                                                           os.path.join(destination_dir,
                                                                        f'{destination_file_counter}.csv'))).start()
                    games = []
                    destination_file_counter += 1
        Thread(target=save_games_to_csv, args=(headers,
                                               list(games[:num_games_per_file]),
                                               os.path.join(destination_dir,
                                                            f'{destination_file_counter}.csv'))).start()


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


# TODO Change code related to castling. Encode promoting a pawn to something else than a queen.
def encode_move_8x8x73(move):
    encoded_position = np.zeros((8, 8, 68), dtype=np.float32)
    if move == 'e1g1':  # white short
        encoded_position[6][7][64] = 1
    elif move == 'e1c1':  # white long
        encoded_position[2][7][65] = 1
    elif move == 'e8g8':  # black short
        encoded_position[6][0][66] = 1
    elif move == 'e8c8':  # black long
        encoded_position[2][0][67] = 1
    else:
        move = [(ord(move[0]) - ord('`')), int(move[1]), (ord(move[2])-ord('`')), int(move[3])]
        coords = move[3] - move[1], move[2] - move[0]
        move[2] -= 1
        move[3] -= 1
        if coords[0] != 0 and coords[1] != 0 and abs(coords[0]) != abs(coords[1]):  # knight
            if coords[0] > 0:  # up
                if coords[1] > 0:  # right
                    if abs(coords[0]) > abs(coords[1]):  # more vertical
                        encoded_position[move[2]][7 - move[3]][56] = 1
                    elif abs(coords[0]) < abs(coords[1]):  # more horizontal
                        encoded_position[move[2]][7 - move[3]][57] = 1
                elif coords[1] < 0:  # left
                    if abs(coords[0]) > abs(coords[1]):  # more vertical
                        encoded_position[move[2]][7 - move[3]][63] = 1
                    elif abs(coords[0]) < abs(coords[1]):  # more horizontal
                        encoded_position[move[2]][7 - move[3]][62] = 1

            elif coords[0] < 0:  # down
                if coords[1] > 0:  # right
                    if abs(coords[0]) > abs(coords[1]):  # more vertical
                        encoded_position[move[2]][7 - move[3]][59] = 1
                    elif abs(coords[0]) < abs(coords[1]):  # more horizontal
                        encoded_position[move[2]][7 - move[3]][58] = 1

                elif coords[1] < 0:  # left
                    if abs(coords[0]) > abs(coords[1]):  # more vertical
                        encoded_position[move[2]][7 - move[3]][60] = 1
                    elif abs(coords[0]) < abs(coords[1]):  # more horizontal
                        encoded_position[move[2]][7 - move[3]][61] = 1

        elif coords[1] == 0:  # vertical
            if coords[0] > 0:  # up
                encoded_position[move[2]][7 - move[3]][coords[0] -1] = 1
            elif coords[0] < 0:  # down
                encoded_position[move[2]][7 - move[3]][abs(coords[0]) + 28 -1] = 1

        elif coords[0] == 0:  # horizontal
            if coords[1] > 0:  # right
                encoded_position[move[2]][7 - move[3]][coords[1] + 14 -1] = 1
            elif coords[1] < 0:  # left
                encoded_position[move[2]][7 - move[3]][abs(coords[1]) + 42 -1] = 1

        elif coords[0] != 0 and coords[1] != 0:  # diagonal
            if coords[1] > 0:  # right
                if coords[0] > 0:  # right up
                    encoded_position[move[2]][7 - move[3]][coords[1] + 7 -1] = 1
                elif coords[0] < 0:  # right down
                    encoded_position[move[2]][7 - move[3]][abs(coords[1]) + 21 -1] = 1

            elif coords[1] < 0:  # left
                if coords[0] > 0:  # left up
                    encoded_position[move[2]][7 - move[3]][coords[0] + 49 -1] = 1
                elif coords[0] < 0:  # left down
                    encoded_position[move[2]][7 - move[3]][abs(coords[1]) + 35 -1] = 1

    x = np.where(encoded_position == 1)
    print(x)
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
