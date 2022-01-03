import numpy as np
import os
import chess
import chess.pgn


def create_dataset_from_pgn(pgn_file_path, destination_dir, cond_func, **kwargs):
    """ Function that reads pgn files one by one
        based on assumption that every pgn contains exactly two empty lines.
        Calls check_elo() on every pgn

        Parameters:
            pgn_file_path (str): path to input file
            destination_dir (str): where to save generated dataset
            cond_func: pointer to the function that takes a pgn and returns bool
    """
    with open(pgn_file_path) as input_file:
        counter = 0
        while True:
            game = chess.pgn.read_game(input_file)
            if game is None:
                break
            if not cond_func(game, **kwargs):
                continue
            game_dir = os.path.join(destination_dir, f'game{counter}')
            os.mkdir(game_dir)
            with open(os.path.join(game_dir, "game.pgn"), "w") as one_game_file:
                print(game, file=one_game_file)
            counter += 1
            board = game.board()
            for actual_index, actual_move in enumerate(game.mainline_moves()):
                for legal_index, legal_move in enumerate(board.legal_moves):
                    if legal_move == actual_move:
                        continue
                    board.push(legal_move)  # add move
                    with open(os.path.join(game_dir, f"pos{actual_index}_legal{legal_index}.txt"), "w") as data_file:
                        print(board.fen(), file=data_file)
                    board.pop()  # undo move
                board.push(actual_move)  # add correct move
                with open(os.path.join(game_dir, f"pos{actual_index}.txt"), "w") as data_file:
                    print(board.fen(), file=data_file)


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
        encoded_position[i / 8, i % 8, piece_channel_dict[flatten_board[i]]] = 1

    encoded_position[:, :, 12] = on_move
    encoded_position[:, :, 13] = white_kingside_castle
    encoded_position[:, :, 14] = white_queenside_castle
    encoded_position[:, :, 15] = black_kingside_castle
    encoded_position[:, :, 16] = black_queenside_castle
    encoded_position[en_passant_index[0], en_passant_index[1], 17] = 1 if en_passant_index is not None else 0

    return encoded_position


def check_elo(game, min_elo, max_elo):
    if (min_elo <= int(game.headers['WhiteElo']) <= max_elo) and (min_elo <= int(game.headers['BlackElo']) <= max_elo):
        return True
    else:
        return False
