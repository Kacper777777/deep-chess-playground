import numpy as np
import chess
import chess.pgn


def convert_fen_to_matrix(fen):
    fen_elements = fen.split(' ')
    flatten_board = str(chess.Board(fen))
    flatten_board = flatten_board.replace(' ', '').replace('\n', '')
    on_move = 1 if fen_elements[1] == 'w' else 0
    white_kingside_castle = 1 if "K" in fen_elements[2] else 0
    white_queenside_castle = 1 if "Q" in fen_elements[2] else 0
    black_kingside_castle = 1 if "k" in fen_elements[2] else 0
    black_queenside_castle = 1 if "q" in fen_elements[2] else 0
    en_passant = fen_elements[3]  # TODO
    half_moves = int(fen_elements[4])
    full_moves = int(fen_elements[5])

    encoded_position = np.zeros((8, 8, 20), dtype=np.float32)



    return encoded_position
