import re
import io
import chess
import chess.pgn

_i = 0
_INDEX = 0

EXIT = 0#set if you want stop after generating X files
BREAK = 0#set if you want stop after checking first X pgns
#better not to run with both at 0s

def read_pgn(in_file, min_elo, max_elo, max_moves):
    """ Function that reads file pgns one by one
    Based on assumption that every pgn contains exactly two empty lines
    Calls check_elo() on every pgn

    Parameters:
        in_file (str): path to input file

    Parameters passed:
        min_elo (int): minimal acceptable elo range (itself included)
        max_elo (int): maximal acceptable elo range (itself included)
        max_moves (int): how many half-moves to process, set to 0 to process all
    """
    all_moves, done_moves = 0, 0
    with open(in_file) as input:
        empty_counter, lines = 0, []
        for line in input:
            lines.append(line)
            if lines[-1] == '\n':
                empty_counter += 1
            if empty_counter == 2:
                temp = check_elo(lines, min_elo, max_elo, max_moves)
                all_moves += temp[0]
                done_moves += temp[1]
                print(all_moves, done_moves)
                global _i
                _i += 1
                if _i == BREAK and BREAK != 0:
                    break
                lines = []
                empty_counter = 0

def check_elo(lines, min_elo, max_elo, max_moves):
    """ Function that uses regex to check if both players elo are in range
    Fuction is looking for '[WhiteElo "' + 3 or 4 digits and
    '[BlackElo "' + 3 or 4 digits
    Calls pgn_to_fen() on every pgn that elos are in range

    Parameters:
        min_elo (int): minimal acceptable elo range (itself included)
        max_elo (int): maximal acceptable elo range (itself included)

    Parameters passed:
        max_moves (int): how many half-moves to process, set to 0 to process all
    """
    text = ' '.join(lines)
    elo = re.search('\[WhiteElo "(\d{3,4})', text), re.search('\[BlackElo "(\d{3,4})', text)
    if elo[0] is not None and elo[1] is not None:
        elo = [int(x[1]) for x in elo] #x[1] is a capturing group
        elo = [True if (x >= min_elo and x <= max_elo) else False for x in elo]
        if all(elo) and (lines[-2])[0:2] == '1.':
            return pgn_to_fen(lines[-2], max_moves)
        else:
            return 0, 0

def pgn_to_fen(pgn, max_moves):
    """ Function that uses python chess
    Calls fens_to_matrix on every pgn that elos are in range

    Parameters:
        max_moves (int): how many half-moves to process, set to 0 to process all
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    moves, legal_moves_sum = 0, 0
    for idx, move in enumerate(game.mainline_moves()):
        if max_moves > 0 and idx == max_moves:
            break #return None#function break
        fens = {}
        legal_moves_sum += board.legal_moves.count()
        moves += 1
        for item in board.legal_moves:
            board.push(item)# add move
            fens[board.fen()] = 0# add label to new fen
            board.pop()# undo move
        board.push(move)#add correct move
        fens[board.fen()] += 1#add label to correct move
        fens_to_matrix(fens)
    return legal_moves_sum, moves

def fens_to_matrix(fens):
    """ Function that iterates over fens
    Calls to_matrix() on every fen, label pair

    Parameters:
        fens (dict): dict of {fen: label}, only one label is 1, others should be 0

    Constants:
        _INDEX (int): value for filename '{_INDEX}_{label}'
    """
    global _INDEX
    for fen, label in fens.items():
        with open(f'dataset/{_INDEX}_{label}.txt', 'w') as file:
            file.write(''.join(str(to_matrix(fen))))
        _INDEX += 1
        if _INDEX == EXIT and EXIT != 0:
            exit()

def to_matrix(fen):
    """ Function that changes chessboard representation to matrix
    Calls to_matrix() on every fen, label pair

    Parameters:
        fens (str): string containing fen

    Returns:
        output (list of lists): matrix
    """
    flatten_board = ''.join(str(chess.Board(fen)))
    flatten_board = flatten_board.replace(' ', '').replace('\n', '')
    line, output = [], []
    for idx, item in enumerate(flatten_board):
        line.append(translate(item))
        if idx % 8 == 7:#split every 8 because one line of chessbord is 8 squares
            output.append(line)
            line = []
    return output

def translate(piece):
    """ Function that translates piece representation to one-hot encoding

    Parameters:
        piece (char): letter representing chess piece, '.' will return list of zeros

    Returns:
         one_hot (list): one-hot encoded piece
    """
    one_hot = [0 for _ in range(12)]
    if piece == '.':
        return one_hot# return list of zeros
    dict = {
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
    one_hot[dict[piece]] += 1
    return one_hot

def profile():
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        read_pgn('chess_data.pgn', 1001, 1400, 0)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='prof.prof')

if __name__ == '__main__':
    # profile()
    read_pgn('chess_data.pgn', 1001, 1400, 0)
