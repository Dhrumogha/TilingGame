'''This module defines the TilingGame class that implements a two-player tiling game and computes
the minimax value under optimal play.'''

MEMORY_BUDGET = 1.5*(2**30)        # 1.5GB available.
NODE_SIZE = 40                # approximately 40 bytes of storage per node of tree.

def maxdepth(m, n):
    '''Return max depth we can store nodes up to. Also return memory used in MB and memory used if
    we store 1 level further. This underestimates the depth since we don't account for the dominoes
    overlapping.'''
    def choose(n, k):
        '''Return {n choose k}.'''
        from math import factorial
        return factorial(n)//(factorial(k)*factorial(n-k))

    max_nodes = MEMORY_BUDGET//NODE_SIZE    # max. no. of nodes that can be stored.
    j = 1
    consmem = 0
    while consmem <= max_nodes and j < m*n/2:
        mem = choose(m*n, j)*(2**j)      # (approx.) no. of ways to place j dominoes on an mxn board
        oldc = consmem
        consmem += mem
        j += 1
    return j-1, int(NODE_SIZE*oldc/2**20), int(NODE_SIZE*consmem/2**20)


class TilingGame(object):
    '''Compute minimax value for a 2-player tiling game with dominoes on an m x n board.'''

    MAX, MIN = 1, -1    # The players: These are class variables.
    def __init__(self, m, n, player=None):
        '''Initializes the game to be played on an m x n board.'''
        from collections import defaultdict
        self.dim = m, n
        self.player = player or TilingGame.MAX                    # player moves first.
        # Current list of available moves. A move is a pair of adjacent squares where each square is
        # represented by its coordinates.
        self.available_moves = [((i, j), (i, j+1)) for i in range(m) for j in range(n-1)] + \
        [((i, j), (i+1, j)) for j in range(n) for i in range(m-1)]
        # Make 1st move optimal(?) move.
        if ((m > 4 and n > 1) or (m > 1 and n > 4)) and player == TilingGame.MAX:
            rem_e = ((2, 0), (2, 1)) if m > 4 else ((0, 2), (1, 2))
            self.available_moves.remove(rem_e)
            self.available_moves.insert(0, rem_e)
        self.occupied_squares = list()   # List of occupied squares in the order they were occupied.
        # For each move, store the list of moves intersected by this. Used for fast move updates.
        self.intersects = dict()
        for move in self.available_moves:
            temp = set()
            for x in self.available_moves:
                if move[0] in x or move[1] in x:
                    temp.add(x)
            self.intersects[move] = temp

        # Estimated score, (optimal score, remaining sequence of moves) at a given position.
        # Used in the min-max algorithm.
        if not hasattr(self, "saved_states"):
            self.saved_states = defaultdict(dict)
        self.storage_level = maxdepth(m, n)[0]            # The max depth to save states

    def score(self, level=0):
        '''Return the estimated score of each child of current node for a given lookahead level.
        This is (only) used for ordering the moves at a node.'''
        # Game over or zero depth? Return exact/estimated no. of moves.
        if not level or not self.available_moves:
            occupied_sorted = tuple(sorted(self.occupied_squares))
            return [self.estimate_moves(occupied_sorted)]
        available_moves1 = self.available_moves[:]        # save state
        child_scores = []                    # Scores of the children of current node.
        for m in available_moves1:
            self.update_state(m)        # Update the board after move
            child_score = self.score(level-1)
            self.player = -self.player          # Reset state
            self.available_moves = available_moves1[:]
            self.occupied_squares.pop()
            self.occupied_squares.pop()
            if self.player == TilingGame.MAX:
                new_max = min(child_score)
            else:
                new_max = max(child_score)
            child_scores.append(new_max)    # Update child scores.
        return child_scores

    def score_alphabeta(self, level=-1, parent_val=None):
        '''Minimax value with alpha-beta pruning, node ordering and caching of intermediate values.
        parent_val is the current min/max value of the parent node. Returns the minimmax value
        and the sequence of moves leading to this value.'''

        max_val = self.dim[0]*self.dim[1]//2 + 1            # max. possible depth of tree.
        parent_val = parent_val or max_val*self.player        # Set default value for parent_val.

        occupied_sorted = tuple(sorted(self.occupied_squares))
        # Return minimax value only when computing it.
        if occupied_sorted in self.saved_states and level < 0:
            return self.saved_states[occupied_sorted]

        # Game over or zero depth? return exact or estimated no. of moves.
        if not self.available_moves or not level:
            return self.estimate_moves(occupied_sorted), [], True

        max_level = -int(max_val**0.5)      # Apply node ordering only above this level.
        if level < 0 and level >= max_level:
            lookahead = level - max_level + 1
            child_scores = self.score(level=lookahead)    # Score of each available move
            # Sort available moves based on their scores; reverse the order for a MAX node.
            self.available_moves = sorted(self.available_moves,
            key=lambda x: child_scores[self.available_moves.index(x)], reverse=(self.player+1)//2)

        available_moves1 = self.available_moves[:]          # Save state
        current_max = -self.player*max_val  # Current node score; -infty for MAX and +infty for MIN

        for m in available_moves1:
            import sys
            import time
            t1 = time.time()
            self.update_state(m)        # Update the board after move
            # Set to default on 1st iteration
            val_to_pass = None if abs(current_max) == max_val else current_max
            new_max, new_moves, exact_val = self.score_alphabeta(level-1, val_to_pass)

            # A new min/max value and computed value is not exact? recompute!
            if self.player*new_max < self.player*current_max and not exact_val:
                occupied_sorted1 = tuple(sorted(self.occupied_squares))
                self.saved_states.pop(occupied_sorted1, None)      # Remove entry from dictionary.
                new_max, new_moves, exact_val = self.score_alphabeta(level-1, None)

            if self.player*new_max < self.player*current_max:
                current_max = new_max
                max_moves = new_moves
                next_move = m
            self.available_moves = available_moves1[:]        # reset state.
            self.occupied_squares.pop()
            self.occupied_squares.pop()
            self.player = -self.player
            # Alphabeta prune? Save state as approximate value and return.
            if self.player*new_max >= self.player*parent_val and m is not available_moves1[-1]:
                self.update_saved_states(occupied_sorted, (current_max, [], False))
                return new_max, [], False
            # Top level? print %of tree searched, current size of saved states, current optimum and
            # sequence of moves leading to this optimum.
            if level == -1:
                print('Searched ', round(100*(1+available_moves1.index(m))/len(available_moves1), 1),
                '% of tree  ', round(time.time() - t1, 1), ' seconds ', len(self.saved_states),
                " states. Size: ", round(sys.getsizeof(self.saved_states)/2**20, 1), "MB",
                " Current Optimum: ", current_max, " move:", m, sep="")
        return_val = current_max, [next_move] + max_moves, True
        if level < 0:        # Save the state only if it corresponds to a minimax value
            self.update_saved_states(occupied_sorted, return_val)

        return return_val

    def update_saved_states(self, occupied_sorted, return_val):
        '''Update the saved states with new minimax value, path leading to the value and whether
        it's accurate or an (pruned) estimate.'''
        m, n = self.dim
        r0, r1, r2 = return_val

        def save_1or2():
            '''Save the state and its reflections on axes of symmetry.'''
            self.saved_states[occupied_sorted] = return_val

            # Reflect on Y-axis
            board_refl_y = tuple(sorted((x, n - y-1) for (x, y) in occupied_sorted))
            ry = (r0, [((a, n-1-b), (c, n-1-d)) for ((a, b), (c, d)) in r1], r2)
            self.saved_states[board_refl_y] = ry

            # Reflect on X-axis.
            board_refl_x = tuple(sorted((m - x-1, y) for (x, y) in occupied_sorted))
            rx = (r0, [((m-1-a, b), (m-1-c, d)) for ((a, b), (c, d)) in r1], r2)
            self.saved_states[board_refl_x] = rx

            # Reflect w.r.t. Y followed by X-axis.
            board_refl_xy = tuple((m - x-1, n-1-y) for (x, y) in reversed(occupied_sorted))
            rxy = (r0, [((m-1-a, n-1-b), (m-1-c, n-1-d)) for ((a, b), (c, d)) in r1], r2)
            self.saved_states[board_refl_xy] = rxy

        if len(self.occupied_squares) > 2*self.storage_level:     #beyond allowed storage? return.
            return
        save_1or2()
        if m == n:                  # Reflect along diagonal and resave
            occupied_sorted = tuple(sorted((y, x) for (x, y) in occupied_sorted))
            return_val = (r0, [((b, a), (d, c)) for ((a, b), (c, d)) in r1], r2)
            save_1or2()

    def update_state(self, move):
        '''Update the 'state' of game after move is made.'''
        self.occupied_squares.extend(move)
        self.player = -self.player
        # Update available moves to those that don't intersect current move.
        clash = self.intersects[move]
        self.available_moves = [x for x in self.available_moves if x not in clash]

    def estimate_moves(self, occupied_sorted):
        '''Return the estimated score of current node. Return from cache if
        available, else store the result in cache.'''
        import itertools
        m, n = self.dim
        if not self.available_moves:        # terminal node? return no. of dominoes on board.
            return len(self.occupied_squares)//2

        size = len(occupied_sorted) // 2            # no. of dominoes so far.
        board = [[0]*n for _ in range(m)]           # an mxn board to generate a maximal matching
        for (x, y) in occupied_sorted:
            board[x][y] = 1
        for i, j in itertools.product(range(m), range(n)):
            if not board[i][j]:
                if j < n-1 and not board[i][j+1]:
                    board[i][j] = board[i][j+1] = 1
                    size += 1
                elif i < m-1 and not board[i+1][j]:
                    board[i][j] = board[i+1][j] = 1
                    size += 1
        return size

    def find_move(self, lookahead=None):
        '''Find the best move for current player given a lookahead factor.'''
        moves = self.score_alphabeta(level=lookahead)[1]
        return moves[0]

    def play_game(self, max_lookahead=None, min_lookahead=None):
        '''Play the game for the given lookahead levels for MAX and MIN. Return the final board and
        number of moves.'''
        while self.available_moves:
            lookahead = max_lookahead if self.player == TilingGame.MAX else min_lookahead
            next_move = self.find_move(lookahead)
            self.update_state(next_move)

        # Game over. Recreate final board, reset to init state and return.
        m, n = self.dim
        board = [[0]*n for _ in range(m)]    # board: mxn 2-d list of zeros.
        # board[x][y] = i if the ith move occupies square (x, y)
        for i, (x, y) in enumerate(self.occupied_squares):
            board[x][y] = 1 + i//2
        num_moves = len(self.occupied_squares)//2
        self.__init__(m, n)
        return num_moves, board


def test_minimax(m, n, player=None, max_l=-1, min_l=-1, filep=None):
    '''test the TilingGame class'''
    print("board dimensions = ", m, " X ", n, "; Lookahead levels for MAX and MIN = ", \
    max_l, ' ', min_l, ". First player: ", player, sep='', file=filep)

    player = TilingGame.MAX if player == 'MAX' else TilingGame.MIN
    myb = TilingGame(m, n, player)

    r0, r1, _ = myb.score_alphabeta(level=max_l)
    myboard = [[0]*n for _ in range(m)]
    for i, [(x, y), (w, z)] in enumerate(r1, 1):
        myboard[x][y] = myboard[w][z] = i
    print("(minimax, final board) = (", r0, ", ", myboard, ")", file=filep, sep="")

def main():
    '''Run the code and save output in filename.'''
    import argparse
    s = "Players MAX and MIN take turns to place non-overlapping dominoes on an mxn checkerboard \
    until no more can be placed. Return the minimax value and final board position under optimal \
    play."
    parser = argparse.ArgumentParser(description=s)
    parser.add_argument("m", help="the number of rows", type=int)
    parser.add_argument("n", help="the number of columns", type=int)
    parser.add_argument("-p", help="the first player (MAX by default)", choices=['MAX', 'MIN'],
                        default='MAX')
    parser.add_argument("-f", help="file to append output to", metavar="filename")
    args = parser.parse_args()

    max_l = min_l = -1
    filename = args.f
    import sys
    import time
    t1 = time.time()
    myf = open(filename, "a") if filename else sys.stdout
    print('\n', time.asctime(), sep='', file=myf)
    test_minimax(args.m, args.n, args.p, max_l, min_l, myf)

    print("Time taken = ", round(time.time() - t1, 2), "seconds", file=myf)
    if filename:
        myf.close()

if __name__ == '__main__':
    main()

