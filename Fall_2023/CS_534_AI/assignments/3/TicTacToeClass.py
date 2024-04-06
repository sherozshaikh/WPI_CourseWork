import sys
import random
import numpy as np
from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, utility, board, moves')


# ______________________________________________________________________________
# Sample Games

class TicTacToe():
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', utility=0, board={},
                                 moves=[(x, y) for x in range(1, h + 1) for y in range(1, v + 1)])

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()
        return None

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k


# ______________________________________________________________________________
# User Query Search

def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


# ______________________________________________________________________________
# Random Search

def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


# ______________________________________________________________________________
# MinMax Search

def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


def minimax_player(game, state):
    """Player using the MiniMax to make decisions."""
    return minmax_decision(state, game)


# ______________________________________________________________________________
# Alpha Beta Search

def alpha_beta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alpha_beta_player(game, state):
    """Player using Alpha-Beta Search to make decisions."""
    return alpha_beta_search(state, game)


def heuristic_alpha_beta_player(game, state):
    """Player using Heuristic Alpha-Beta Search to make decisions."""
    return alpha_beta_cutoff_search(state, game)


# ______________________________________________________________________________
# Monte Carlo Tree Search

class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent=None, state=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None


def ucb(n, C=1.4):
    return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)


def monte_carlo_tree_search(state, game, N=1000):
    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state):
        """simulate the utility of current state by random picking a step"""
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)

    for _ in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)


def mcts_player(game, state):
    """Player using Monte Carlo Tree Search to make decisions."""
    return monte_carlo_tree_search(state, game)


# ______________________________________________________________________________
# Simulations

given_players: dict = {
    1: 'Random Player',
    2: 'MiniMax Player',
    3: 'Alpha Beta Player',
    4: 'Heuristic Alpha Beta Player',
    5: 'MCTS Player',
    6: 'Query Player',
}


def _get_players(ques: str) -> int:
    """Get the player selection from the user based on the given players list."""
    player_cond: bool = True
    player_selected: int = int(input(f'\nPlease enter your {ques} player: '))
    while player_cond:
        if given_players.get(player_selected, 0) == 0:
            player_selected: int = int(input(f'\nCould not find your player, please try again: '))
            continue
        else:
            player_cond: bool = False
    return player_selected


def _actionable_items(id1, game, nextState) -> tuple:
    """Perform an action based on the player ID and return the result as a tuple."""
    if id1 == 1:
        return random_player(game, nextState)
    elif id1 == 2:
        return minimax_player(game, nextState)
    elif id1 == 3:
        return alpha_beta_player(game, nextState)
    elif id1 == 4:
        return heuristic_alpha_beta_player(game, nextState)
    elif id1 == 5:
        return mcts_player(game, nextState)
    elif id1 == 6:
        return query_player(game, nextState)
    else:
        return tuple()


class TicTacToeSimulation():
    """
    A class for running a Tic-Tac-Toe game between two players.
    
    Methods:
            game_simulation(): Play a Tic-Tac-Toe game between two players."""

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def game_simulation(self) -> None:

        print(
            '\nPlayer Selection:\n1. Random Player\n2. MiniMax Player\n3. Alpha Beta Player\n4. Heuristic Alpha Beta Player\n5. MCTS Player\n6. Query Player\n')

        current_round: int = 0
        winner_x: int = 0
        winner_o: int = 0
        cond1: bool = True

        first_player, second_player = _get_players(ques='first'), _get_players(ques='second')
        print()

        while cond1:

            current_round += 1
            game = TicTacToe()
            nextState = game.initial
            print()

            player = nextState.to_move
            print(f'Round {current_round}:\n')
            game.display(nextState)
            print(f'\nAvailable Action by the Player {player}: {game.actions(nextState)}\n')

            ply1_PlayerAction = _actionable_items(first_player, game, nextState)
            print(f'The Action by the Player {nextState.to_move} is {ply1_PlayerAction}\n')
            nextState = game.result(nextState, ply1_PlayerAction)
            game.display(nextState)
            print()
            utility = game.compute_utility(nextState.board, ply1_PlayerAction, player)
            print(f"Player {player}'s Utility: {utility}")
            print()

            ply2_PlayerAction = _actionable_items(second_player, game, nextState)
            nextState = game.result(nextState, ply2_PlayerAction)
            game.display(nextState)
            print()
            player = nextState.to_move
            utility = game.compute_utility(nextState.board, ply2_PlayerAction, player)
            print(f"Player {player}'s Utility: {utility}")
            print()

            terminate = game.terminal_test(nextState)

            while (not terminate):
                player = nextState.to_move
                if player == 'X':
                    ply1_PlayerAction = _actionable_items(first_player, game, nextState)
                    nextState = game.result(nextState, ply1_PlayerAction)
                    game.display(nextState)
                    print()
                    utility = game.compute_utility(nextState.board, ply1_PlayerAction, 'X')
                    utility = abs(utility)
                    print(f"Player {player}'s Utility: {utility}")
                    if utility == 1:
                        print(f"Player {player} won the game.")
                        winner_x += 1
                    terminate = game.terminal_test(nextState)
                    print()
                else:
                    ply2_PlayerAction = _actionable_items(second_player, game, nextState)
                    nextState = game.result(nextState, ply2_PlayerAction)
                    game.display(nextState)
                    print()
                    utility = game.compute_utility(nextState.board, ply2_PlayerAction, 'O')
                    utility = abs(utility)
                    print(f"Player {player}'s Utility: {utility}")
                    if utility == 1:
                        print(f"Player {player} won the game.")
                        winner_o += 1
                    terminate = game.terminal_test(nextState)
                    print()

            if ((current_round == 3) or (winner_x == 2) or (winner_o == 2)):
                cond1: bool = False

        if winner_x == 2:
            print('Player X won the game in two out of three rounds')
        elif winner_o == 2:
            print('Player O won the game in two out of three rounds')
        else:
            print('No Player can win two out of three rounds in the game. The game was a draw.')
        return None


if __name__ == '__main__':
    ttt_obj = TicTacToeSimulation()
    ttt_obj.game_simulation()
    sys.exit()
