# Tic-Tac-Toe Q-Learning

This project is a small reinforcement learning exercise using 3x3 tic-tac-toe.

The goal is to train `X` as a learning agent against a random `O` opponent using a tabular Q-learning approach. The board is small enough that we do not need PyTorch or neural networks yet; a Python dictionary is enough to store the Q-table.

## Board representation

The board is represented as a NumPy array of length 9.

```text
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

The values in the array mean:

```text
 0  = empty square
 1  = X, the learning agent
-1  = O, the random opponent
```

For example:

```python
np.array([ 1,  0, -1,
           0,  1,  0,
          -1,  0,  0])
```

means X has marks at positions `0` and `4`, while O has marks at positions `2` and `6`.

## What the code implements

The code is organized into four parts.

### 1. Environment

These functions define the tic-tac-toe rules:

```python
legal_moves(board)
make_move(board, move, player)
winner(board)
game_over(board)
reward(board)
```

They answer questions such as:

- Which moves are legal?
- What does the board look like after a move?
- Has X or O won?
- Is the game finished?
- What reward should X receive?

The reward is always from X's perspective:

```text
X wins: +1
O wins: -1
Draw:    0
Ongoing: 0
```

### 2. Agent utilities

These functions define the Q-table interface and action selection:

```python
board_state(board)
get_q(q_table, board, move)
max_q(q_table, board)
best_move(q_table, board)
random_move(board)
choose_move(q_table, board, epsilon)
```

The Q-table is a dictionary with keys of the form:

```python
(board_state, move)
```

For example:

```python
q_table[((0, 0, 0, 0, 0, 0, 0, 0, 0), 4)] = 0.8
```

means:

> From the empty board, move `4` has estimated value `0.8`.

The agent uses an epsilon-greedy policy:

```text
with probability epsilon: choose a random move
with probability 1 - epsilon: choose the best known move
```

### 3. Learning

The central Q-learning update is:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max Q(s_next, a_next) - Q(s, a))
```

In code, this is implemented by:

```python
update_q(q_table, board, move, step_reward, next_board, alpha, gamma)
```

Training is done with:

```python
train_one_game(q_table, alpha, gamma, epsilon)
train_many_games(n_games, alpha=0.1, gamma=0.9)
```

The current training setup is:

```text
X = learning agent
O = random opponent
```

The code also includes a small improvement: if O wins immediately after X's previous move, the previous X move is punished with reward `-1`.

### 4. Evaluation

After training, the learned agent can be tested with:

```python
play_trained_game(q_table)
evaluate_agent(q_table, n_games)
```

During evaluation:

```text
X uses the learned Q-table greedily
O still plays randomly
```

No learning happens during evaluation.

## How to run

Install NumPy if needed:

```bash
pip install numpy
```

Run the script:

```bash
python tictactoe_q_learning.py
```

Example output will look like:

```text
Number of learned state-action pairs: 1234
Evaluation over 1,000 games: (950, 10, 40)

Opening move Q-values:
Move 0: 0.0000
Move 1: 0.0000
...
```

The exact numbers will vary because training includes randomness.

## Current limitation

This version learns useful late-game tactics, but it does not yet fully propagate rewards back to early moves.

For example, the empty-board Q-values may remain close to zero even if the agent performs well during evaluation. This happens because the current training loop often updates X immediately after X moves, before the long-term consequences of that move are fully known.

In other words, the agent learns things like:

```text
This move wins immediately.
This move allowed O to win immediately.
```

But it does not yet fully learn:

```text
This opening move eventually led to a better position several turns later.
```

That is the main next step.

## Next step: update using X-to-X transitions

The next improvement is to define one learning step from one X-turn to the next X-turn.

Instead of updating immediately after X moves, the training loop should treat this as one transition:

```text
A_old   = board when it is X's turn
move    = X's chosen move
A_mid   = board after X moves
A_next  = board after O responds, when it is X's turn again
reward  = final reward if the game ended, otherwise 0
```

Then update:

```text
Q(A_old, move) using A_next
```

This lets rewards propagate backward through the Q-table over many games.

The intended structure is:

```text
start with an empty board

while the game is not over:
    save current board as A_old

    X chooses a move
    apply X's move

    if X wins or the game is a draw:
        update Q(A_old, move) using the final reward
        stop

    O chooses a random move
    apply O's move

    if O wins or the game is a draw:
        update Q(A_old, move) using the final reward
        stop

    update Q(A_old, move) with reward 0 and next_board = board after O's move
```

This is closer to the real reinforcement learning transition:

```text
state -> action -> reward -> next_state
```

where `next_state` is the next state where the learning agent gets to act again.

## Possible later extensions

After implementing X-to-X updates, useful extensions are:

1. Train both X and O instead of only training X.
2. Add self-play, where both sides use learned policies.
3. Compare random play, greedy Q-play, and epsilon-greedy Q-play.
4. Save and load the Q-table with `pickle` or JSON.
5. Add a small command-line interface so a human can play against the trained agent.
6. Replace the Q-table with a neural network later, turning this into a small Deep Q-Learning project.

## Learning goal

This project is intentionally small. The point is not to make the strongest possible tic-tac-toe engine. The point is to understand the core reinforcement learning loop:

```text
state -> action -> reward -> next_state -> update value estimate
```
