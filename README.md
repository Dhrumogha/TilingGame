# TilingGame
This application pertains to a tiling game on a checkerboard. Players MAX and MIN take turns placing non-overlapping dominoes on an m x n checkerboard until no more can be placed. The goal of MAX (resp. MIN) is to maximize (resp. minimize) the number of dominoes on the board. Under optimal play, this number reaches a certain value that we will call minimax(player, m, n) where player is the one who makes the first move. 

For example, suppose we label the squares of the board by (i, j) i = 0,...,m-1 and j = 0,..., n-1. Then given a 1x6 board, MAX's first move would be [(0,2),(0,3)] resulting in minimax(MAX,1,6) = 3. On the other hand if MIN plays first, then his first move would be [(0,1),(0,2)] and minimax(MIN, 1, 6) = 2.

Once the minimax value is computed, this can be converted to a game. After all the moves are made, the number of dominoes on the board is compared with the minimax value. If it's more, then MAX is the winner and if it's less, then MIN is the winner. If the two values are equal then its a draw.

I used the minimax algorithm on a game tree to compute the minimax value. The nodes of the tree correspond to the board positions. The nodes are labeled MAX or MIN depending on whether it's MAX's or MIN's turn to play. The number of children of a node are the number of possible moves from the corresponding position. At a terminal node, all the moves are completed and we can define its value as the number of dominoes on the board. We can then back these values up the tree in the following way:
The value of a MAX (resp. MIN) node is the maximum (minimum) of the values of its children. The value of the root node is then the minimax value of the game.

I use the standard technique of alpha-beta pruning to speed up the algorithm. I save intermediate states using transposition tables. I only save the states within a threshold level as shallower nodes take longer to evaluate. I calculate the threshold by estimating the number of nodes within a given level, the size of each storage unit using an available memory budget. Further I only store nodes that are fully evaluated (i.e. not pruned) or at level 1.

To improve the effect of pruning, I order the nodes at each level by estimating the score of each child of the node. For this I prune the tree to a specific depth below that level and use an evaluation function to estimate the value of this node. If, at this node there are k dominoes on the board and the unoccupied squares can be split into r components with sizes x1,..., xr, then the value = k + [x1/2] + ... + [xr/2] where [.] is the floor function. Obviously this is an overestimate.

On command line, the program computes the minimax value and prints a progress report (type python <filename.py> -h for usage information). Apart from this the tilingGame class also provides interfaces for playing the game for given lookahead levels for MAX & MIN.

Computation results:
These are stored in the file tilingResults.txt. For each run, I store the timestamp, the board dimensions, the lookahead levels (None indicates minimax computation), the minimax value, final board and time taken. The final board is represented as a 2-d list L where L[i][j] = k if the kth move occupies the (i, j)th square. Zero entries in L correspond to unoccupied squares.

I have done the computations using different versions of the application, so some of the times may look bizarre. The really long times count the intervals that my machine was hibernated, so they are not accurate. I've been able to go up to 30 squares (5x6) but beyond this, the times are just way too long. I would really like to see the solution for the standard 8x8 checkerboard.

Suggestions on improving the efficiency are welcome. 

Reference: “e-Valuate: A Two-player Game on Arithmetic Expressions”, Sarang Aravamuthan and Biswajit
Ganguly, CoRR abs/1202.0862: (2012). Available online at https://arxiv.org/pdf/1202.0862v4.pdf

