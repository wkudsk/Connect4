import numpy as np

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    #MAX_DEPTH = 10

    def findOtherPlayerNumber(self):
        if(self.player_number == 1):
            return 2
        else:
            return 1

    def game_completed(self, board):
        player_1_win_str = '{0}{0}{0}{0}'.format(1)
        player_2_win_str = '{0}{0}{0}{0}'.format(2)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_1_win_str in to_str(row):
                    return 1
                elif player_2_win_str in to_str(row):
                    return 2
            return 0

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_1_win_str in to_str(root_diag):
                    return 1
                elif player_2_win_str in to_str(root_diag):
                    return 2

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_1_win_str in diag:
                            return 1
                        elif player_2_win_str in diag:
                            return 2

            return 0
        
        return (check_horizontal(board) or
            check_verticle(board) or
            check_diagonal(board))

    def update_board(self, board, move, player_num):
        if 0 in board[:,move]:
            update_row = -1
            for row in range(1, board.shape[0]):
                update_row = -1
                if((board[row, move] > 0) and (board[row-1, move] == 0)):
                    update_row = row-1
                elif row==board.shape[0]-1 and board[row, move] == 0:
                    update_row = row
                if update_row >= 0:
                    board[update_row, move] = player_num
                    return True
            
        else:
            return False

    def MAX_VALUE(self, b, alpha, beta, depth):
        
        if(self.game_completed(b) == self.findOtherPlayerNumber() or depth ==  1):
            return self.evaluation_function(b)*depth
        
        maxAction = -10000
        for i in range(3, 10):
            board = b.copy()
            didBoardUpdate = self.update_board(board, i%7, self.player_number)
            if(didBoardUpdate):
                action = self.MIN_VALUE(board, alpha, beta, depth - 1)
                
                maxAction = max(action, maxAction)
                alpha = max(alpha, maxAction)
                if alpha >= beta:
                    return maxAction
        
        return maxAction

    def MIN_VALUE(self, board, alpha, beta, depth):
        if(self.game_completed(board) == self.player_number or depth == 1):
            return self.evaluation_function(board)*depth

        minAction = 10000
        for i in range(3, 10):
            newBoard = board.copy()
            didBoardUpdate = self.update_board(newBoard, i%7, self.findOtherPlayerNumber())
            if(didBoardUpdate):
                action = self.MAX_VALUE(newBoard, alpha, beta, depth - 1)
                
                minAction = min(action, minAction)
                beta = min(beta, minAction)
                if alpha >= beta:
                    return minAction
        return minAction
        
    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        evaluation = -10000
        action = 0
        for i in range (3, 10):
            b = board.copy()
            #print(i&7)
            if(self.update_board(b, i%7, self.player_number)):
                newEval = self.MIN_VALUE(b, -10000, 10000, 5)
                #print("Action: " + str(i%7) + " Eval: " + str(newEval))
                if(newEval > evaluation):
                    evaluation = newEval
                    action = i%7

        return action
        #raise NotImplementedError('Whoops I don\'t know what to do')

    def MAX_EXP_VALUE(self, b, alpha, beta, depth):
        board = b.copy()

        if(self.game_completed(board) == self.findOtherPlayerNumber() or depth ==  1):
            return self.evaluation_function(board)*depth
        
        maxAction = -10000
        for i in range(3, 10):
            didBoardUpdate = self.update_board(board, i%7, self.player_number)
            if(didBoardUpdate):
                action = self.MIN_EXP_VALUE(board, alpha, beta, depth - 1)
                
                maxAction = max(action, maxAction)
        
        return maxAction

    def MIN_EXP_VALUE(self, board, alpha, beta, depth):
        newBoard = board.copy()

        if(self.game_completed(board) == self.player_number or depth == 1):
            return self.evaluation_function(board)*depth

        expect = -10000
        for i in range(3, 10):
            didBoardUpdate = self.update_board(newBoard, i%7, self.findOtherPlayerNumber())
            if(didBoardUpdate):
                expect = self.MAX_EXP_VALUE(newBoard, alpha, beta, depth - 1)*(1/7)

        return expect

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        evaluation = -10000
        action = 0
        for i in range (3, 10):
            b = board.copy()
            if(self.update_board(b, i%7, self.player_number)):
                newEval = self.MIN_EXP_VALUE(b, -10000, 10000, 3)
                #print("Action: " + str(i%7) + " Eval: " + str(newEval))
                if(newEval > evaluation):
                    evaluation = newEval
                    action = i%7

        return action

    def checkWinScore(self, board):
        self_right_player_3_win_str = '{0}{0}{0}'.format(self.player_number) + '{0}'.format(0)
        self_right_player_2_win_str = '{0}{0}'.format(self.player_number) + '{0}{0}'.format(0)
        other_right_player_3_win_str = '{0}{0}{0}'.format(self.findOtherPlayerNumber()) + '{0}'.format(0)
        self_left_player_3_win_str = '{0}'.format(0) + '{0}{0}{0}'.format(self.player_number)
        self_left_player_2_win_str = '{0}{0}'.format(0) + '{0}{0}'.format(self.player_number)
        other_left_player_3_win_str =  '{0}'.format(0) + '{0}{0}{0}'.format(self.findOtherPlayerNumber())
        
        iGotTwoInARow = 100
        iGotThreeInARow = 500
        theyHaveThreeInARow = -1000

        to_str = lambda a: ''.join(a.astype(str))

        totalScore = 0

        def check_horizontal(b):
            for row in b:
                if self_right_player_2_win_str in to_str(row):
                    totalScore += iGotTwoInARow
                elif self_right_player_3_win_str in to_str(row):
                    totalScore += iGotThreeInARow
                elif other_right_player_3_win_str in to_str(row):
                    totalScore += theyHaveThreeInARow
                elif self_left_player_2_win_str in to_str(row):
                    totalScore += iGotTwoInARow
                elif self_left_player_3_win_str in to_str(row):
                    totalScore += iGotThreeInARow
                elif other_left_player_3_win_str in to_str(row):
                    totalScore += theyHaveThreeInARow
            totalScore += 0

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if self_right_player_2_win_str in to_str(row):
                    totalScore += iGotTwoInARow
                elif self_right_player_3_win_str in to_str(row):
                    totalScore += iGotThreeInARow
                elif other_right_player_3_win_str in to_str(row):
                    totalScore += theyHaveThreeInARow
                elif self_left_player_2_win_str in to_str(row):
                    totalScore += iGotTwoInARow
                elif self_left_player_3_win_str in to_str(row):
                    totalScore += iGotThreeInARow
                elif other_left_player_3_win_str in to_str(row):
                    totalScore += theyHaveThreeInARow

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if self_right_player_2_win_str in to_str(row):
                            totalScore += iGotTwoInARow
                        elif self_right_player_3_win_str in to_str(row):
                            totalScore += iGotThreeInARow
                        elif other_right_player_3_win_str in to_str(row):
                            totalScore += theyHaveThreeInARow
                        elif self_left_player_2_win_str in to_str(row):
                            totalScore += iGotTwoInARow
                        elif self_left_player_3_win_str in to_str(row):
                            totalScore += iGotThreeInARow
                        elif other_left_player_3_win_str in to_str(row):
                            totalScore += theyHaveThreeInARow

            totalScore += 0
        
        return totalScore

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        winScore = 0

        if(self.game_completed(board) == self.player_number):
            winScore += 1000
        elif(self.game_completed(board) ==  self.findOtherPlayerNumber()):
            winScore -= 1000
        else:
            #print("Nope")
            winScore += self.checkWinScore(board)

        return winScore


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

