# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState, funcEval = scoreEvaluationFunction):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        infinity = float('inf')

        def max_value(gameState, depth, numGhosts):
            """For the Max Player here Pacman. Por eso en getLegalActions, siempre va con CERO"""
            #Si se ha ganado, se ha perdido, si ya no hay más profundidades permitidas para buscar, o si pacman no
            #tiene acciones legales, retornar el valor de funcEval del gameState
            if gameState.isWin() or gameState.isLose() or depth == 0 or (len(gameState.getLegalActions(0)) == 0):
                return funcEval(gameState)
            v = -infinity
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                v = max(v, min_value(gameState.generateSuccessor(0, action), depth - 1, 1, numGhosts))
            return v
        
        def min_value(gameState, depth, agentIndex, numGhosts):
            """For the MIN Players or Agents"""
            # Check that successors exist
            if gameState.isWin() or gameState.isLose() or depth==0: #si es un estado terminal
                return funcEval(gameState)
                #Quiero el mínimo de: TODOS los ghosts, y el maxvalue (del pacman)
                #el poner agentIndex == numGhosts me asegura que compararé con tooodos los ghosts
                #antes de considerar al pacman
            v=-infinity
            legalActions = gameState.getLegalActions(agentIndex)
            if agentIndex == numGhosts:
                for action in legalActions:
                    v = min(v, max_value(gameState.generateSuccessor(agentIndex, action), depth - 1, numGhosts))
            else:
                for action in legalActions:
                    v = min(v, min_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, numGhosts))
            return v

        legalActions = gameState.getLegalActions(0)
        numGhosts = gameState.getNumAgents() - 1
        bestaction = Directions.STOP
        score = -infinity
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevscore = score
            score = max(score, min_value(nextState, self.depth, 1, numGhosts))
            if score > prevscore:
                bestaction = action
        return bestaction

        # Copy current state
        #state = GameState(self)   

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    infinity = float('inf')
    def maxvalue(gameState, alpha, beta, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -infinity
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            v = max(v, minvalue(nextState, alpha, beta, gameState.getNumAgents() - 1, depth))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def minvalue(gameState, alpha, beta, agentindex, depth):
        numghosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = infinity
        legalActions = gameState.getLegalActions(agentindex)
        for action in legalActions:
            nextState = gameState.generateSuccessor(agentindex, action)
            if agentindex == numghosts:
                v = min(v, maxvalue(nextState, alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            else:
                v = min(v, minvalue(nextState, alpha, beta, agentindex + 1, depth))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v
    
    

    legalActions = gameState.getLegalActions(0)
    bestaction = Directions.STOP
    score = -infinity
    alpha = -infinity
    beta  = +infinity
    for action in legalActions:
        nextState = gameState.generateSuccessor(0, action)
        prevscore = score
        score = max(score, minvalue(nextState, alpha, beta, 1, self.depth))
        if score > prevscore:
            bestaction = action
        if score >= beta:
            return bestaction
        alpha = max(alpha, score)
    return bestaction
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
