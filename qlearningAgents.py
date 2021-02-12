# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.Qvalues = util.Counter()




  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    return(self.Qvalues[(state,action)])

    util.raiseNotDefined()


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    legalActions = self.getLegalActions(state)

    if len(legalActions) == 0:  # if terminal state
        return 0.0

    qvals = [self.getQValue(state, action) for action in legalActions]
    return(max(qvals))

    util.raiseNotDefined()

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = self.getLegalActions(state)

    if len(legalActions) == 0:
      return None

    qvals = [self.getQValue(state, action) for action in legalActions]
    maxQval = max(qvals)
    maxActions = [action for index, action in enumerate(legalActions) if qvals[index] == maxQval]
    return(random.choice(maxActions)) #makes sure the choice is random if there are multiple actions w/ same qval


    util.raiseNotDefined()

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return None

    bestAction = self.getPolicy(state)
    takeRandomAction = util.flipCoin(self.epsilon)
    if takeRandomAction:
      return(random.choice(legalActions))
    else:
      return(bestAction)


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """

    learningRate = self.alpha
    discount = self.discount
    beforeQval = self.getQValue(state, action)
    sample = reward + discount*self.getValue(nextState)
    updateQval = (1-learningRate)*beforeQval + learningRate*sample
    self.Qvalues[(state,action)] = updateQval




class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    features = self.featExtractor.getFeatures(state,action)
    return sum([self.weights[feature]*featureValue for feature,featureValue in features.items()])

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """

    features = self.featExtractor.getFeatures(state,action)
    successorActions = self.getLegalActions(nextState)
    successorQvalues = [self.getQValue(nextState,successorAction) for successorAction in successorActions]

    if len(successorQvalues) == 0:
      successorValue = 0
    else:
      successorValue = max(successorQvalues)

    correction = (reward + self.discount*successorValue) - self.getQValue(state, action)
    for feature,featureVal in features.items():
      oldWeight = self.weights[feature]
      weightAdjustment = self.alpha*correction*featureVal
      self.weights[feature] = oldWeight + weightAdjustment

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
