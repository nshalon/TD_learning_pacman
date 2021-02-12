# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            self.iterationStep = iteration
            for state in states:
                self.values[state] = self.getValue(state)

    def getValue(self, state):
        """
      Return the value of the state (computed in __init__).
    """
        if self.mdp.isTerminal(state):
            'Do we need to end the iteration somehow?'
            return 0
            (terminal, prob) = self.mdp.getTransitionStatesAndProbs(state, self.mdp.getPossibleActions(state))
            self.values[state] = self.mdp.getReward(state, exit, terminal)
            return self.values[state]

        actions = self.mdp.getPossibleActions(state)
        'V=maxa(sum[T(s,a)(R(s,a,s\')+(gamma)Vnext(s\')])'
        successorActionValues = []
        for action in actions:
            val = 0
            transitions = self.mdp.getTransitionStatesAndProbs(state, action)
            for (nextState, transitionProb) in transitions:
                val += transitionProb * (
                            self.mdp.getReward(state, action, nextState) + self.discount * (self.values[( nextState, self.iterationStep - 1)]))
            successorActionValues.append(val)

        maxSuccessorVal = max(successorActionValues)

        #change so state val is tied to iteration step, so that an iteration in ply k
        #only accesses values for successor states in ply k-1 and not ply k
        self.values[ ( state, self.iterationStep ) ] = maxSuccessorVal

        return maxSuccessorVal

    def getQValue(self, state, action):
        """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
        if self.mdp.isTerminal(state):
            'Do we need to end the iteration somehow?'
            (terminal, prob) = self.mdp.getTransitionStatesAndProbs(state, action)
            self.values[state] = self.mdp.getReward(state, exit, terminal)
            return self.values[state]

        'V=maxa(sum[T(s,a)(R(s,a,s\')+(gamma)Vnext(s\')])'
        'Q=sum[T(s,a)(R(s,a,s\')+(gamma)Vnext(s\')]'
        qval = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for (nextState, transitionProb) in transitions:
            qval = qval + transitionProb * (
                        self.mdp.getReward(state, action, nextState) + self.discount * (self.getValue(nextState)))

        return qval

    def getPolicy(self, state):
        """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        qvals = [self.getQValue(state, action) for action in actions]
        return actions[ qvals.index( max( qvals ) ) ]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)