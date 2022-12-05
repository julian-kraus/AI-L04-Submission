# myTeam.py
# ---------
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
import math
import random
import pickle
from FeatureExtractor import FeatureExtractor

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from os.path import exists



##########
# Agents #
##########


#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='QAttacker', second='QAttacker2', num_training=0):
    """This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest."""

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class QLearningAgent(CaptureAgent):

    def register_initial_state(self, state):

        self.epsilon = 0.1
        self.discountFactor = 0.8
        self.learningRate = 0.2
        self.training = False
        if exists(self.path):
            file = open(self.path, 'rb')
            self.weights = pickle.load(file)
            file.close()

        self.amountOfOpponents = len(self.get_opponents(state))


        CaptureAgent.register_initial_state(self, state)

    def getQValue(self, features):
        return features * self.weights

    def getHightestQValueFromState(self, state):
        legalActions = state.get_legal_actions(self.index)
        if not legalActions:
            return 0.0
        return max([self.getQValue(self.features.get(state, a)) for a in legalActions])

    def computeBestActionFromQValues(self, state):
        if not state.get_legal_actions(self.index):
            return None
        else:
            legalActions = state.get_legal_actions(self.index)
            c = util.Counter()
            for action in legalActions:
                c[action] = self.getQValue(self.features.get(state, action))
            return c.argMax()

    def get_action(self, state):
        if util.flipCoin(self.epsilon):
            return random.choice(state.get_legal_actions(self.index))
        else:
            self.is_pacman = state.get_agent_state(self.index).is_pacman
            bestAction = self.computeBestActionFromQValues(state)
            self.features.current = self.features.get(state, bestAction)
            if self.training:
                successor = self.getSuccessor(state, bestAction)
                self.update(successor, self.getReward(state, successor))
            return bestAction

    def getSuccessor(self, state, action):
        successor = state.generate_successor(self.index, action)
        successorPosition = successor.get_agent_state(self.index).get_position()
        if successorPosition != util.nearestPoint(successorPosition):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def update(self, successor, reward):
        features = self.features.current
        qValue = self.getQValue(features)
        maxQ = self.getHightestQValueFromState(successor)
        difference = (reward + self.discountFactor * maxQ) - qValue
        for feature in features:
            featureVal = features[feature]
            addedWeigth = self.learningRate * difference * features[feature]
            self.weights[feature] += addedWeigth

    def final(self, state):
        print(self.weights)
        if self.training:
            file = open(self.path, 'wb')
            pickle.dump(self.weights, file)
            file.close()
        CaptureAgent.final(self, state)

class QAttacker(QLearningAgent):

    def register_initial_state(self, state):

        self.weights = {
            "minEnemyDis": -10,
            "aboutToGetEaten": -100,
            "victimDis": +5,
            "aboutToKill": 30,
            "lowestFoodDistanceInArea": -20,
            #"amountOfPosActions": 1.5,
            "amountOfFoodEaten": 0.25,
            #"back": -5,
            "timeToGoBack": -8,
            "eatCapsule": 3,
            "Stop": -1000
        }
        self.path = "agents/KaCleJu/weights_attack_" + str(self.index) + ".pkl"

        self.start = state.get_agent_position(self.index)
        self.foodEaten = 0
        grid = state.get_walls()
        self.width = grid.width
        self.height = grid.height
        self.maxDis = self.width * self.height
        if self.red:
            xArea = (self.width / 2) + (self.height / 4)
            yArea = (self.height / 2) + (self.height / 4)
        else:
            xArea = (self.width / 2) - (self.height / 4)
            yArea = (self.height / 2) + (self.height / 4)
        self.area = (xArea, yArea)

        self.features = FeatureExtractor(self, state)

        QLearningAgent.register_initial_state(self, state)

    """def getFeatures(self, state, action):
        successor = self.getSuccessor(state, action)
        features = util.Counter()
        features["aboutToGetEaten"] = self.aboutToGetEaten(successor)
        features["lowestFoodDistance"] = self.getLowestFoodDistanceFeature(state, successor, False)
       # features["aboutToEat"] = self.getAboutToEatFeature(state, successor, False)
        features["timeToGoBack"] = self.computeTimeToGoBackFeature(state, successor)
        dis = self.getClostestEnemies(successor, 14, False)
        if dis == []:
            min_dis = 1
        else:
            min_dis = min(dis)/self.maxDis
        features["ghostNear"] = min_dis
        return features"""

    def getReward(self, state, successor):
        reward = -1

        # Score reward
        newScore = self.getScore(successor)
        oldScore = self.getScore(state)
        if newScore != oldScore:
            if self.red and newScore > oldScore:
                reward += abs(self.get_score(successor) - self.get_score(state) * 10)
            elif not self.red and newScore < oldScore:
                reward += abs((self.get_score(successor) - self.get_score(state)) * 10)


        # Dead reward
        #enemy_dis = self.getClostestEnemies(state, 2, False)
        #if not enemy_dis is None and not enemy_dis == []:
        #    nextPos = successor.get_agent_state(self.index).get_position()
        #    if nextPos == self.start:
        #        reward -= 100

        # Food reward
        if self.features.getNumFoodLeft(state, False) < self.features.getNumFoodLeft(successor, False):
            reward += 2
        if self.features.getAboutToEatCapsule(state, successor):
            reward += 3

        #if self.features.getClosestEnemiesDistances(state, self.is_pacman) and successor.get_agent_position(self.index) == self.start:
        #    reward -= 100
        succVictims = self.features.getClosestVictimDistances(successor, self.is_pacman)
        currVictims = [d for d in self.features.getClosestVictimDistances(state, self.is_pacman) if d < 3]
        if succVictims < currVictims:
            reward += 5



        return reward

    def getScore(self, state):
        return state.get_score()

class QAttacker2(QLearningAgent):

    def register_initial_state(self, state):

        self.weights = {
            "minEnemyDis": 10,
            "aboutToGetEaten": -10,
            "victimDis": -5,
            "aboutToKill": 5,
            "lowestFoodDistanceInArea": -20,
            #"amountOfPosActions": 0.25,
            "amountOfFoodEaten": 2,
            #"back": -5,
            "timeToGoBack": -8,
            "eatCapsule": 3,
            "Stop": -1000
        }
        self.path = "agents/KaCleJu/weights_attack_" + str(self.index) + ".pkl"

        self.start = state.get_agent_position(self.index)
        self.foodEaten = 0
        grid = state.get_walls()
        self.width = grid.width
        self.height = grid.height
        #self.width * self.height
        if self.red:
            xArea = (self.width / 2) + (self.height / 4)
            yArea = (self.height / 2) - (self.height / 4)
        else:
            xArea = (self.width / 2) - (self.height / 4)
            yArea = (self.height / 2) - (self.height / 4)
        self.area = (xArea, yArea)

        self.features = FeatureExtractor(self, state)

        QLearningAgent.register_initial_state(self, state)

    """def getFeatures(self, state, action):
        successor = self.getSuccessor(state, action)
        features = util.Counter()
        features["aboutToGetEaten"] = self.aboutToGetEaten(successor)
        features["lowestFoodDistance"] = self.getLowestFoodDistanceFeature(state, successor, False)
       # features["aboutToEat"] = self.getAboutToEatFeature(state, successor, False)
        features["timeToGoBack"] = self.computeTimeToGoBackFeature(state, successor)
        dis = self.getClostestEnemies(successor, 14, False)
        if dis == []:
            min_dis = 1
        else:
            min_dis = min(dis)/self.maxDis
        features["ghostNear"] = min_dis
        return features"""

    def getReward(self, state, successor):

        reward = -1

        # Score reward
        newScore = self.getScore(successor)
        oldScore = self.getScore(state)
        if newScore != oldScore:
            if self.red and newScore > oldScore:
                reward += abs(self.get_score(successor) - self.get_score(state) * 10)
            elif not self.red and newScore < oldScore:
                reward += abs((self.get_score(successor) - self.get_score(state)) * 10)

        # Dead reward
        # enemy_dis = self.getClostestEnemies(state, 2, False)
        # if not enemy_dis is None and not enemy_dis == []:
        #    nextPos = successor.get_agent_state(self.index).get_position()
        #    if nextPos == self.start:
        #        reward -= 100

        # Food reward
        if self.features.getNumFoodLeft(state, False) < self.features.getNumFoodLeft(successor, False):
            reward += 2
        if self.features.getAboutToEatCapsule(state, successor):
            reward += 3

        # if self.features.getClosestEnemiesDistances(state, self.is_pacman) and successor.get_agent_position(self.index) == self.start:
        #    reward -= 100
        succVictims = self.features.getClosestVictimDistances(successor, self.is_pacman)
        currVictims = [d for d in self.features.getClosestVictimDistances(state, self.is_pacman) if d < 3]
        if succVictims < currVictims:
            reward += 5

        return reward

    def getScore(self, state):
        return state.get_score()

class DummyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def register_initial_state(self, game_state):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.register_initial_state in captureAgents.py.
        '''
        CaptureAgent.register_initial_state(self, game_state)

        '''
        Your initialization code goes here, if you need any.
        '''

    def choose_action(self, game_state):
        """
        Picks among actions randomly.
        """
        actions = game_state.get_legal_actions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)


"""class QDefender(QLearningAgent):

    def register_initial_state(self, state):
        self.path = "agents/KaCleJu/weights_defend_" + str(self.index) + ".pkl"
        self.weights = {
            "lowestPacmanDistance": -100,
            "averageFoodDistance": -40,
            "distanceToMiddle": 0,
            "defending": 5,
            "scared": -10
        }
        self.currentClosest = []
        QLearningAgent.register_initial_state(self, state)

    def getFeatures(self, state, action):
        successor = self.getSuccessor(state, action)
        features = util.Counter()
        features["lowestPacmanDistance"] = self.getLowestPacmanDistanceFeature(successor)
        features["averageFoodDistance"] = self.getAverageFoodDistanceFeature(successor, True)
        features["distanceToMiddle"] = self.getDistanceToMiddleFeature(successor)
        features["defending"] = self.getIsDefendingFeature(successor)
        features["scared"] = self.getScaredFeature(state)
        return features

    def getLowestPacmanDistanceFeature(self, state):
        closestPacmans = self.getClostestEnemies(state, 20, True)
        if self.currentClosest > closestPacmans:
            return 0
        if closestPacmans:
            return min(closestPacmans) / self.maxDis
        else:
            return 1

    def getScaredFeature(self, state):
        st = state.get_agent_state(self.index)
        if state.get_agent_state(self.index).scared_timer != 0:
            return 1
        return 0
    def getDistanceToMiddleFeature(self, state):
        return self.getDistanceToMiddle(state) / self.maxDis

    def getDistanceToMiddle(self, state):
        pos = state.get_agent_position(self.index)
        grid = state.get_walls()
        width = grid.width
        height = grid.height
        if self.red:
            goal = (round(width / 2) - 1, round(height / 2))
        else:
            goal = (round(width / 2) + 1, round(height / 2))
        return util.manhattanDistance(pos, goal)

    def getIsDefendingFeature(self, state):
        return self.isDefending(state)

    def isDefending(self, state):
        if not state.get_agent_state(self.index).is_pacman:
            return 1
        return 0

    def getReward(self, state, successor):
        reward = -15

        # Score reward
        newScore = self.getScore(successor)
        oldScore = self.getScore(state)
        if newScore != oldScore:
            if (self.red and newScore < oldScore) or (not self.red and newScore > oldScore):
                reward -= abs(self.get_score(successor) - self.get_score(state) * 10)

        # Kill reward
        future_closest = self.getClostestEnemies(successor, 5, True)

        #enemies = [enemy for enemy in self.get_opponents(state) if state.get_agent_state(enemy).get_position() != None and state.get_agent_state(enemy).isPacman]
        #newEnemies = [enemy for enemy in enemies if not successor.getAgentState(enemy).getPosition() is None]
        if len(self.currentClosest) > len(future_closest):
            reward += 100
        self.currentClosest = self.getClostestEnemies(successor, 3, True)
        # Not defending
        #if self.isDefending(successor) == 0:
        #    reward -= 5

        #reward -= self.getAverageFoodDistance(successor, True)

        return reward"""
