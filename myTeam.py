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
                first='Sergio', second='Andersson', num_training=0):
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
        self.foodEaten = 0
        grid = state.get_walls()
        self.width = grid.width
        self.height = grid.height

        self.features = FeatureExtractor(self, state)
        self.weights = {"minEnemyDis": -92.123124592,
                        "victimDis": 23.242351351,
                        "aboutToGetEaten": -232.2135821,
                        "lowestFoodDistanceInArea": -19.9845321,
                        "amountOfPosActions": 8.12351,
                        "timeToGoBack": 72.1231491,
                        "eatCapsule": 32.1234961,
                        "oneWay": -98.18321471,
                        "Stop": -1000,
                        }

        self.init = False
        self.path = "agents/KaCleJu/weights_attack_" + str(self.index) + ".pkl"
        self.epsilon = 0
        self.discountFactor = 0.8
        self.learningRate = 0.1
        self.training = False
        if exists(self.path):
            file = open(self.path, 'rb')
            self.weights = pickle.load(file)
            file.close()

        self.amountOfOpponents = len(self.get_opponents(state))

        CaptureAgent.register_initial_state(self, state)

    def getQValue(self, features):
        return features * self.weights

    def getHighestQValueFromState(self, state):
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
        if not self.init and not self.red:
            x, y = self.areaPos
            self.areaPos = (x - 21, y)
            self.init = True

        self.is_pacman = state.get_agent_state(self.index).is_pacman
        if not self.is_pacman:
            self.foodEaten = 0

        if util.flipCoin(self.epsilon):
            return random.choice(state.get_legal_actions(self.index))
        else:
            bestAction = self.computeBestActionFromQValues(state)
            self.features.current = self.features.get(state, bestAction)
            successor = self.getSuccessor(state, bestAction)
            if self.features.getLowestFoodDistance(state, successor, False) == 0:
                self.foodEaten += 1
            if self.training:
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
        maxQ = self.getHighestQValueFromState(successor)
        difference = (reward + self.discountFactor * maxQ) - qValue
        for feature in features:
            addedWeight = self.learningRate * difference * features[feature]
            self.weights[feature] += addedWeight

    def getReward(self, state, successor):

        reward = 0

        # Score reward
        newScore = self.getScore(successor)
        oldScore = self.getScore(state)
        if newScore != oldScore:
            if self.red and newScore > oldScore:
                reward += abs(self.get_score(successor) - self.get_score(state) * 10)
            elif not self.red and newScore < oldScore:
                reward += abs((self.get_score(successor) - self.get_score(state)) * 10)

        # Dead reward
        succEnemies = self.features.getClosestEnemiesDistances(successor, self.is_pacman)
        currEnemies = [d for d in self.features.getClosestEnemiesDistances(state, self.is_pacman) if d < 3]
        if succEnemies < currEnemies:
            reward -= 100
        if not currEnemies is None and not currEnemies == []:
            nextPos = successor.get_agent_state(self.index).get_position()
            if nextPos == self.start:
                reward -= 100
            elif min(succEnemies) > 10:
                reward -= 100

        # Food reward
        if self.features.getNumFoodLeft(state, False) < self.features.getNumFoodLeft(successor, False):
            reward += 5
        if self.features.getAboutToEatCapsule(state, successor):
            reward += 7

        succVictims = self.features.getClosestVictimDistances(successor, self.is_pacman)
        currVictims = [d for d in self.features.getClosestVictimDistances(state, self.is_pacman) if d < 3]
        if succVictims < currVictims:
            reward += 9

        return reward

    def final(self, state):
        print(self.weights)
        if self.training:
            file = open(self.path, 'wb')
            pickle.dump(self.weights, file)
            file.close()
        CaptureAgent.final(self, state)

    def getScore(self, state):
        return state.get_score()


class Sergio(QLearningAgent):

    def register_initial_state(self, state):
        self.areaPos = (26, 14)

        self.start = state.get_agent_position(self.index)

        QLearningAgent.register_initial_state(self, state)


class Andersson(QLearningAgent):

    def register_initial_state(self, state):
        self.areaPos = (26, 1)

        self.start = state.get_agent_position(self.index)

        QLearningAgent.register_initial_state(self, state)
