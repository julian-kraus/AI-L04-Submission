import math
import util
from sklearn import preprocessing

import numpy as np


class FeatureExtractor:
    def __init__(self, agent, state):
        self.agent = agent
        self.currentStateId = 0
        self.current = None
        self.totalFood = self.getNumFoodLeft(state, False)
        self.maxDis = 76

    def get(self, state, action):
        features = util.Counter()
        successor = self.agent.getSuccessor(state, action)
        enemyDis = self.getClosestEnemiesDistances(successor, self.agent.is_pacman)
        if not enemyDis:
            minEnemyDisFeature = 10
        else:
            minEnemyDisFeature = min(enemyDis)
        # The bigger the better
        features["minEnemyDis"] = 1 - (minEnemyDisFeature / 10)
        # 0 for no close, otherwise amount of close enemies -> smaller the better
        features["aboutToGetEaten"] = self.aboutToGetEaten(state, successor, self.agent.is_pacman)
        victimDis = self.getClosestVictimDistances(successor, self.agent.is_pacman)
        if not victimDis:
            minVictimDisFeature = 10
        else:
            minVictimDisFeature = min(victimDis)
        # The smaller the better
        features["victimDis"] = 1 - (minVictimDisFeature / 10)
        # 0 for no close, otherwise amount of close enemies -> bigger the better
        features["aboutToKill"] = self.aboutToKill(state, successor, self.agent.is_pacman) / 2
        # The lower the better
        features["lowestFoodDistanceInArea"] = self.getLowestFoodDistanceInArea(state, successor, False) / self.maxDis
        # The higher the better
        #features["amountOfPosActions"] = len(successor.get_legal_actions(self.agent.index)) / 6

        if self.getNumFoodLeft(state, False) > self.getNumFoodLeft(successor, False):
            self.agent.foodEaten += 1

        if self.agent.red:
            half = (self.agent.width / 2) - 1
            if state.get_agent_position(self.agent.index)[0] < half:
                self.agent.foodEaten = 0
        else:
            half = (self.agent.width / 2) + 1
            if state.get_agent_position(self.agent.index)[0] > half:
                self.agent.foodEaten = 0

        features["amountOfFoodEaten"] = self.agent.foodEaten / self.totalFood
        #features["back"] = self.agent.get_maze_distance(successor.get_agent_position(self.agent.index), self.agent.start) * self.agent.weights["amountOfFoodEaten"] * self.agent.foodEaten
        features["timeToGoBack"] = self.computeTimeToGoBackFeature(state, successor) / self.maxDis
        features["eatCapsule"] = self.getAboutToEatCapsule(state, successor) / 2
        if action == "Stop":
            stop = 1
        else:
            stop = 0
        features["Stop"] = stop / 2
        #data = np.array([val for val in features.values()])
        #data_norm = list(data / self.maxDis)
        #i = 0
        #for value in features.keys():
        #    features[value] = data_norm[i]
        #    i += 1

        return features

    def getFood(self, state, myTeam):
        if self.agent.red:
            if myTeam:
                return state.get_red_food()
            else:
                return state.get_blue_food()
        else:
            if myTeam:
                return state.get_blue_food()
            else:
                return state.get_red_food()

    def getLowestFoodDistanceInArea(self, state, successor, myTeam):
        agent_pos = successor.get_agent_position(self.agent.index)
        dis = self.getLowestFoodDistance(state, successor, myTeam)
        if dis == 0:
            return 0
        return dis + (util.manhattanDistance(agent_pos, self.agent.area) / 4)

    def getLowestFoodDistance(self, state, successor, myTeam):
        if self.getNumFoodLeft(successor, myTeam) == 0:
            return 0
        if self.getNumFoodLeft(state, myTeam) > self.getNumFoodLeft(successor, myTeam):
            return 0
        foodMap = self.getFood(successor, myTeam)
        min_food_dis = math.inf
        pos = successor.get_agent_position(self.agent.index)
        for x, y in foodMap.as_list():
            dis = self.agent.get_maze_distance(pos, (x, y))
            min_food_dis = min(min_food_dis, dis)
        return min_food_dis

    def getAverageFoodDistanceFeature(self, state, myTeam):
        return self.getAverageFoodDistance(state, myTeam) / self.maxDis

    def getAverageFoodDistance(self, state, myTeam):
        num_food = self.getNumFoodLeft(state, myTeam)
        pos = state.get_agent_position(self.agent.index)
        if num_food == 0:
            return 0
        foodMap = self.getFood(state, myTeam)
        food_dis = 0
        for x, y in foodMap.as_list():
            food_dis += util.manhattanDistance(pos, (x, y))
        return food_dis / num_food

    def getAboutToEatFeature(self, state, successor, myTeam):
        if self.getNumFoodLeft(state, myTeam) > self.getNumFoodLeft(successor, myTeam) \
                | len(state.get_capsules()) > len(successor.get_capsules()):
            return 1
        return 0

    def getNumFoodLeft(self, state, myTeam):
        return len(self.getFood(state, myTeam).as_list())

    def getAboutToEatCapsule(self, state, successor):
        if self.agent.red:
            if len(state.get_blue_capsules()) > len(successor.get_blue_capsules()):
                return 1
            return 0
        else:
            if len(state.get_red_capsules()) > len(successor.get_red_capsules()):
                return 1
            return 0

    def computeTimeToGoBackFeature(self, state, successor):
        if self.agent.foodEaten < 5:
            return 0
        val = self.computeTimeToGoBack(state, successor)
        if val == 0:
            return 0
        l = len(str(val))

        return val

    def computeTimeToGoBack(self, state, successor):
        grid = state.get_walls()
        width = grid.width
        if self.agent.red:
            half = (width / 2) - 1
            if state.get_agent_position(self.agent.index)[0] < half:
                self.agent.foodEaten = 0
        else:
            half = (width / 2) + 1
            if state.get_agent_position(self.agent.index)[0] > half:
                self.agent.foodEaten = 0
        return self.agent.get_maze_distance(self.agent.start, successor.get_agent_position(self.agent.index)) #self.totalFood - self.getNumFoodLeft(successor, False)) *

    def getFeatureVal(self, val, num):
        strVal = str(round(val))
        while len(strVal) < num:
            strVal = "0" + strVal
        return float("0." + strVal)

    def aboutToGetEaten(self, state, successor, pacman):
        #agentPos = successor.get_agent_position(self.agent.index)
        #if self.getClosestEnemiesDistances(state, pacman) and  util.manhattanDistance(agentPos, self.agent.start) < 4:
        #    return 1
        #return 0
        succEnemies = self.getClosestVictimDistances(successor, pacman)
        currEnemies = [d for d in self.getClosestVictimDistances(state, pacman) if d < 3]
        if succEnemies < currEnemies:
            return 1
        return 0

    def getClostestEnemies(self, state, pacman):
        if pacman:
            # Get all not scared Ghosts
            return [
                state.get_agent_position(index)
                               for index in self.agent.get_opponents(state)
                               if not state.get_agent_position(index) is None
                               and (
                                   not state.get_agent_state(index).is_pacman
                                   and state.get_agent_state(index).scared_timer <= 3
                                  )]
        else:
            # Get all pacmans if scared
            if state.get_agent_state(self.agent.index).scared_timer > 0:
                return [
                    state.get_agent_position(index)
                    for index in self.agent.get_opponents(state)
                    if not state.get_agent_position(index) is None
                       and (
                               state.get_agent_state(index).is_pacman
                       )]
        return []

    def getClosestEnemiesDistances(self, state, pacman):
        enemy_positions = self.getClostestEnemies(state, pacman)
        agent_pos = state.get_agent_position(self.agent.index)
        dis = [self.agent.get_maze_distance(enemy_pos, agent_pos) for enemy_pos in enemy_positions]
        return [d for d in dis if d <= 10]

    def aboutToKill(self, state, successor, pacman):
        #closestVictims = self.getClosestVictimDistances(state, pacman)
        #if closestVictims and min(closestVictims) <= 2:
        #    return len(closestVictims) / self.agent.amountOfOpponents
        #else:
        #    return 0
        succVictims = self.getClosestVictimDistances(successor, pacman)
        currVictims = [d for d in self.getClosestVictimDistances(state, pacman) if d < 3]
        if succVictims < currVictims:
            return 1
        return 0

    def getClosestVictims(self, state, pacman):
        if pacman:
            # Get all scared Ghosts
            return [
                state.get_agent_position(index)
                for index in self.agent.get_opponents(state)
                if not state.get_agent_position(index) is None
                   and (
                           not state.get_agent_state(index).is_pacman
                           or state.get_agent_state(index).scared_timer > 3
                   )]
        else:
            # Get all pacmans if scared
            if state.get_agent_state(self.agent.index).scared_timer == 0:
                return [
                    state.get_agent_position(index)
                    for index in self.agent.get_opponents(state)
                    if not state.get_agent_position(index) is None
                       and (
                           state.get_agent_state(index).is_pacman
                       )]
        return []

    def getClosestVictimDistances(self, state, pacman):
        victim_positions = self.getClosestVictims(state, pacman)
        agent_pos = state.get_agent_position(self.agent.index)
        return [self.agent.get_maze_distance(victim_pos, agent_pos) for victim_pos in victim_positions]

