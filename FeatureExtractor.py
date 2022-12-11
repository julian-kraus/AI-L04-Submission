import math
import util
from game import Actions

oneWayPositions = [(5, 1), (7, 3), (10, 3), (9, 5), (3, 6), (10, 7), (6, 9), (10, 9), (14, 9), (8, 11), (5, 12),
                   (6, 14), (8, 14), (10, 14), (3, 8), (26, 14), (24, 12), (21, 12), (22, 10), (28, 9), (21, 8),
                   (25, 6),
                   (21, 6), (16, 6), (23, 4), (26, 3), (25, 1), (23, 1), (21, 1), (28, 7)]


class FeatureExtractor:
    def __init__(self, agent, state):
        self.agent = agent
        self.current = None
        self.totalFood = self.getNumFoodLeft(state, False)
        self.maxDis = 76
        self.goBackNum = 6

    def get(self, state, action):
        features = util.Counter()
        successor = self.agent.getSuccessor(state, action)
        self.is_pacman = successor.get_agent_state(self.agent.index).is_pacman
        (xC, yC) = state.get_agent_position(self.agent.index)
        self.succPos = successor.get_agent_position(self.agent.index)
        enemyDis = [dis for dis in self.getClosestEnemiesDistances(successor, self.agent.is_pacman) if dis < 8]
        if enemyDis == []:
            minEnemyDisFeature = self.maxDis
            self.chased = False
        else:
            minEnemyDisFeature = min(enemyDis)
            self.chased = True

        if self.chased:
            self.goBackNum = 0
            features["lowestFoodDistanceInArea"] = 1
            features["amountOfPosActions"] = len(successor.get_legal_actions(self.agent.index)) / 6
            if self.succPos in oneWayPositions and len(successor.get_legal_actions(self.agent.index)) > 1:
                features["oneWay"] = 1
            if self.succPos in oneWayPositions and len(successor.get_legal_actions(self.agent.index)) == 2:
                features["oneWay"] = 0.5
            else:
                features["oneWay"] = 0
        else:
            self.goBackNum = 6
            features["lowestFoodDistanceInArea"] = self.getLowestFoodDistanceInArea(state, successor,
                                                                                    False) / self.maxDis
            features["amountOfPosActions"] = 0
            features["oneWay"] = 0

        features["aboutToGetEaten"] = self.aboutToGetEaten(state, successor, self.is_pacman)
        features["timeToGoBack"] = (self.maxDis - self.computeTimeToGoBackFeature(state, successor)) / self.maxDis
        features["minEnemyDis"] = (self.maxDis - minEnemyDisFeature) / self.maxDis
        features["eatCapsule"] = self.getAboutToEatCapsule(state, successor) / 2

        victimDis = self.getClosestVictimDistances(state, successor, self.is_pacman)
        if not victimDis:
            minVictimDisFeature = self.maxDis
        else:
            minVictimDisFeature = min(victimDis)
        features["victimDis"] = (self.maxDis - minVictimDisFeature) / self.maxDis
        features["timeToGoBack"] = (self.maxDis - self.computeTimeToGoBackFeature(state, successor)) / self.maxDis
        features["eatCapsule"] = self.getAboutToEatCapsule(state, successor) / 2
        if action == "Stop":
            stop = 1
        else:
            stop = 0
        features["Stop"] = stop / 2

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
        numFood = self.getNumFoodLeft(successor, myTeam)
        x, y = successor.get_agent_position(self.agent.index)
        dis = self.getLowestFoodDistance(state, successor, myTeam)
        if self.getNumFoodLeft(successor, not myTeam) < 9:
            return self.getLowestFoodDistance(state, successor, not myTeam)
        if dis == 0:
            return 0
        if numFood > self.totalFood - 7 and dis > 10:
            return self.agent.get_maze_distance((x, y), self.agent.areaPos) * 1.5
        return dis

    def getLowestFoodDistance(self, state, successor, myTeam):
        if self.getNumFoodLeft(successor, myTeam) == 0 and self.getNumFoodLeft(state, myTeam) == 0:
            return self.maxDis
        ax, ay = successor.get_agent_position(self.agent.index)

        if state.has_food(ax, ay) and (self.is_pacman is None or self.is_pacman):
            return 0
        foodMap = self.getFood(successor, myTeam)
        min_food_dis = math.inf
        for x, y in foodMap.as_list():
            dis = self.agent.get_maze_distance((ax, ay), (x, y))
            min_food_dis = min(min_food_dis, dis)
        return min_food_dis

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
        val = self.computeTimeToGoBack(state, successor)
        if self.agent.foodEaten <= self.goBackNum and self.getNumFoodLeft(successor, False) != 0:
            return self.maxDis
        if val == 0:
            return self.maxDis
        return val

    def aboutToGetEaten(self, state, successor, pacman):
        opponnents = [successor.get_agent_position(e) for e in self.agent.get_opponents(state)]
        walls = successor.get_walls()

        ls = [Actions.get_legal_neighbors(g, walls) for g in opponnents if not g is None]
        if self.succPos in ls:
            return 1
        else:
            return 0

    def computeTimeToGoBack(self, state, successor):
        grid = state.get_walls()
        return self.agent.get_maze_distance(self.agent.start, successor.get_agent_position(
            self.agent.index))

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
        return [self.agent.get_maze_distance(enemy_pos, agent_pos) for enemy_pos in enemy_positions]

    def getClosestVictims(self, state, pacman):
        if pacman:
            # Get all scared Ghosts
            return [
                state.get_agent_position(index)
                for index in self.agent.get_opponents(state)
                if not state.get_agent_position(index) is None
                and (
                       not state.get_agent_state(index).is_pacman
                       and state.get_agent_state(index).scared_timer > 3
                   )]
        else:
            # Get all pacmans if not scared
            if state.get_agent_state(self.agent.index).scared_timer == 0:
                return [
                    state.get_agent_position(index)
                    for index in self.agent.get_opponents(state)
                    if not state.get_agent_position(index) is None
                       and (
                           state.get_agent_state(index).is_pacman
                       )]
        return []

    def getClosestVictimDistances(self, state, successor, pacman):
        agent_pos = successor.get_agent_position(self.agent.index)
        currentPos = [self.agent.get_maze_distance(victim_pos, agent_pos) for victim_pos in
                      self.getClosestVictims(state, pacman)]
        succVictims = [self.agent.get_maze_distance(victim_pos, agent_pos) for victim_pos in
                       self.getClosestVictims(successor, pacman)]
        currVictims = [d for d in currentPos if d < 3]
        if len(succVictims) < len(currVictims):
            return [0]
        return succVictims
