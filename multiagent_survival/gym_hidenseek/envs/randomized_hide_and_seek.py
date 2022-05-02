import gym
from gym_hidenseek.modules.grid import *
import gym_hidenseek.utils.graphic_rendering as graphic_rendering

import random

class RandomizedHideAndSeek15x15Env(gym.Env):

    grid = None
    gridSize = 15.0

    window = None

    maxTries = 30

    def __init__(self):

        self.grid = Grid(self.gridSize)

        wall1 = Wall((7.5, 0.0), (7.5, 7.5))
        self.grid.addWallToGrid(wall1, True)
        self.grid.addDoorToWall(wall1, "end", "small")

        wall2 = Wall((7.5, 7.5), (15.0, 7.5))
        self.grid.addWallToGrid(wall2, True)

        self.window = graphic_rendering.RenderingWindow(self.grid)

        self.reset()

    def step(self, agent, action):

        if (action == "forward"):
            actionReturn = agent.translate("forward")
            if ((actionReturn != None) and
                (actionReturn[0] == 4) and                  # If agent touched a box ...
                (actionReturn[1].moveable == True) and      # .. and the box is unlocked ..
                (actionReturn[1].held == False)):           # .. and the agent doesn't hold the box
                moveObjectDueHit(agent, actionReturn[1], actionReturn[2])
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    if ((observedObject.moveable == True) and (observedObject.held == True)):
                        moveTranslateObject(agent, observedObject)

        elif (action == "backward"):
            actionReturn = agent.translate("backward")
            if ((actionReturn != None) and
                (actionReturn[0] == 4) and                  # If agent touched a box ...
                (actionReturn[1].moveable == True) and      # .. and the box is unlocked ..
                (actionReturn[1].held == False)):           # .. and the agent doesn't hold the box
                moveObjectDueHit(agent, actionReturn[1], actionReturn[2])
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    if ((observedObject.moveable == True) and (observedObject.held == True)):
                        moveTranslateObject(agent, observedObject)

        elif (action == "clockwise"):
            actionReturn = agent.rotate("clockwise")
            if ((actionReturn != None) and
                (actionReturn[0] == 4) and                  # If agent touched a box ...
                (actionReturn[1].moveable == True) and      # .. and the box is unlocked ..
                (actionReturn[1].held == False)):           # .. and the agent doesn't hold the box
                moveObjectDueHit(agent, actionReturn[1], actionReturn[2])
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    if ((observedObject.moveable == True) and (observedObject.held == True)):
                        moveTranslateObject(agent, observedObject)

        elif (action == "counterclockwise"):
            actionReturn = agent.rotate("counterclockwise")
            if ((actionReturn != None) and
                (actionReturn[0] == 4) and                  # If agent touched a box ...
                (actionReturn[1].moveable == True) and      # .. and the box is unlocked ..
                (actionReturn[1].held == False)):           # .. and the agent doesn't hold the box
                moveObjectDueHit(agent, actionReturn[1], actionReturn[2])
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    if ((observedObject.moveable == True) and (observedObject.held == True)):
                        moveTranslateObject(agent, observedObject)

        elif (action == "lock"):
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    for intersectionPoint in agent.objectsWithDistances[observedObject]:
                        if (intersectionPoint[1] <= 0.8):
                            agent.lock(observedObject)

        elif (action == "unlock"):
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    for intersectionPoint in agent.objectsWithDistances[observedObject]:
                        if (intersectionPoint[1] <= 0.8):
                            agent.unlock(observedObject)

        elif (action == "hold"):
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    for intersectionPoint in agent.objectsWithDistances[observedObject]:
                        if ((intersectionPoint[1] <= 1.3) and (intersectionPoint[1] > 0.8)):
                            agent.hold(observedObject)

        elif (action == "release"):
            for observedObject in (agent.objectsWithDistances).keys():
                if (isinstance(observedObject, Box) or isinstance(observedObject, Ramp)):
                    if (observedObject.held == True):
                        agent.release(observedObject)

        observationFromAgent = agent.computeObservationSpace()
        observationMasked = {}
        observationOmniscient = {}

        observationMasked['lidar'] = observationFromAgent["lidarSensor"]
        observationMasked['currentAgent'] = [agent.position[0], agent.position[1], agent.orientation]
        observationMasked['otherAgents'] = []
        observationMasked['boxes'] = []
        observationMasked['ramps'] = []

        observationOmniscient['lidar'] = observationFromAgent["lidarSensor"]
        observationOmniscient['currentAgent'] = [agent.position[0], agent.position[1], agent.orientation]
        observationOmniscient['otherAgents'] = []
        observationOmniscient['boxes'] = []
        observationOmniscient['ramps'] = []

        for otherAgent in self.grid.getAgentList():
            saw = 0
            if (otherAgent == agent):
                continue
            observationOmniscient['otherAgents'].append([otherAgent.position[0], otherAgent.position[1], otherAgent.orientation])
            for object in observationFromAgent["visualField"]:
                if (isinstance(object, Agent)):
                    if (otherAgent == object):
                        observationMasked['otherAgents'].append([object.position[0], object.position[1], object.orientation])
                        saw = 1
                        break
            if (saw == 0):
                observationMasked['otherAgents'].append([0.0, 0.0, 0.0])

        for box in self.grid.getBoxList():
            saw = 0
            observationOmniscient['boxes'].append([box.position[0], box.position[1], box.orientation, box.side1, box.side2])
            for object in observationFromAgent["visualField"]:
                if (isinstance(object, Box)):
                    if (box == object):
                        observationMasked['boxes'].append([object.position[0], object.position[1], object.orientation, object.side1, object.side2])
                        saw = 1
                        break
            if (saw == 0):
                observationMasked['boxes'].append([0.0, 0.0, 0.0, 0.0, 0.0])

        for ramp in self.grid.getRampList():
            saw = 0
            observationOmniscient['ramps'].append([ramp.position[0], ramp.position[1], ramp.orientation])
            for object in observationFromAgent["visualField"]:
                if (isinstance(object, Ramp)):
                    if (ramp == object):
                        observationMasked['ramps'].append([object.position[0], object.position[1], object.orientation])
                        saw = 1
                        break
            if (saw == 0):
                observationMasked['ramps'].append([0.0, 0.0, 0.0])

        # Print of all the objects in the grid
        """for agent in self.grid.getAgentList():
            print("Agents: ", agent.id, agent.position, agent.orientation)
        for box in self.grid.getBoxList():
            print("Box: ", box.id, box.position, box.orientation)
            print("     team:", box.lockedTeam, "moveable", box.moveable, "held", box.held)
        for ramp in self.grid.getRampList():
            print("Ramp: ", ramp.id, ramp.position, ramp.orientation)
            print("      team:", ramp.lockedTeam, "moveable", ramp.moveable, "held", ramp.held)
        for cylinder in self.grid.getCylinderList():
            print("Cylinder: ", cylinder.id, cylinder.center, cylinder.radius)
        print("--------------------------------------------------")"""

        observation = [observationMasked, observationOmniscient]
        reward = agent.computeReward()
        done = False
        info = None

        self.render(Parameters.renderArgs, reset = 1)

        return observation, reward, done, info

    def reset(self):

        for currentTry in range(0, self.maxTries):
            box1Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            box1Orientation = random.uniform(0.0, 360.0)
            box1 = Box(box1Position, box1Orientation, "squared", tangible = True, moveable = True, lockedTeam = None)
            ret = self.grid.addBoxToGrid(box1)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            box2Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            box2Orientation = random.uniform(0.0, 360.0)
            box2 = Box(box2Position, box2Orientation, "squared", tangible = True, moveable = True, lockedTeam = None)
            ret = self.grid.addBoxToGrid(box2)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            ramp1Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            ramp1Orientation = random.uniform(0.0, 360.0)
            ramp1 = Ramp(ramp1Position, ramp1Orientation, tangible = True, moveable = True, lockedTeam = None)
            ret = self.grid.addRampToGrid(ramp1)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            ramp2Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            ramp2Orientation = random.uniform(0.0, 360.0)
            ramp2 = Ramp(ramp2Position, ramp2Orientation, tangible = True, moveable = True, lockedTeam = None)
            ret = self.grid.addRampToGrid(ramp2)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            hider1Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            hider1Orientation = random.uniform(0.0, 360.0)
            hider1 = Agent("hiders", hider1Position, hider1Orientation, tangible = True, controllable = False)
            ret = self.grid.addAgentToGrid(hider1)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            hider2Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            hider2Orientation = random.uniform(0.0, 360.0)
            hider2 = Agent("hiders", hider2Position, hider2Orientation, tangible = True, controllable = False)
            ret = self.grid.addAgentToGrid(hider2)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            seeker1Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            seeker1Orientation = random.uniform(0.0, 360.0)
            seeker1 = Agent("seekers", seeker1Position, seeker1Orientation, tangible = True, controllable = False)
            ret = self.grid.addAgentToGrid(seeker1)
            if (ret == 0):
                break

        for currentTry in range(0, self.maxTries):
            seeker2Position = (random.uniform(0.0, 15.0), random.uniform(0.0, 15.0))
            seeker2Orientation = random.uniform(0.0, 360.0)
            seeker2 = Agent("seekers", seeker2Position, seeker2Orientation, tangible = True, controllable = False)
            ret = self.grid.addAgentToGrid(seeker2)
            if (ret == 0):
                break

        try:
            hider1.computeEnvironmentLines()
            hider1.computeObservationSpace()
        except:
            None

        try:
            hider2.computeEnvironmentLines()
            hider2.computeObservationSpace()
        except:
            None

        try:
            seeker1.computeEnvironmentLines()
            seeker1.computeObservationSpace()
        except:
            None

        try:
            seeker2.computeEnvironmentLines()
            seeker2.computeObservationSpace()
        except:
            None

        self.render(Parameters.renderArgs, reset = 0)

    def render(self, renderArgs, close = False, reset = 0):
        self.window.render(self.grid, renderArgs, reset)