import gym
from gym_hidenseek.modules.grid import *
import gym_hidenseek.utils.graphic_rendering as graphic_rendering

import json, pathlib, re

class JsonHideAndSeek15x15Env(gym.Env):

    grid = None
    gridSize = 15.0

    jsonFilePath = str(pathlib.Path(__file__).parent.absolute()) + "/env_setting.json"

    window = None

    def __init__(self):

        self.grid = Grid(self.gridSize)

        jsonFile = open(self.jsonFilePath)
        data = json.load(jsonFile)

        wallsData = data["walls"]
        doorsData = data["doors"]
        cylindersData = data["cylinders"]

        for wall in wallsData:
            if (wall[2] == "True"):
                addEvenValue = True
            else:
                addEvenValue = False
            currentWall = Wall(wall[0], wall[1])
            self.grid.addWallToGrid(currentWall, addEvenValue)

        for door in doorsData:
            doorIndex = int(re.findall(r'\d+', door[0])[0]) - 1
            for index, wall in enumerate(self.grid.getWallList()):
                if ((wall.startPoint == wallsData[doorIndex][0]) and (wall.endPoint == wallsData[doorIndex][1])):
                    self.grid.addDoorToWall(self.grid.getWallList()[index], door[1], door[2])

        for cylinder in cylindersData:
            currentCylinder = Cylinder(cylinder[0], cylinder[1], cylinder[2])
            self.grid.addCylinderToGrid(currentCylinder)

        jsonFile.close()

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

        jsonFile = open(self.jsonFilePath)
        data = json.load(jsonFile)

        boxesData = data["boxes"]
        rampsData = data["ramps"]
        hidersData = data["hiders"]
        seekersData = data["seekers"]

        for box in boxesData:
            if (box[3] == "True"):
                tangibleValue = True
            else:
                tangibleValue = False
            if (box[4] == "True"):
                moveableValue = True
            else:
                moveableValue = False
            if (box[5] == "None"):
                teamValue = None
            currentBox = Box(box[0], box[1], box[2], tangibleValue, moveableValue, teamValue)
            self.grid.addBoxToGrid(currentBox)

        for ramp in rampsData:
            if (ramp[2] == "True"):
                tangibleValue = True
            else:
                tangibleValue = False
            if (ramp[3] == "True"):
                moveableValue = True
            else:
                moveableValue = False
            if (ramp[4] == "None"):
                teamValue = None
            currentRamp = Ramp(ramp[0], ramp[1], tangibleValue, moveableValue, teamValue)
            self.grid.addRampToGrid(currentRamp)

        for hider in hidersData:
            if (hider[2] == "True"):
                tangibleValue = True
            else:
                tangibleValue = False
            if (hider[3] == "True"):
                moveableValue = True
            else:
                moveableValue = False
            currentHider = Agent("hiders", hider[0], hider[1], tangibleValue, moveableValue)
            self.grid.addAgentToGrid(currentHider)

        for seeker in seekersData:
            if (seeker[2] == "True"):
                tangibleValue = True
            else:
                tangibleValue = False
            if (seeker[3] == "True"):
                moveableValue = True
            else:
                moveableValue = False
            currentSeeker = Agent("seekers", seeker[0], seeker[1], tangibleValue, moveableValue)
            self.grid.addAgentToGrid(currentSeeker)

        for agent in self.grid.getAgentList():
            agent.computeEnvironmentLines()
            agent.computeObservationSpace()

        self.render(Parameters.renderArgs, reset = 0)

    def render(self, renderArgs, close = False, reset = 0):
        self.window.render(self.grid, renderArgs, reset)