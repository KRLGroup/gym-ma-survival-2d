from multiagent_survival.gym_hidenseek.utils.utils import *
from multiagent_survival.gym_hidenseek.modules.wall import *

class Grid():

    gridSize = 0
    corners = []
    perimeterWalls = []

    wallList = []
    wallAmount = 0

    boxList =[]
    boxAmount = 0

    rampList =[]
    rampAmount = 0

    cylinderList = []
    cylinderAmount = 0

    agentList = []
    agentAmount = 0
    hiderList = []
    hiderAmount = 0
    seekerList = []
    seekerAmount = 0

    # EP: Grid End Point
    # EPLowSX left low point
    # EPLowRX right low point
    # EPHighRX right high point
    # EPHighSX left low point
    def __init__(self, gridSize):
        self.gridSize = gridSize
        EPLowSX = (0.0, 0.0)
        EPLowRX = (gridSize, 0.0)
        EPHighRX = (gridSize, gridSize)
        EPHighSX = (0.0, gridSize)
        self.corners = [EPLowSX, EPLowRX, EPHighRX, EPHighSX]

        lowerPerimeterWall = Wall(EPLowSX, EPLowRX)
        rightPerimeterWall = Wall(EPLowRX, EPHighRX)
        upperPerimeterWall = Wall(EPHighSX, EPHighRX)
        leftPerimeterWall = Wall(EPHighSX, EPLowSX)
        self.perimeterWalls = [lowerPerimeterWall, rightPerimeterWall, upperPerimeterWall, leftPerimeterWall]
        self.wallList = [lowerPerimeterWall, rightPerimeterWall, upperPerimeterWall, leftPerimeterWall]

    # Add a wall to the grid.
    # addEvenIfOutOfBound : boolean. If true, the wall portion inside the grid is added even if the entire wall is not
    #                       fully included in the grid.
    # return values: the wall - wall correctly added to the grid
    #                1 - wall not added to the grid because it coincides with a perimeter wall
    #                2 - wall not added to the grid because it is out of the grid
    #                the new wall - wall correctly but not entirely added to the grid, following addEvenIfOutOfBound param
    #                4 - wall not added to the grid, following addEvenIfOutOfBound param
    #                5 - wall not added to the grid because start and end points coincide
    def addWallToGrid(self, wall, addEvenIfOutOfBound):

        wallStartPoint, wallEndPoint = wall.getWallEndPoints()[0], wall.getWallEndPoints()[1]

        if (wallStartPoint == wallEndPoint):
            return 5

        newWallXList = [wallStartPoint[0], wallEndPoint[0]]
        newWallYList = [wallStartPoint[1], wallEndPoint[1]]
        newWallXList.sort()
        newWallYList.sort()

        if ((wallStartPoint[0] == 0.0 and wallEndPoint[0] == 0.0) or
            (wallStartPoint[0] == self.gridSize and wallEndPoint[0] == self.gridSize) or
            (wallStartPoint[1] == 0.0 and wallEndPoint[1] == 0.0) or
            (wallStartPoint[1] == self.gridSize and wallEndPoint[1] == self.gridSize)):
            return 1

        intersectionPointList = []
        for perimeterWall in self.perimeterWalls:
            intersectionPoint = intersectionTwoLines(perimeterWall.startPoint, perimeterWall.endPoint, wallStartPoint, wallEndPoint)
            if (intersectionPoint == (None, None)):
                continue
            else:
                if ((intersectionPoint[0] > newWallXList[0]) and
                    (intersectionPoint[0] < newWallXList[1]) and
                    (intersectionPoint[1] > newWallYList[0]) and
                    (intersectionPoint[1] < newWallYList[1])):
                    intersectionPointList.append(intersectionPoint)

        if (len(intersectionPointList) > 0):
            if (addEvenIfOutOfBound == True):
                if (len(intersectionPointList) == 1):
                    if ((wallStartPoint[0] <= 0.0) or
                        (wallStartPoint[0] >= self.gridSize) or
                        (wallStartPoint[1] <= 0.0) or
                        (wallStartPoint[1] >= self.gridSize)):
                        newWall = Wall(intersectionPointList[0], wallEndPoint)
                    elif ((wallEndPoint[0] <= 0.0) or
                          (wallEndPoint[0] >= self.gridSize) or
                          (wallEndPoint[1] <= 0.0) or
                          (wallEndPoint[1] >= self.gridSize)):
                        newWall = Wall(wallStartPoint, intersectionPointList[0])
                else:
                    newWall = Wall(intersectionPointList[0], intersectionPointList[1])
                self.wallList.append(newWall)
                self.wallAmount += 1
                return newWall
            else:
                return 4
        else:
            if ((wallStartPoint[0] >= 0.0 and wallStartPoint[0] <= self.gridSize) and
                (wallStartPoint[1] >= 0.0 and wallStartPoint[1] <= self.gridSize) and
                (wallEndPoint[0] >= 0.0 and wallEndPoint[0] <= self.gridSize) and
                (wallEndPoint[1] >= 0.0 and wallEndPoint[1] <= self.gridSize)):
                self.wallList.append(wall)
                self.wallAmount += 1
                return wall
            else:
                return 2

    # Remove a wall from the grid.
    # return values: 0 - wall correctly removed from the grid
    #                1 - wall not removed from the grid
    def removeWallFromGrid(self, wall):
        try:
            self.wallList.remove(wall)
            self.wallAmount = self.wallAmount - 1
            return 0
        except:
            return 1

    # Add a door of size doorSize, in position doorPosition to the wall wall.
    # doorPosition: position of the door in the wall. Can be:
    #               "begin" - door at the wall begin
    #               "middle" - door at the wall middle
    #               "end" - door at the wall end
    # doorSize: "small" - small door with size 2.0
    #           "large" - large door with size 4.0
    # return values: 0 - door correctly added to the wall
    #                1 - doot not added to the wall
    def addDoorToWall(self, wall, doorPosition, doorSize):

        doorCords = wall.computeDoorCoords(doorPosition, doorSize)

        newWall1 = Wall(wall.startPoint, doorCords[0])
        newWall2 = Wall(doorCords[1], wall.endPoint)
        added = self.addWallToGrid(newWall1, True)
        if (added == 1):
            return 1
        added = self.addWallToGrid(newWall2, True)
        if (added == 1):
            return 1
        removed = self.removeWallFromGrid(wall)
        if (removed == 1):
            return 1

        return 0

    # Build different scenarios inside the grid:
    # scenario: "half" - one wall in the middle with a random door
    #           "quadrant" - one quadrant is walled off with random door(s)
    #           "var_quadrant" - same as "quadrant" but with random room size
    #           "var_tri" - three rooms, one taking the half of the grid, two taking a remaining quarter each
    # return values: 0 - scenario correctly built
    #                1 - scenario not built
    def addScenario(self, scenario):
        if (scenario == "half"):
            # randomHorOrVer: 0 - horizontal wall (parallel to x axis)
            #                 1 - vertical wall (parallel to y axis)
            randomHorOrVer = random.randint(0, 1)

            if (randomHorOrVer == 0):
                wall = Wall((0.0, (self.gridSize / 2)), (self.gridSize, (self.gridSize / 2)))
            elif (randomHorOrVer == 1):
                wall = Wall(((self.gridSize / 2), 0.0), ((self.gridSize / 2), self.gridSize))
            else:
                return 1

            doorPosition, doorSize = randomDoorGenerator()

            self.addWallToGrid(wall, True)
            self.addDoorToWall(wall, doorPosition, doorSize)
            return 0

        elif (scenario == "quadrant"):
            # randomQuadrant: 0 - lower left quadrant
            #                 1 - lower right quadrant
            #                 2 - upper right quadrant
            #                 3 - upper left quadrant
            # randomDoorAmount: door amount
            randomQuadrant = random.randint(0, 3)
            randomDoorAmount = random.randint(0, 2)

            if (randomQuadrant == 0):
                wall1 = Wall(((self.gridSize / 2), 0.0), ((self.gridSize / 2), (self.gridSize / 2)))
                wall2 = Wall((0.0, (self.gridSize / 2)), ((self.gridSize / 2), (self.gridSize / 2)))
            elif (randomQuadrant == 1):
                wall1 = Wall(((self.gridSize / 2), 0.0), ((self.gridSize / 2), (self.gridSize / 2)))
                wall2 = Wall(((self.gridSize / 2), (self.gridSize / 2)), (self.gridSize, (self.gridSize / 2)))
            elif (randomQuadrant == 2):
                wall1 = Wall(((self.gridSize / 2), (self.gridSize / 2)), (self.gridSize, (self.gridSize / 2)))
                wall2 = Wall(((self.gridSize / 2), (self.gridSize / 2)), ((self.gridSize / 2), self.gridSize))
            elif (randomQuadrant == 3):
                wall1 = Wall(((self.gridSize / 2), (self.gridSize / 2)), ((self.gridSize / 2), self.gridSize))
                wall2 = Wall((0.0, (self.gridSize / 2)), ((self.gridSize / 2), (self.gridSize / 2)))
            else:
                return 1

            self.addWallToGrid(wall1, True)
            self.addWallToGrid(wall2, True)

            randomWallSelect = 0
            for _ in range(0, randomDoorAmount):
                # randomWallSelect: 0 - first wall
                #                   1 - second wall
                if (randomWallSelect == 0):
                    randomWallSelect = random.randint(1, 2)

                if (randomWallSelect == 1):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall1, doorPosition, doorSize)
                    randomWallSelect = 2

                elif (randomWallSelect == 2):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall2, doorPosition, doorSize)
                    randomWallSelect = 1
                else:
                    return 1
            return 0

        elif (scenario == "var_quadrant"):
            # randomQuadrant: 0 - lower left quadrant
            #                 1 - lower right quadrant
            #                 2 - upper right quadrant
            #                 3 - upper left quadrant
            # randomDoorAmount: door amount
            randomQuadrant = random.randint(0, 3)
            randomDoorAmount = random.randint(0, 2)

            randomRoomSize = round(random.uniform((largeDoorSize * 2.5), (self.gridSize / 2)), 2)

            if (randomQuadrant == 0):
                wall1 = Wall((randomRoomSize, 0.0), (randomRoomSize, randomRoomSize))
                wall2 = Wall((0.0, randomRoomSize), (randomRoomSize, randomRoomSize))
            elif (randomQuadrant == 1):
                wall1 = Wall(((self.gridSize - randomRoomSize), 0.0), ((self.gridSize - randomRoomSize), randomRoomSize))
                wall2 = Wall(((self.gridSize - randomRoomSize), randomRoomSize), (self.gridSize, randomRoomSize))
            elif (randomQuadrant == 2):
                wall1 = Wall(((self.gridSize - randomRoomSize), (self.gridSize - randomRoomSize)), (self.gridSize, (self.gridSize - randomRoomSize)))
                wall2 = Wall(((self.gridSize - randomRoomSize), (self.gridSize - randomRoomSize)), ((self.gridSize - randomRoomSize), self.gridSize))
            elif (randomQuadrant == 3):
                wall1 = Wall((0.0, (self.gridSize - randomRoomSize)), (randomRoomSize, (self.gridSize - randomRoomSize)))
                wall2 = Wall((randomRoomSize, (self.gridSize - randomRoomSize)), (randomRoomSize, self.gridSize))
            else:
                return 1

            self.addWallToGrid(wall1, True)
            self.addWallToGrid(wall2, True)

            randomWallSelect = 0
            for _ in range(0, randomDoorAmount):
                # randomWallSelect: 0 - first wall
                #                   1 - second wall
                #                   2 - third wall
                if (randomWallSelect == 0):
                    randomWallSelect = random.randint(1, 2)

                if (randomWallSelect == 1):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall1, doorPosition, doorSize)
                    randomWallSelect = 2

                elif (randomWallSelect == 2):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall2, doorPosition, doorSize)
                    randomWallSelect = 1
                else:
                    return 1
            return 0

        elif (scenario == "var_tri"):
            # randomBigRoom: 0 - lower left + lower right quadrants
            #                1 - lower right + upper right quadrants
            #                2 - upper right + upper left quadrants
            #                3 - upper left + lower left quadrants
            # randomDoorAmount: door amount
            randomBigRoom = random.randint(0, 3)
            randomDoorAmount = random.randint(0, 3)
            randomWallSelect = []
            for doorIndex in range(0, randomDoorAmount):
                randomWallSelect.append(doorIndex + 1)

            if (randomBigRoom == 0):
                wall1 = Wall((0.0, (self.gridSize / 2)), ((self.gridSize / 2), (self.gridSize / 2)))
                wall2 = Wall(((self.gridSize / 2), (self.gridSize / 2)), (self.gridSize, (self.gridSize / 2)))
                wall3 = Wall(((self.gridSize / 2), (self.gridSize / 2)), ((self.gridSize / 2), self.gridSize))
            elif (randomBigRoom == 1):
                wall1 = Wall(((self.gridSize / 2), 0.0), ((self.gridSize / 2), (self.gridSize / 2)))
                wall2 = Wall(((self.gridSize / 2), (self.gridSize / 2)), ((self.gridSize / 2), self.gridSize))
                wall3 = Wall((0.0, (self.gridSize / 2)), ((self.gridSize / 2), (self.gridSize / 2)))
            elif (randomBigRoom == 2):
                wall1 = Wall((0.0, (self.gridSize / 2)), ((self.gridSize / 2), (self.gridSize / 2)))
                wall2 = Wall(((self.gridSize / 2), (self.gridSize / 2)), (self.gridSize, (self.gridSize / 2)))
                wall3 = Wall(((self.gridSize / 2), 0.0), ((self.gridSize / 2), (self.gridSize / 2)))
            elif (randomBigRoom == 3):
                wall1 = Wall(((self.gridSize / 2), 0.0), ((self.gridSize / 2), (self.gridSize / 2)))
                wall2 = Wall(((self.gridSize / 2), (self.gridSize / 2)), ((self.gridSize / 2), self.gridSize))
                wall3 = Wall(((self.gridSize / 2), (self.gridSize / 2)), (self.gridSize, (self.gridSize / 2)))
            else:
                return 1

            self.addWallToGrid(wall1, True)
            self.addWallToGrid(wall2, True)
            self.addWallToGrid(wall3, True)

            for _ in range(0, randomDoorAmount):
                # randomWallSelect: 0 - first wall
                #                   1 - second wall

                selectedDoor = random.choice(randomWallSelect)

                if (selectedDoor == 1):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall1, doorPosition, doorSize)
                elif (selectedDoor == 2):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall2, doorPosition, doorSize)
                elif (selectedDoor == 3):
                    doorPosition, doorSize = randomDoorGenerator()
                    self.addDoorToWall(wall3, doorPosition, doorSize)
                else:
                    return 1
                randomWallSelect.remove(selectedDoor)
            return 0

        else:
            return 1

    # Randomly add walls to the grid:
    # roomsAmount: number of rooms to create
    # minimumRoomsSize: minimum size of the rooms
    def addRandomWallsScenario(self, roomsAmount, minimumRoomsSize):

        for _ in range(0, roomsAmount):
            # randomPerimeterWall: 0 - lower perimeter wall
            #                      1 - right perimeter wall
            #                      2 - upper perimeter wall
            #                      3 - left perimeter wall
            randomPerimeterWall = random.randint(0, 3)
            randomRoomSize = round(random.uniform(minimumRoomsSize, self.gridSize), 2)
            randomRoomStartPoint = round(random.uniform(0.0, (self.gridSize - randomRoomSize)), 2)

            if (randomPerimeterWall == 0):
                # left wall
                wall1 = Wall((randomRoomStartPoint, 0.0), (randomRoomStartPoint, randomRoomSize))
                # upper wall
                wall2 = Wall((randomRoomStartPoint, randomRoomSize), (round((randomRoomStartPoint + randomRoomSize), 2), randomRoomSize))
                # right wall
                wall3 = Wall((round((randomRoomStartPoint + randomRoomSize), 2), randomRoomSize), (round((randomRoomStartPoint + randomRoomSize), 2), 0.0))

            elif (randomPerimeterWall == 1):
                # lower wall
                wall1 = Wall((round((self.gridSize - randomRoomSize), 2), randomRoomStartPoint), (self.gridSize, randomRoomStartPoint))
                # left wall
                wall2 = Wall((round((self.gridSize - randomRoomSize), 2), randomRoomStartPoint), (round((self.gridSize - randomRoomSize), 2), round((randomRoomStartPoint + randomRoomSize), 2)))
                # upper wall
                wall3 = Wall((round((self.gridSize - randomRoomSize), 2), round((randomRoomStartPoint + randomRoomSize), 2)), (self.gridSize, round((randomRoomStartPoint + randomRoomSize), 2)))

            elif (randomPerimeterWall == 2):
                # right wall
                wall1 = Wall((randomRoomStartPoint, self.gridSize), (randomRoomStartPoint, round((self.gridSize - randomRoomSize), 2)))
                # lower wall
                wall2 = Wall((randomRoomStartPoint, round((self.gridSize - randomRoomSize), 2)), (round((randomRoomStartPoint + randomRoomSize), 2), round((self.gridSize - randomRoomSize), 2)))
                # left wall
                wall3 = Wall((round((randomRoomStartPoint + randomRoomSize), 2), round((self.gridSize - randomRoomSize), 2)), (round((randomRoomStartPoint + randomRoomSize), 2), self.gridSize))

            # DA FARE
            elif (randomPerimeterWall == 3):
                # lower wall
                wall1 = Wall((0.0, randomRoomStartPoint), (randomRoomSize, randomRoomStartPoint))
                # left wall
                wall2 = Wall((randomRoomSize, randomRoomStartPoint), (randomRoomSize, round((randomRoomStartPoint + randomRoomSize), 2)))
                # upper wall
                wall3 = Wall((randomRoomSize, round((randomRoomStartPoint + randomRoomSize), 2)), (0.0, round((randomRoomStartPoint + randomRoomSize), 2)))

            else:
                return 1

            self.addWallToGrid(wall1, True)
            self.addWallToGrid(wall2, True)
            self.addWallToGrid(wall3, True)

            if (random.randint(0, 1) == 0):
                doorPosition, doorSize = randomDoorGenerator()
                self.addDoorToWall(wall1, doorPosition, doorSize)
            if (random.randint(0, 1) == 0):
                doorPosition, doorSize = randomDoorGenerator()
                self.addDoorToWall(wall2, doorPosition, doorSize)
            if (random.randint(0, 1) == 0):
                doorPosition, doorSize = randomDoorGenerator()
                self.addDoorToWall(wall3, doorPosition, doorSize)
        return 0

    # Add a box to the grid.
    # return values: 0 - box correctly added to the grid
    #                1 - box not added to the grid
    def addBoxToGrid(self, box):
        intersectionWithGridObjectsRet = intersectionWithGridObjects(self, box)
        if (intersectionWithGridObjectsRet[0] == 0):
            self.boxList.append(box)
            box.belongingGrid = self

            self.boxAmount += 1
            box.id = "Box-" + str(self.boxAmount)

            return 0
        else:
            return 1

    # Remove a box from the grid.
    # return values: 0 - box correctly removed from the grid
    #                1 - box not removed from the grid
    def removeBoxFromGrid(self, box):
        try:
            self.boxList.remove(box)
            self.boxAmount = self.boxAmount - 1
            return 0
        except:
            return 1

    # Add a ramp to the grid.
    # return values: 0 - ramp correctly added to the grid
    #                1 - ramp not added to the grid
    def addRampToGrid(self, ramp):
        intersectionWithGridObjectsRet = intersectionWithGridObjects(self, ramp)

        if (intersectionWithGridObjectsRet[0] == 0):
            self.rampList.append(ramp)
            ramp.belongingGrid = self

            self.rampAmount += 1
            ramp.id = "Ramp-" + str(self.rampAmount)

            return 0

        else:
            return 1

    # Remove a ramp from the grid.
    # return values: 0 - ramp correctly removed from the grid
    #                1 - ramp not removed from the grid
    def removeRampFromGrid(self, ramp):
        try:
            self.rampList.remove(ramp)
            self.rampAmount = self.rampAmount - 1
            return 0
        except:
            return 1

    # Add a agent to the grid.
    # return values: 0 - agent correctly added to the grid
    #                1 - agent not added to the grid
    def addAgentToGrid(self, agent):
        intersectionWithGridObjectsRet = intersectionWithGridObjects(self, agent)

        if (intersectionWithGridObjectsRet[0] == 0):
            self.agentList.append(agent)
            agent.belongingGrid = self

            self.agentAmount += 1
            agent.id = "Agent-" + str(self.agentAmount)

            if (agent.team == "hiders"):
                self.hiderList.append(agent)
                self.hiderAmount += 1

            elif (agent.team == "seekers"):
                self.seekerList.append(agent)
                self.seekerAmount += 1

            return 0

        else:
            return 1

    # Remove a agent from the grid.
    # return values: 0 - agent correctly removed from the grid
    #                1 - agent not removed from the grid
    def removeAgentFromGrid(self, agent):
        try:
            self.agentList.remove(agent)
            self.agentAmount = self.agentAmount - 1

            if (agent.team == "hiders"):
                self.hiderList.remove(agent)
                self.hiderAmount = self.hiderAmount - 1

            elif (agent.team == "seekers"):
                self.seekerList.remove(agent)
                self.seekerAmount = self.seekerAmount - 1
            return 0
        except:
            return 1

    # Add a cylinder to the grid.
    # return values: 0 - cylinder correctly added to the grid
    #                1 - cylinder not added to the grid
    def addCylinderToGrid(self, cylinder):
        intersectionWithGridObjectsRet = intersectionWithGridObjects(self, cylinder)

        if (intersectionWithGridObjectsRet == 0):
            self.cylinderList.append(cylinder)
            cylinder.belongingGrid = self

            self.cylinderAmount += 1
            cylinder.id = "Cylinder-" + str(self.cylinderAmount)

            return 0

        else:
            return 1

    # Remove a cylinder from the grid.
    # return values: 0 - cylinder correctly removed from the grid
    #                1 - cylinder not removed from the grid
    def removeCylinderFromGrid(self, cylinder):
        try:
            self.cylinderList.remove(cylinder)
            self.cylinderAmount = self.cylinderAmount - 1
            return 0
        except:
            return 1

    # Return the wall list.
    def getWallList(self):
        return self.wallList

    # Return the wall amount.
    def getWallAmount(self):
        return self.wallAmount

    # Return the box list.
    def getBoxList(self):
        return self.boxList

    # Return the box amount.
    def getBoxAmount(self):
        return self.boxAmount

    # Return the ramp list.
    def getRampList(self):
        return self.rampList

    # Return the ramp amount.
    def getRampAmount(self):
        return self.rampAmount

    # Return the cylinder list.
    def getCylinderList(self):
        return self.cylinderList

    # Return the cylinder amount.
    def getCylinderAmount(self):
        return self.cylinderAmount

    # Return the agent list.
    def getAgentList(self):
        return self.agentList

    # Return the agent amount.
    def getAgentAmount(self):
        return self.agentAmount

    # Return the hiders list.
    def getHidersList(self):
        return self.hiderList

    # Return the hiders amount.
    def getHidersAmount(self):
        return self.hiderAmount

    # Return the seekers list.
    def getSeekersList(self):
        return self.seekerList

    # Return the seekers amount.
    def getSeekersAmount(self):
        return self.seekerAmount