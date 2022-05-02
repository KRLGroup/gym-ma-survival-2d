from multiagent_survival.gym_hidenseek.utils import utils
import multiagent_survival.gym_hidenseek.utils.parameters as Parameters
from multiagent_survival.gym_hidenseek.modules.cylinder import Cylinder

import math
import numpy as np

class Agent():

    id = None
    belongingGrid = None
    team = None

    position = None
    orientation = None
    tangible = True

    side1 = None
    side2 = None

    vertex1 = None
    vertex2 = None
    vertex3 = None
    vertex4 = None

    vertsList = []

    lidarSensor = []
    lidarDistances = []

    lidarPoints = []

    objectLines = []
    linesEnvironment = None

    visualFieldObservation = []
    objectsWithDistances = {}

    controllable = False

    onRamp = True

    # Angle described in thesis
    constantBeta = None
    constantGamma = None
    constantDistanceHeldObject = None

    alreadyLocked = 0

    vec1 = None
    vec2 = None

    def __init__(self, team, position, orientation, tangible, controllable):
        self.team = team

        self.position = position
        self.orientation = math.radians(orientation % 360.0)

        self.tangible = tangible
        self.controllable = controllable

        self.side1 = Parameters.agentSide1
        self.side2 = Parameters.agentSide2

        utils.computeHitbox(self, self.side1, self.side2)

        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]

        # 30 laser as the original simulator
        lidarLines = []
        for lidarIndex in range(0, 30):
            lidarLines.append(((self.position),
                               ((Parameters.lidarLength * math.cos(math.radians(12.0 * lidarIndex))),
                                (Parameters.lidarLength * math.sin(math.radians(12.0 * lidarIndex))))))
        self.lidarSensor = lidarLines

    def computeEnvironmentLines(self):
        arrayLines = []

        for wall in self.belongingGrid.getWallList():
            wallVertices = wall.getVertList()
            for indexVertex in range(0, len(wallVertices)):
                secondVertex = indexVertex + 1
                if (secondVertex == len(wallVertices)):
                    secondVertex = 0
                arrayLines.append([wallVertices[indexVertex], wallVertices[secondVertex]])
                self.objectLines.append(wall)

        for agent in self.belongingGrid.getAgentList():
            if (self.id == agent.id):
                continue
            agentVertices = agent.getVertList()
            for indexVertex in range(0, len(agentVertices)):
                secondVertex = indexVertex + 1
                if (secondVertex == len(agentVertices)):
                    secondVertex = 0
                arrayLines.append([agentVertices[indexVertex], agentVertices[secondVertex]])
                self.objectLines.append(agent)

        for box in self.belongingGrid.getBoxList():
            if (self.id == box.id):
                continue
            boxVertices = box.getVertList()
            for indexVertex in range(0, len(boxVertices)):
                secondVertex = indexVertex + 1
                if (secondVertex == len(boxVertices)):
                    secondVertex = 0
                arrayLines.append([boxVertices[indexVertex], boxVertices[secondVertex]])
                self.objectLines.append(box)

        for ramp in self.belongingGrid.getRampList():
            if (self.id == ramp.id):
                continue
            rampVertices = ramp.getVertList()
            for indexVertex in range(0, len(rampVertices)):
                secondVertex = indexVertex + 1
                if (secondVertex == len(rampVertices)):
                    secondVertex = 0
                arrayLines.append([rampVertices[indexVertex], rampVertices[secondVertex]])
                self.objectLines.append(ramp)

        self.linesEnvironment = np.array(arrayLines, dtype = "float32")

    def computeObservationSpace(self):

        self.visualFieldObservation = self.computeVisualField()

        # First version
        #lidarDistanceObservation = self.computeLidarDistancesv1()

        # Vectorized version
        #lidarDistanceObservation = self.computeLidarDistancesv2()

        # First version with distances assumptions
        lidarDistanceObservation = self.computeLidarDistancesv3()

        return {"visualField": self.visualFieldObservation, "lidarSensor" : lidarDistanceObservation}

    def computeVisualField(self):
        observation = []

        limitPointSX = ((self.position[0] + (Parameters.radiusVisualField * math.cos(self.orientation + (Parameters.angleVisualFieldRad / 2)))),
                        (self.position[1] + (Parameters.radiusVisualField * math.sin(self.orientation + (Parameters.angleVisualFieldRad / 2)))))
        limitPointDX = ((self.position[0] + (Parameters.radiusVisualField * math.cos(self.orientation - (Parameters.angleVisualFieldRad / 2)))),
                        (self.position[1] + (Parameters.radiusVisualField * math.sin(self.orientation - (Parameters.angleVisualFieldRad / 2)))))
        self.vec1 = limitPointSX
        self.vec2 = limitPointDX

        for wall in self.belongingGrid.getWallList():
            wallVertices = wall.getVertList()
            for indexVertex in range(0, len(wallVertices)):
                if (utils.pointInsideCircularSector(wallVertices[indexVertex], self, "squared") == True):
                    if (wall not in observation):
                        observation.append(wall)
                        continue

                secondVertex = indexVertex + 1
                if (secondVertex == len(wallVertices)):
                    secondVertex = 0

                intersectionsCL = utils.intersectionCircleLine(self.position, Parameters.radiusVisualField, wallVertices[indexVertex], wallVertices[secondVertex])

                intersectionLLLeft = utils.intersectionTwoLines(wallVertices[indexVertex], wallVertices[secondVertex], limitPointSX, self.position)
                intersectionLLRight = utils.intersectionTwoLines(wallVertices[indexVertex], wallVertices[secondVertex], limitPointDX, self.position)

                intersections = []
                for intersectionPoint in intersectionsCL:
                    intersections.append(intersectionPoint)
                intersections.append(intersectionLLLeft)
                intersections.append(intersectionLLRight)

                for intersectionPoint in intersections:
                    if (intersectionPoint == (None, None)):
                        continue
                    else:
                        if (utils.pointInsideCircularSector(intersectionPoint, self, "squared") == True):
                            if (wall not in observation):
                                observation.append(wall)

        for agent in self.belongingGrid.getAgentList():
            if (self.id == agent.id):
                continue
            agentVertices = agent.getVertList()
            for indexVertex in range(0, len(agentVertices)):
                if (utils.pointInsideCircularSector(agentVertices[indexVertex], self, "squared") == True):
                    if (agent not in observation):
                        observation.append(agent)
                        continue

                secondVertex = indexVertex + 1
                if (secondVertex == len(agentVertices)):
                    secondVertex = 0

                intersectionsCL = utils.intersectionCircleLine(self.position, Parameters.radiusVisualField, agentVertices[indexVertex], agentVertices[secondVertex])

                intersectionLLLeft = utils.intersectionTwoLines(agentVertices[indexVertex], agentVertices[secondVertex], limitPointSX, self.position)
                intersectionLLRight = utils.intersectionTwoLines(agentVertices[indexVertex], agentVertices[secondVertex], limitPointDX, self.position)

                intersections = []
                for intersectionPoint in intersectionsCL:
                    intersections.append(intersectionPoint)
                intersections.append(intersectionLLLeft)
                intersections.append(intersectionLLRight)

                for intersectionPoint in intersections:
                    if (intersectionPoint == (None, None)):
                        continue
                    else:
                        if (utils.pointInsideCircularSector(intersectionPoint, self, "squared") == True):
                            if (agent not in observation):
                                observation.append(agent)

        for box in self.belongingGrid.getBoxList():
            if (self.id == box.id):
                continue
            boxVertices = box.getVertList()
            for indexVertex in range(0, len(boxVertices)):
                if (utils.pointInsideCircularSector(boxVertices[indexVertex], self, "squared") == True):
                    if (box not in observation):
                        observation.append(box)
                        continue

                secondVertex = indexVertex + 1
                if (secondVertex == len(boxVertices)):
                    secondVertex = 0

                intersectionsCL = utils.intersectionCircleLine(self.position, Parameters.radiusVisualField, boxVertices[indexVertex], boxVertices[secondVertex])

                intersectionLLLeft = utils.intersectionTwoLines(boxVertices[indexVertex], boxVertices[secondVertex], limitPointSX, self.position)
                intersectionLLRight = utils.intersectionTwoLines(boxVertices[indexVertex], boxVertices[secondVertex], limitPointDX, self.position)

                intersections = []
                for intersectionPoint in intersectionsCL:
                    intersections.append(intersectionPoint)
                intersections.append(intersectionLLLeft)
                intersections.append(intersectionLLRight)

                for intersectionPoint in intersections:
                    if (intersectionPoint == (None, None)):
                        continue
                    else:
                        if (utils.pointInsideCircularSector(intersectionPoint, self, "squared") == True):
                            if (box not in observation):
                                observation.append(box)

        for ramp in self.belongingGrid.getRampList():
            if (self.id == ramp.id):
                continue
            rampVertices = ramp.getVertList()
            for indexVertex in range(0, len(rampVertices)):
                if (utils.pointInsideCircularSector(rampVertices[indexVertex], self, "squared") == True):
                    if (ramp not in observation):
                        observation.append(ramp)
                        continue

                secondVertex = indexVertex + 1
                if (secondVertex == len(rampVertices)):
                    secondVertex = 0

                intersectionsCL = utils.intersectionCircleLine(self.position, Parameters.radiusVisualField, rampVertices[indexVertex], rampVertices[secondVertex])

                intersectionLLLeft = utils.intersectionTwoLines(rampVertices[indexVertex], rampVertices[secondVertex], limitPointSX, self.position)
                intersectionLLRight = utils.intersectionTwoLines(rampVertices[indexVertex], rampVertices[secondVertex], limitPointDX, self.position)

                intersections = []
                for intersectionPoint in intersectionsCL:
                    intersections.append(intersectionPoint)
                intersections.append(intersectionLLLeft)
                intersections.append(intersectionLLRight)

                for intersectionPoint in intersections:
                    if (intersectionPoint == (None, None)):
                        continue
                    else:
                        if (utils.pointInsideCircularSector(intersectionPoint, self, "squared") == True):
                            if (ramp not in observation):
                                observation.append(ramp)

        for cylinder in self.belongingGrid.getCylinderList():
            if (self.id == cylinder.id):
                continue
            if (utils.distanceTwoPoints(cylinder.center, self.position) <= (Parameters.radiusVisualField + cylinder.radius)):
                if (utils.pointInsideCircularSector(cylinder.center, self, "circular") == True):
                    if (cylinder not in observation):
                        observation.append(cylinder)

                if ((utils.distanceTwoPoints(limitPointSX, cylinder.center) <= cylinder.radius) or
                    (utils.distanceTwoPoints(limitPointDX, cylinder.center) <= cylinder.radius)):
                    if (cylinder not in observation):
                        observation.append(cylinder)

                intersect = False
                intersectionsWithLimitPointSX = utils.intersectionCircleLine(cylinder.center, cylinder.radius, self.position, limitPointSX)
                if (intersectionsWithLimitPointSX is not None):
                    for intersectionPoint in intersectionsWithLimitPointSX:
                        if (intersectionPoint != (None, None)):
                            intersect = True

                intersectionsWithLimitPointDX = utils.intersectionCircleLine(cylinder.center, cylinder.radius, self.position, limitPointDX)
                if (intersectionsWithLimitPointDX is not None):
                    for intersectionPoint in intersectionsWithLimitPointDX:
                        if (intersectionPoint != (None, None)):
                            intersect = True

                if (intersect == True):
                    if (cylinder not in observation):
                        observation.append(cylinder)

        for observedObject in observation:

            for comparedObservedObject in observation:

                try:
                    if (observedObject.id == comparedObservedObject.id):
                        continue
                except:
                    None


                intersectionCount = 0
                obsObjVerts = []
                comObsObjVerts = []
                if (isinstance(observedObject, Cylinder)):
                    obsObjVerts = [((observedObject.center[0] + (observedObject.radius / 2)), (observedObject.center[1])),
                                   ((observedObject.center[0]), (observedObject.center[1] + (observedObject.radius / 2))),
                                   ((observedObject.center[0] - (observedObject.radius / 2)), (observedObject.center[1])),
                                   ((observedObject.center[0]), (observedObject.center[1] - (observedObject.radius / 2)))]
                else:
                    obsObjVerts = observedObject.getVertList()

                for obsObjVertex in obsObjVerts:
                    intersectionFound = 0

                    if (isinstance(comparedObservedObject, Cylinder)):
                        comObsObjVerts = [((comparedObservedObject.center[0] + (comparedObservedObject.radius / 2)), (comparedObservedObject.center[1])),
                                          ((comparedObservedObject.center[0]), (comparedObservedObject.center[1] + (comparedObservedObject.radius / 2))),
                                          ((comparedObservedObject.center[0] - (comparedObservedObject.radius / 2)), (comparedObservedObject.center[1])),
                                          ((comparedObservedObject.center[0]), (comparedObservedObject.center[1] - (comparedObservedObject.radius / 2)))]
                    else:
                        comObsObjVerts = comparedObservedObject.getVertList()

                    for indexComObsObjVertex in range(0, len(comObsObjVerts)):
                        secondIndexComObsObjVertex = indexComObsObjVertex + 1
                        if (secondIndexComObsObjVertex == len(comObsObjVerts)):
                            secondIndexComObsObjVertex = 0

                        intersectionPointF = utils.intersectionTwoLines(obsObjVertex, self.position, comObsObjVerts[indexComObsObjVertex], comObsObjVerts[secondIndexComObsObjVertex])
                        if (intersectionPointF != (None, None)):
                            #intersectionCount += 1
                            intersectionFound = 1
                            break
                    if (intersectionFound == 1):
                        intersectionCount += 1
                if (intersectionCount == 4):
                    observation.remove(observedObject)
                    break


        return observation

    def computeLidarDistancesv1(self):

        lidarIntersectionPoints = []
        lidarDistances = []
        lidarObjects = []

        for lidarLine in self.lidarSensor:

            # candidates = point, distances, object
            candidates = [[], [], []]
            maxLineDistance = 9999.0

            for wall in self.belongingGrid.getWallList():
                selectedIntersectionPointWall = (None, None)
                wallVertices = wall.getVertList()
                for indexVertex in range(0, len(wallVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(wallVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], wallVertices[indexVertex], wallVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointWall = intersectionPoint
                candidates[0].append(selectedIntersectionPointWall)
                candidates[1].append(maxLineDistance)
                candidates[2].append(wall)

            for agent in self.belongingGrid.getAgentList():
                if (self.id == agent.id):
                    continue
                selectedIntersectionPointAgent = (None, None)
                agentVertices = agent.getVertList()
                for indexVertex in range(0, len(agentVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(agentVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], agentVertices[indexVertex], agentVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointAgent = intersectionPoint
                candidates[0].append(selectedIntersectionPointAgent)
                candidates[1].append(maxLineDistance)
                candidates[2].append(agent)

            for box in self.belongingGrid.getBoxList():
                if (self.id == box.id):
                    continue
                selectedIntersectionPointBox = (None, None)
                boxVertices = box.getVertList()
                for indexVertex in range(0, len(boxVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(boxVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], boxVertices[indexVertex], boxVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointBox = intersectionPoint
                candidates[0].append(selectedIntersectionPointBox)
                candidates[1].append(maxLineDistance)
                candidates[2].append(box)

            for ramp in self.belongingGrid.getRampList():
                if (self.id == ramp.id):
                    continue
                selectedIntersectionPointRamp = (None, None)
                rampVertices = ramp.getVertList()
                for indexVertex in range(0, len(rampVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(rampVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], rampVertices[indexVertex], rampVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointRamp = intersectionPoint
                candidates[0].append(selectedIntersectionPointRamp)
                candidates[1].append(maxLineDistance)
                candidates[2].append(ramp)

            for cylinder in self.belongingGrid.getCylinderList():
                if (self.id == cylinder.id):
                    continue
                selectedIntersectionPointCylinder = (None, None)
                intersect = (None, None)
                intersectionPoints = utils.intersectionCircleLine(cylinder.center, cylinder.radius, lidarLine[0], lidarLine[1])
                for intersectionPoint in intersectionPoints:
                    if (intersectionPoint != (None, None)):
                        intersect = intersectionPoint
                        currentDistance = utils.distanceTwoPoints(self.position, intersect)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointCylinder = intersect
                candidates[0].append(selectedIntersectionPointCylinder)
                candidates[1].append(maxLineDistance)
                candidates[2].append(cylinder)

            lidarIntersectionPoints.append(candidates[0][candidates[1].index(min(candidates[1]))])
            self.lidarPoints = lidarIntersectionPoints

            lidarDistances.append(min(candidates[1]))

            lidarObjects.append(candidates[2][candidates[1].index(min(candidates[1]))])

        lidarPointsInVisualField = {}
        for pointIndex in range(0, len(lidarIntersectionPoints)):
            if (utils.pointInsideCircularSector(lidarIntersectionPoints[pointIndex], self, "squared")):
                if (lidarObjects[pointIndex] not in lidarPointsInVisualField):
                    lidarPointsInVisualField[lidarObjects[pointIndex]] = []
                lidarPointsInVisualField[lidarObjects[pointIndex]].append([lidarIntersectionPoints[pointIndex], lidarDistances[pointIndex]])

        self.objectsWithDistances = lidarPointsInVisualField

        return lidarDistances

    def computeLidarDistancesv2(self):

        self.lidarPoints = []
        lidarDistances = []

        npLidarSensor = np.array(self.lidarSensor, dtype = "float32")

        vecIntersectionTwoLinesVector = np.vectorize(utils.intersectionTwoLinesVector, excluded = ['lidarLines', 'environmentLines', 'objectList', 'currentAgent'])
        lidarObjects = vecIntersectionTwoLinesVector(lidarLines = npLidarSensor,
                                                     environmentLines = self.linesEnvironment,
                                                     objectList = self.objectLines,
                                                     currentAgent = self)

        for index, lidarLine in enumerate(self.lidarSensor):
            for cylinder in self.belongingGrid.getCylinderList():
                intersect = (None, None)
                intersectionPoints = utils.intersectionCircleLine(cylinder.center, cylinder.radius, lidarLine[0], lidarLine[1])
                for intersectionPoint in intersectionPoints:
                    if (intersectionPoint != (None, None)):
                        intersect = intersectionPoint
                        currentDistance = utils.distanceTwoPoints(self.position, intersect)
                        if (currentDistance < lidarObjects[index][2]):
                            lidarObjects[index][0] = cylinder
                            lidarObjects[index][1] = intersect
                            lidarObjects[index][2] = currentDistance

        for lidarPoint in lidarObjects:
            self.lidarPoints.append(lidarPoint[1])
            lidarDistances.append(lidarPoint[2])

        lidarPointsInVisualField = {}
        for pointIndex in range(0, len(self.lidarPoints)):
            if (utils.pointInsideCircularSector(self.lidarPoints[pointIndex], self, "squared")):
                if (lidarObjects[pointIndex][0] not in lidarPointsInVisualField):
                    lidarPointsInVisualField[lidarObjects[pointIndex][0]] = []
                lidarPointsInVisualField[lidarObjects[pointIndex][0]].append([self.lidarPoints[pointIndex], lidarDistances[pointIndex]])

        self.objectsWithDistances = lidarPointsInVisualField

        return lidarDistances

    def computeLidarDistancesv3(self):

        lidarIntersectionPoints = []
        lidarDistances = []
        lidarObjects = []

        for lidarLine in self.lidarSensor:

            # candidates = point, distances, object
            candidates = [[], [], []]
            maxLineDistance = 100.0

            for wall in self.belongingGrid.getWallList():
                selectedIntersectionPointWall = (None, None)
                wallVertices = wall.getVertList()
                for indexVertex in range(0, len(wallVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(wallVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], wallVertices[indexVertex], wallVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointWall = intersectionPoint
                if (selectedIntersectionPointWall != (None, None)):
                    candidates[0].append(selectedIntersectionPointWall)
                    candidates[1].append(maxLineDistance)
                    candidates[2].append(wall)

            for agent in self.belongingGrid.getAgentList():
                if (self.id == agent.id):
                    continue
                distance = utils.distanceTwoPoints(self.position, agent.position)
                if (distance >= (maxLineDistance * 1.5)):
                    continue
                selectedIntersectionPointAgent = (None, None)
                agentVertices = agent.getVertList()
                for indexVertex in range(0, len(agentVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(agentVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], agentVertices[indexVertex], agentVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointAgent = intersectionPoint
                if (selectedIntersectionPointAgent != (None, None)):
                    candidates[0].append(selectedIntersectionPointAgent)
                    candidates[1].append(maxLineDistance)
                    candidates[2].append(agent)

            for box in self.belongingGrid.getBoxList():
                if (self.id == box.id):
                    continue
                distance = utils.distanceTwoPoints(self.position, box.position)
                if (distance >= (maxLineDistance * 1.5)):
                    continue
                selectedIntersectionPointBox = (None, None)
                boxVertices = box.getVertList()
                for indexVertex in range(0, len(boxVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(boxVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], boxVertices[indexVertex], boxVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointBox = intersectionPoint
                if (selectedIntersectionPointBox != (None, None)):
                    candidates[0].append(selectedIntersectionPointBox)
                    candidates[1].append(maxLineDistance)
                    candidates[2].append(box)

            for ramp in self.belongingGrid.getRampList():
                if (self.id == ramp.id):
                    continue
                distance = utils.distanceTwoPoints(self.position, ramp.position)
                if (distance >= (maxLineDistance * 1.5)):
                    continue
                selectedIntersectionPointRamp = (None, None)
                rampVertices = ramp.getVertList()
                for indexVertex in range(0, len(rampVertices)):
                    secondVertex = indexVertex + 1
                    if (secondVertex == len(rampVertices)):
                        secondVertex = 0
                    intersectionPoint = utils.intersectionTwoLines(lidarLine[0], lidarLine[1], rampVertices[indexVertex], rampVertices[secondVertex])
                    if (intersectionPoint != (None, None)):
                        currentDistance = utils.distanceTwoPoints(self.position, intersectionPoint)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointRamp = intersectionPoint
                if (selectedIntersectionPointRamp != (None, None)):
                    candidates[0].append(selectedIntersectionPointRamp)
                    candidates[1].append(maxLineDistance)
                    candidates[2].append(ramp)

            for cylinder in self.belongingGrid.getCylinderList():
                if (self.id == cylinder.id):
                    continue
                distance = utils.distanceTwoPoints(self.position, cylinder.center)
                if (distance >= (maxLineDistance * 1.5)):
                    continue
                selectedIntersectionPointCylinder = (None, None)
                intersect = (None, None)
                intersectionPoints = utils.intersectionCircleLine(cylinder.center, cylinder.radius, lidarLine[0], lidarLine[1])
                for intersectionPoint in intersectionPoints:
                    if (intersectionPoint != (None, None)):
                        intersect = intersectionPoint
                        currentDistance = utils.distanceTwoPoints(self.position, intersect)
                        if (currentDistance < maxLineDistance):
                            maxLineDistance = currentDistance
                            selectedIntersectionPointCylinder = intersect
                if (selectedIntersectionPointCylinder != (None, None)):
                    candidates[0].append(selectedIntersectionPointCylinder)
                    candidates[1].append(maxLineDistance)
                    candidates[2].append(cylinder)

            lidarIntersectionPoints.append(candidates[0][candidates[1].index(min(candidates[1]))])
            self.lidarPoints = lidarIntersectionPoints

            lidarDistances.append(min(candidates[1]))

            lidarObjects.append(candidates[2][candidates[1].index(min(candidates[1]))])

        lidarPointsInVisualField = {}
        for pointIndex in range(0, len(lidarIntersectionPoints)):
            if (utils.pointInsideCircularSector(lidarIntersectionPoints[pointIndex], self, "squared")):
                if (lidarObjects[pointIndex] not in lidarPointsInVisualField):
                    lidarPointsInVisualField[lidarObjects[pointIndex]] = []
                lidarPointsInVisualField[lidarObjects[pointIndex]].append([lidarIntersectionPoints[pointIndex], lidarDistances[pointIndex]])

        self.objectsWithDistances = lidarPointsInVisualField

        return lidarDistances

    def computeReward(self):

        if (self.team == "hiders"):
            for seeker in self.belongingGrid.getSeekersList():
                for object in seeker.visualFieldObservation:
                    if (isinstance(object, Agent)):
                        if (object.team == "hiders"):
                            return -1
            return 1

        if (self.team == "seekers"):
            hidersList = self.belongingGrid.getHidersList()
            hidersWithInfo = {}
            for hider in hidersList:
                hidersWithInfo[hider] = 0
            for seeker in self.belongingGrid.getSeekersList():
                for object in seeker.visualFieldObservation:
                    if (isinstance(object, Agent)):
                        if (object.team == "hiders"):
                            hidersWithInfo[object] = 1

            for value in hidersWithInfo.values():
                if (value == 0):
                    return -1

            return 1

    def computeRewardLockAndReturn(self, agentInitialPosition):
        reward = 0
        box = self.belongingGrid.getBoxList()[0]
        agent = self.belongingGrid.getAgentList()[0]

        if (box.moveable == True):
            targetPosition = box.position
        else:
            targetPosition = agentInitialPosition

        if (box.moveable == False):
            reward += 5
            self.alreadyLocked = 1

        elif ((box.moveable == True) and (self.alreadyLocked == 1)):
            reward -= 5

        if ((box.moveable == False) and (utils.distanceTwoPoints(agentInitialPosition, agent.position)) < 0.1):
            reward += 1

        reward += 0.5 * utils.distanceTwoPoints(targetPosition, agent.position)

        return reward

    # ACTIONS:
    #   translate: agent movement
    #   rotate: agent rotation
    #   lock: object locking, forbiddening their movement
    #   unlock: object unlocking, allowing their movement
    #   hold: agent grabs the object
    #   release: agent release the object
    def translate(self, movingDirection):

        if (movingDirection == "forward"):
            futurePosition = ((self.position[0] + (Parameters.unitMotion * math.cos(self.orientation))),
                             (self.position[1] + (Parameters.unitMotion * math.sin(self.orientation))))

        elif (movingDirection == "backward"):
            futurePosition = (self.position[0] - (Parameters.unitMotion * math.cos(self.orientation)),
                             self.position[1] - (Parameters.unitMotion * math.sin(self.orientation)))

        futureObject = Agent(self.team, futurePosition, math.degrees(self.orientation), self.tangible, self.controllable)
        futureObject.id = self.id
        intersect = utils.intersectionWithGridObjects(self.belongingGrid, futureObject)
        del(futureObject)
        if (intersect[0] == 0 or intersect[0] == 5):
            self.position = futurePosition
            utils.computeHitbox(self, Parameters.agentSide1, Parameters.agentSide2)
            onGround = 0
            for vertex in self.getVertList():
                for ramp in self.belongingGrid.getRampList():
                    if (utils.pointInsidePolygon(vertex, ramp.vertex1, ramp.vertex2, ramp.vertex3, ramp.vertex4)):
                        onGround += 1
            if (onGround > 1):
                self.onRamp = True
            else:
                self.onRamp = False

        elif ((intersect[0] == 3) and (self.onRamp == True)):
            self.position = futurePosition
            utils.computeHitbox(self, Parameters.agentSide1, Parameters.agentSide2)
        elif (intersect[0] == 4):
            return intersect
        else:
            return

    def rotate(self, rotatingDirection):

        if (rotatingDirection == "clockwise"):
            futureOrientation = self.orientation - Parameters.unitAngleRad

        elif (rotatingDirection == "counterclockwise"):
            futureOrientation = self.orientation + Parameters.unitAngleRad

        futureOrientation = futureOrientation % 6.28319

        futureObject = Agent(self.team, self.position, math.degrees(futureOrientation), self.tangible, self.controllable)
        futureObject.id = self.id
        intersect = utils.intersectionWithGridObjects(self.belongingGrid, futureObject)
        del(futureObject)

        if (intersect[0] == 0 or intersect[0] == 5):
            self.orientation = futureOrientation
            utils.computeHitbox(self, Parameters.agentSide1, Parameters.agentSide2)
            onGround = 0
            for vertex in self.getVertList():
                for ramp in self.belongingGrid.getRampList():
                    if (utils.pointInsidePolygon(vertex, ramp.vertex1, ramp.vertex2, ramp.vertex3, ramp.vertex4)):
                        onGround += 1
            if (onGround > 1):
                self.onRamp = True
            else:
                self.onRamp = False

        elif ((intersect[0] == 3) and (self.onRamp == True)):
            self.orientation = futureOrientation
            utils.computeHitbox(self, Parameters.agentSide1, Parameters.agentSide2)

        elif (intersect[0] == 4):
            if (intersect[1].held == True):
                self.orientation = futureOrientation
                utils.computeHitbox(self, Parameters.agentSide1, Parameters.agentSide2)
            else:
                return intersect
        else:
            return

    def lock(self, object):
        if (object.moveable == True):
            object.moveable = False
            object.lockedTeam = self.team

    def unlock(self, object):
        if ((object.moveable == False) and (object.lockedTeam == self.team)):
            object.moveable = True

    def hold(self, object):
        object.held = True

        # Angle described in thesis
        alfa = (math.atan2(object.position[1] - self.position[1], object.position[0] - self.position[0]))

        if (alfa >= self.orientation):
            self.constantBeta = [alfa - self.orientation, 0]
        else:
            self.constantBeta = [self.orientation - alfa, 1]

        if (object.orientation >= self.orientation):
            self.constantGamma = [object.orientation - self.orientation , 0]
        else:
            self.constantGamma = [self.orientation - object.orientation, 1]

        self.constantDistanceHeldObject = utils.distanceTwoPoints(self.position, object.position)

    def release(self, object):
        object.held = False
        self.constantOrientationHeldObject = None
        self.constantDistanceHeldObject = None

    def getVertList(self):
        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]
        return self.vertsList

    def getLidarSensor(self):
        lidarLines = []
        for lidarIndex in range(0, 30):
            lidarLines.append(((self.position),
                               ((Parameters.lidarLength * math.cos(math.radians(12.0 * lidarIndex))),
                                (Parameters.lidarLength * math.sin(math.radians(12.0 * lidarIndex))))))

        self.lidarSensor = lidarLines

        return self.lidarSensor