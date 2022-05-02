from multiagent_survival.gym_hidenseek.modules.box import *
from multiagent_survival.gym_hidenseek.modules.ramp import *
from multiagent_survival.gym_hidenseek.modules.agent import *
import multiagent_survival.gym_hidenseek.utils.parameters as parameters

import math
import numpy as np
import random


# Distance between point1 and point2.
def distanceTwoPoints(point1, point2):
    distance = math.sqrt((((point2[0] - point1[0])) * ((point2[0] - point1[0]))) + (((point2[1] - point1[1])) * ((point2[1] - point1[1]))))
    return round(distance, 2)

# Computation of slope and intercept of a line.
def computeSlopeAndIntercept(point1, point2):
    if (point1[0] == point2[0]):
        point2 = (point2[0] + 0.00000000001, point2[1])

    if (point1[1] == point2[1]):
        slope = 0.0
    else:
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])

    intercept = ((point2[0] * point1[1]) - (point1[0] * point2[1])) / (point2[0] - point1[0])
    return slope, intercept

# Check if point point is on line (point1Line, point2Line)
def pointInsideSegment(point, point1Line, point2Line):

    xList = [point1Line[0], point2Line[0]]
    yList = [point1Line[1], point2Line[1]]
    xList.sort()
    yList.sort()
    if (point[0] >= xList[0] and point[0] <= xList[1]):
        if (point[1] >= yList[0] and point[1] <= yList[1]):
            slope, intercept = computeSlopeAndIntercept(point1Line, point2Line)
            if (point1Line[0] == point2Line[0]):
                if (point[0] == point1Line[0]):
                    return True
                else:
                    return False
            elif (point[1] == round(((slope * point[0]) + intercept)), 2):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

# Check if point point in inside the circular sector
def pointInsideCircularSector(point, agent, figureType):

    result = None

    slope, _ = computeSlopeAndIntercept(point, agent.position)
    pointAngle = (math.atan2(point[1] - agent.position[1], point[0] - agent.position[0]))
    if (pointAngle < 0.0):
        pointAngle += 6.28319

    # maggiore = angleLimit1 = limite destro
    # minore = angleLimit2 = limite sinistro
    angleLimit1 = (agent.orientation - (parameters.angleVisualFieldRad / 2))
    angleLimit1 = angleLimit1 % 6.28319
    angleLimit2 = (agent.orientation + (parameters.angleVisualFieldRad / 2))
    angleLimit2 = angleLimit2 % 6.28319

    pointAngle = round(pointAngle, 3)
    angleLimit1 = round(angleLimit1, 3)
    angleLimit2 = round(angleLimit2, 3)

    if (angleLimit1 < angleLimit2):
        if ((pointAngle >= angleLimit1) and (pointAngle <= angleLimit2)):
            result = True
        else:
            result = False
    else:
        if ((pointAngle >= angleLimit1) and (pointAngle < 6.28319)):
            result = True
        elif ((pointAngle >= 0.0) and (pointAngle <= angleLimit2)):
            result = True
        else:
            result = False

    if (figureType == "circular"):
        return result
    if (figureType == "squared"):
        distancePointCenter = distanceTwoPoints(point, agent.position)
        if (distancePointCenter <= parameters.radiusVisualField):
            if (result == True):
                return True

    return False

# Check if point is inside the polygon (polygonPoint1, polygonPoint2, polygonPoint3, polygonPoint4).
# Computation with formula (yB-yA)(xC-xA) - (yC-yA)(xB-xA), to check the position of C with respect to segment A->B:
# if formula < 0: C is to the left of A->B. If this is true for all the polygon's segments, C is inside the polygon.
def pointInsidePolygon(point, polygonPoint1, polygonPoint2, polygonPoint3, polygonPoint4):
    if ((((polygonPoint2[1] - polygonPoint1[1]) * (point[0] - polygonPoint1[0])) - ((point[1] - polygonPoint1[1]) * (polygonPoint2[0] - polygonPoint1[0]))) <= 0):
        if ((((polygonPoint3[1] - polygonPoint2[1]) * (point[0] - polygonPoint2[0])) - ((point[1] - polygonPoint2[1]) * (polygonPoint3[0] - polygonPoint2[0]))) <= 0):
            if ((((polygonPoint4[1] - polygonPoint3[1]) * (point[0] - polygonPoint3[0])) - ((point[1] - polygonPoint3[1]) * (polygonPoint4[0] - polygonPoint3[0]))) <= 0):
                if ((((polygonPoint1[1] - polygonPoint4[1]) * (point[0] - polygonPoint4[0])) - ((point[1] - polygonPoint4[1]) * (polygonPoint1[0] - polygonPoint4[0]))) <= 0):
                    return True
    return False

# Check if point is inside the cylinder (with center center and radius radius).
def pointInsideCylinder(point, cylinder):
    if (distanceTwoPoints(point, cylinder.center) <= cylinder.radius):
        return True
    else:
        return False

# Intersection between line (point1Line1, point2Line1) and line (point1Line2, point2Line2).
def intersectionTwoLines(point1Line1, point2Line1, point1Line2, point2Line2):

    pointList = [point1Line1, point2Line1, point1Line2, point2Line2]

    for indexPoint in range(0, len(pointList)):
        for indexPoint2 in range(indexPoint + 1, len(pointList)):
            if ((pointList[indexPoint][0] == pointList[indexPoint2][0]) and (pointList[indexPoint][1] == pointList[indexPoint2][1])):
                intersectionPointX = pointList[indexPoint][0]
                intersectionPointY = pointList[indexPoint][1]
                return (intersectionPointX, intersectionPointY)

    if (point1Line1[0] == point2Line1[0]):
        slope2, intercept2 = computeSlopeAndIntercept(point1Line2, point2Line2)
        intersectionPointX = point1Line1[0]
        intersectionPointY = (slope2 * (intersectionPointX) + intercept2)

    elif (point1Line2[0] == point2Line2[0]):
        slope1, intercept1 = computeSlopeAndIntercept(point1Line1, point2Line1)
        intersectionPointX = point1Line2[0]
        intersectionPointY = (slope1 * (intersectionPointX) + intercept1)

    else:
        try:
            slope1, intercept1 = computeSlopeAndIntercept(point1Line1, point2Line1)
            slope2, intercept2 = computeSlopeAndIntercept(point1Line2, point2Line2)

            intersectionPointX = ((intercept2 - intercept1) / (slope1 - slope2))
            if (slope1 == 0.0):
                intersectionPointY = (slope1 * (intersectionPointX) + intercept1)
            elif (slope2 == 0.0):
                intersectionPointY = (slope2 * (intersectionPointX) + intercept2)
            else:
                intersectionPointY = (slope1 * (intersectionPointX) + intercept1)
        except:
            intersectionPointX = None
            intersectionPointY = None

            return (intersectionPointX, intersectionPointY)

    if (pointInsideSegment((intersectionPointX, intersectionPointY), point1Line1, point2Line1)):
        if (pointInsideSegment((intersectionPointX, intersectionPointY), point1Line2, point2Line2)):
            return (intersectionPointX, intersectionPointY)

    return (None, None)

# Intersection betweenn lines for vectorized version of Lidar sensor
def intersectionTwoLinesVector(lidarLines, environmentLines, objectList, currentAgent):

    lidarObjects = []

    for lidarLine in lidarLines:
        maxLineDistance = 999
        selectedIntersectionPoint = None
        object = None
        for index, environmentLine in enumerate(environmentLines):
            if (isinstance(objectList[index], Agent) and objectList[index].id == currentAgent.id):
                continue

            A1 = environmentLine[1][1] - environmentLine[0][1]
            B1 = environmentLine[0][0] - environmentLine[1][0]
            C1 = np.multiply(A1, environmentLine[0][0]) + np.multiply(B1, environmentLine[0][1])

            A2 = lidarLine[1][1] - lidarLine[0][1]
            B2 = lidarLine[0][0] - lidarLine[1][0]
            C2 = np.multiply(A2, lidarLine[0][0]) + np.multiply(B2, lidarLine[0][1])

            matrix = np.array([[A1, A2], [B1,B2]])
            det = np.linalg.det(matrix)
            intersectionPoint = (None, None)
            if (det != 0):
                x = np.divide((np.multiply(B2, C1) - np.multiply(B1, C2)), det)
                y = np.divide((np.multiply(A1, C2) - np.multiply(A2, C1)), det)
                if (pointInsideSegment((round(x, 2), round(y, 2)), environmentLine[0], environmentLine[1])):
                    if (pointInsideSegment((round(x, 2), round(y, 2)), lidarLine[0], lidarLine[1])):
                        intersectionPoint = (x, y)

            if (intersectionPoint != (None, None)):
                currentDistance = utils.distanceTwoPoints(currentAgent.position, intersectionPoint)
                if (currentDistance < maxLineDistance):
                    maxLineDistance = currentDistance
                    selectedIntersectionPoint = intersectionPoint
                    object = objectList[index]

        lidarObjects.append([object, selectedIntersectionPoint, maxLineDistance])

    return lidarObjects

# Intersection between cylinder (with center center and radius radius) and wall (startPoint, endPoint).
def intersectionCylinderWall(cylinder, wall):

    if (wall.startPoint[0] == wall.endPoint[0]):
        alfa = 1
        beta = -2 * cylinder.center[1]
        gamma = pow(wall.startPoint[0], 2) - (2 * cylinder.center[0] * wall.startPoint[0]) + pow(cylinder.center[0], 2) + pow(cylinder.center[1], 2) - pow(cylinder.radius, 2)

    else:
        slope, intercept = computeSlopeAndIntercept(wall.startPoint, wall.endPoint)

        alfa = pow(slope, 2) + 1
        beta = (2 * slope * intercept) + (-2 * cylinder.center[0]) + (-2 * cylinder.center[1] * slope)
        gamma = pow(intercept, 2) + (-2 * cylinder.center[1] * intercept) + pow(cylinder.center[0], 2) + pow(cylinder.center[1], 2) - pow(cylinder.radius, 2)

    delta = pow(beta, 2) - (4 * alfa * gamma)

    if (delta < 0):
        return False
    else:
        return True

# Intersection between cylinder (with center center and radius radius) and cylinder (with center center and radius radius).
def intersectionCylinderCylinder(cylinder1, cylinder2):
    centersDistance = distanceTwoPoints(cylinder1.center, cylinder2.center)

    if (centersDistance <= (cylinder1.radius + cylinder2.radius)):
        return True
    else:
        return False

# Intersection between circle with center circleCenter, radius circleRadius and line (point1, point2)
def intersectionCircleLine(circleCenter, circleRadius, point1, point2):

    if (point1[1] == point2[1]):
        alfa = 1
        beta = -2 * circleCenter[0]
        gamma = pow(point1[1], 2) - (2 * circleCenter[1] * point1[1]) + pow(circleCenter[0], 2) + pow(circleCenter[1], 2) - pow(circleRadius, 2)

        delta = pow(beta, 2) - 4 * alfa * gamma
        if (delta < 0):
            return ((None, None), (None, None))
        elif (delta == 0):
            x1 = (-beta / (2 * alfa))
            y1 = point1[1]
            if (pointInsideSegment((x1, y1), point1, point2) == False):
                return ((None, None), (None, None))
            else:
                return ((x1, y1), (None, None))
        elif (delta > 0):
            x1 = (- beta + math.sqrt(pow(beta, 2) - 4 * alfa * gamma)) / (2 * alfa)
            x2 = (- beta - math.sqrt(pow(beta, 2) - 4 * alfa * gamma)) / (2 * alfa)
            y1 = point1[1]
            y2 = point1[1]
            if (pointInsideSegment((x1, y1), point1, point2) == False):
                x1 = y1 = None
            if (pointInsideSegment((x2, y2), point1, point2) == False):
                x2 = y2 = None
            return ((x1, y1), (x2, y2))

    elif (point1[0] == point2[0]):
        alfa = 1
        beta = -2 * circleCenter[1]
        gamma = pow(point1[0], 2) - (2 * circleCenter[0] * point1[0]) + pow(circleCenter[0], 2) + pow(circleCenter[1], 2) - pow(circleRadius, 2)

        delta = pow(beta, 2) - 4 * alfa * gamma
        if (delta < 0):
            return ((None, None), (None, None))
        elif (delta == 0):
            x1 = (-beta / (2 * alfa))
            y1 = point1[0]
            if (pointInsideSegment((x1, y1), point1, point2) == False):
                return ((None, None), (None, None))
            else:
                return ((x1, y1), (None, None))
        elif (delta > 0):
            x1 = point1[0]
            x2 = point1[1]
            y1 = (- beta + math.sqrt(pow(beta, 2) - 4 * alfa * gamma)) / (2 * alfa)
            y2 = (- beta - math.sqrt(pow(beta, 2) - 4 * alfa * gamma)) / (2 * alfa)
            if (pointInsideSegment((x1, y1), point1, point2) == False):
                x1 = y1 = None
            if (pointInsideSegment((x2, y2), point1, point2) == False):
                x2 = y2 = None
            return ((x1, y1), (x2, y2))
    else:
        slope, intercept = computeSlopeAndIntercept(point1, point2)

        alfa = 1 + pow(slope, 2)
        beta = (2 * slope * intercept) - (2 * circleCenter[0]) - (2 * circleCenter[1] * slope)
        gamma = pow(intercept, 2) - (2 * circleCenter[1] * intercept) + pow(circleCenter[0], 2) + pow(circleCenter[1], 2) - pow(circleRadius, 2)

        delta = pow(beta, 2) - 4 * alfa * gamma
        if (delta < 0):
            return ((None, None), (None, None))
        elif (delta == 0):
            x1 = (-beta / (2 * alfa))
            y1 = (slope * x1) + intercept
            if (pointInsideSegment((x1, y1), point1, point2) == False):
                return ((None, None), (None, None))
            else:
                return ((x1, y1), (None, None))
        elif (delta > 0):
            x1 = (- beta + math.sqrt(pow(beta, 2) - 4 * alfa * gamma)) / (2 * alfa)
            x2 = (- beta - math.sqrt(pow(beta, 2) - 4 * alfa * gamma)) / (2 * alfa)
            y1 = (slope * x1) + intercept
            y2 = (slope * x2) + intercept

            if (pointInsideSegment((x1, y1), point1, point2) == False):
                x1 = y1 = None
            if (pointInsideSegment((x2, y2), point1, point2) == False):
                x2 = y2 = None
            return ((x1, y1), (x2, y2))

# General intersection check between object and all the objects on the grid.
# return values: 0 - object allowed to be added to the grid
#                1 - intangible object, allowed to be added to the grid
#                2 - intangible object, not allowed to be added to the grid
#                3 - tangible object, not allowed to be added to the grid due intersection with wall
#                4 - tangible object, not allowed to be added to the grid due intersection with box
#                5 - tangible object, not allowed to be added to the grid due intersection with ramp
#                6 - tangible object, not allowed to be added to the grid due intersection with cylinder
#                7 - tangible object, not allowed to be added to the grid due intersection with agent
def intersectionWithGridObjects(grid, object):

    if (object.__class__.__name__ == "Box" or
        object.__class__.__name__ == "Ramp" or
        object.__class__.__name__ == "Agent"):
        if (object.tangible == False):
            for objectVert in object.getVertList():
                if (pointInsidePolygon(objectVert, grid.corners[0], grid.corners[1], grid.corners[2], grid.corners[3])):
                    return 1
            return 2

        elif (object.tangible == True):

            objectVerts = object.getVertList()

            for wall in grid.getWallList():
                wallVerts = wall.getVertList()
                for vertex in object.getVertList():
                    if (pointInsidePolygon(vertex, wallVerts[0], wallVerts[1], wallVerts[2], wallVerts[3])):
                        return (3, wall, vertex)
                for wallVertex in wallVerts:
                    if (pointInsidePolygon(wallVertex, objectVerts[0], objectVerts[1], objectVerts[2], objectVerts[3])):
                        return (3, wall, wallVertex)
            for box in grid.getBoxList():
                if (box.id == object.id):
                    continue
                boxVerts = box.getVertList()
                for vertex in object.getVertList():
                    if (pointInsidePolygon(vertex, boxVerts[0], boxVerts[1], boxVerts[2], boxVerts[3])):
                        grid.tempPoint = vertex
                        return (4, box, vertex)
                for boxVertex in boxVerts:
                    if (pointInsidePolygon(boxVertex, objectVerts[0], objectVerts[1], objectVerts[2], objectVerts[3])):
                        grid.tempPoint = boxVertex
                        return (4, box, boxVertex)
            for ramp in grid.getRampList():
                if (ramp.id == object.id):
                    continue
                rampVerts = ramp.getVertList()
                for vertex in object.getVertList():
                    if (pointInsidePolygon(vertex, rampVerts[0], rampVerts[1], rampVerts[2], rampVerts[3])):
                        return (5, ramp, vertex)
                for rampVertex in rampVerts:
                    if (pointInsidePolygon(rampVertex, objectVerts[0], objectVerts[1], objectVerts[2], objectVerts[3])):
                        return (5, ramp, vertex)
            for cylinder in grid.getCylinderList():
                if (cylinder.id == object.id):
                    continue
                for vertex in object.vertsList:
                    if (pointInsideCylinder(vertex, cylinder)):
                        return (6, cylinder, vertex)
                for vertexIndex in range(0, len(objectVerts)):
                    secondVertex = vertexIndex + 1
                    if (secondVertex == len(objectVerts)):
                        secondVertex = 0
                    intersectionPoints = intersectionCircleLine(cylinder.center, cylinder.radius, objectVerts[vertexIndex], objectVerts[secondVertex])
                    for intersect in intersectionPoints:
                        if (intersect != (None, None)):
                            return (6, cylinder, intersect)
            for agent in grid.getAgentList():
                if (agent.id == object.id):
                    continue
                agentVerts = agent.getVertList()
                for vertex in object.getVertList():
                    if (pointInsidePolygon(vertex, agentVerts[0], agentVerts[1], agentVerts[2], agentVerts[3])):
                        return (7, agent, vertex)
                for agentVertex in agentVerts:
                    if (pointInsidePolygon(agentVertex, objectVerts[0], objectVerts[1], objectVerts[2], objectVerts[3])):
                        return (7, agent, agentVertex)
        return (0, None, None)

    elif (object.__class__.__name__ == "Cylinder"):
        if (object.tangible == False):
            for wall in grid.getWallList():
                cylinderWallIntersect = intersectionCylinderWall(object, wall)
                if (pointInsidePolygon(object.center, grid.corners[0], grid.corners[1], grid.corners[2], grid.corners[3])):
                    if (cylinderWallIntersect == True):
                        return 1
            return 2

        elif (object.tangible == True):

            for wall in grid.getWallList():
                wallVerts = wall.getVertList()
                for wallVertex in wallVerts:
                    if (pointInsideCylinder(wallVertex, object)):
                        return (3, wall, wallVertex)
                for wallVertexIndex in range(0, len(wallVerts)):
                    wallSecondVertex = wallVertexIndex + 1
                    if (wallSecondVertex == len(wallVerts)):
                        wallSecondVertex = 0
                    intersectionPoints = intersectionCircleLine(object.center, object.radius, wallVerts[wallVertexIndex], wallVerts[wallSecondVertex])
                    for intersect in intersectionPoints:
                        if (intersect != (None, None)):
                            return (3, wall, intersect)

            for box in grid.getBoxList():
                if (box.id == object.id):
                    continue
                boxVerts = box.getVertList()
                for boxVertex in boxVerts:
                    if (pointInsideCylinder(boxVertex, object)):
                        return (4, box, boxVertex)
                for boxVertexIndex in range(0, len(boxVerts)):
                    boxSecondVertex = boxVertexIndex + 1
                    if (boxSecondVertex == len(boxVerts)):
                        boxSecondVertex = 0
                    intersectionPoints = intersectionCircleLine(object.center, object.radius, boxVerts[boxVertexIndex], boxVerts[boxSecondVertex])
                    for intersect in intersectionPoints:
                        if (intersect != (None, None)):
                            return (4, box, intersect)

            for ramp in grid.getRampList():
                if (ramp.id == object.id):
                    continue
                rampVerts = ramp.getVertList()
                for rampVertex in rampVerts:
                    if (pointInsideCylinder(rampVertex, object)):
                        return (5, ramp, rampVertex)
                for rampVertexIndex in range(0, len(rampVerts)):
                    rampSecondVertex = rampVertexIndex + 1
                    if (rampSecondVertex == len(rampVerts)):
                        rampSecondVertex = 0
                    intersectionPoints = intersectionCircleLine(object.center, object.radius, rampVerts[rampVertexIndex], rampVerts[rampSecondVertex])
                    for intersect in intersectionPoints:
                        if (intersect != (None, None)):
                            return (5, ramp, intersect)

            for cylinder in grid.getCylinderList():
                if (cylinder.id == object.id):
                    continue
                if (intersectionCylinderCylinder(object, cylinder)):
                    return (6, cylinder, (None, None))

            for agent in grid.getAgentList():
                if (agent.id == object.id):
                    continue
                agentVerts = agent.getVertList()
                for agentVertex in agentVerts:
                    if (pointInsideCylinder(agentVertex, object)):
                        return (7, agent, agentVertex)
                for agentVertexIndex in range(0, len(agentVerts)):
                    agentSecondVertex = agentVertexIndex + 1
                    if (agentSecondVertex == len(agentVerts)):
                        agentSecondVertex = 0
                    intersectionPoints = intersectionCircleLine(object.center, object.radius, agentVerts[agentVertexIndex], agentVerts[agentSecondVertex])
                    for intersect in intersectionPoints:
                        if (intersect != (None, None)):
                            return (7, agent, intersect)
        return 0

# Computation of objects' hitboxes vertices
def computeHitbox(object, side1, side2):
    # 1.5708 radian = 1 degree

    # lower left point, point C in the drawing
    object.vertex1 = ((object.position[0] - (side1 / 2) * math.cos(object.orientation) + (side2 / 2) * math.cos(object.orientation - 1.5708)),
                      (object.position[1] - (side1 / 2) * math.sin(object.orientation) + (side2 / 2) * math.sin(object.orientation - 1.5708)))
    object.vertex1 = (round(object.vertex1[0], 2), round(object.vertex1[1], 2))

    # lower right point, point D in the drawing
    object.vertex2 = ((object.position[0] + (side1 / 2) * math.cos(object.orientation) + (side2 / 2) * math.cos(object.orientation - 1.5708)),
                      (object.position[1] + (side1 / 2) * math.sin(object.orientation) + (side2 / 2) * math.sin(object.orientation - 1.5708)))
    object.vertex2 = (round(object.vertex2[0], 2), round(object.vertex2[1], 2))

    # upper right point, point A in the drawing
    object.vertex3 = ((object.position[0] + (side1 / 2) * math.cos(object.orientation) - (side2 / 2) * math.sin(object.orientation)),
                      (object.position[1] + (side1 / 2) * math.sin(object.orientation) + (side2 / 2) * math.cos(object.orientation)))
    object.vertex3 = (round(object.vertex3[0], 2), round(object.vertex3[1], 2))

    # upper left point, point B in the drawing
    object.vertex4 = ((object.position[0] - (side1 / 2) * math.cos(object.orientation) - (side2 / 2) * math.sin(object.orientation)),
                      (object.position[1] - (side1 / 2) * math.sin(object.orientation) + (side2 / 2) * math.cos(object.orientation)))
    object.vertex4 = (round(object.vertex4[0], 2), round(object.vertex4[1], 2))

# Computation of walls' hitboxes vertices
def computeWallHitbox(wall):
    # 1.5708 radian = 1 degree

    slope, _ = computeSlopeAndIntercept(wall.startPoint, wall.endPoint)
    wallOrientation = math.atan(slope)

    # lower left point, point C in the drawing
    pointC = ((wall.startPoint[0] + (parameters.wallThickness / 2) * math.cos(wallOrientation - 1.5708)),
              (wall.startPoint[1] + (parameters.wallThickness / 2) * math.sin(wallOrientation - 1.5708)))
    pointC = (round(pointC[0], 2), round(pointC[1], 2))

    # lower right point, point D in the drawing
    pointD = ((wall.endPoint[0] + (parameters.wallThickness / 2) * math.cos(wallOrientation - 1.5708)),
              (wall.endPoint[1] + (parameters.wallThickness / 2) * math.sin(wallOrientation - 1.5708)))
    pointD = (round(pointD[0], 2), round(pointD[1], 2))

    # upper right point, point A in the drawing
    pointA = ((wall.endPoint[0] - (parameters.wallThickness / 2) * math.sin(wallOrientation)),
              (wall.endPoint[1] + (parameters.wallThickness / 2) * math.cos(wallOrientation)))
    pointA = (round(pointA[0], 2), round(pointA[1], 2))

    # upper left point, point B in the drawing
    pointB = ((wall.startPoint[0] - (parameters.wallThickness / 2) * math.sin(wallOrientation)),
              (wall.startPoint[1] + (parameters.wallThickness / 2) * math.cos(wallOrientation)))
    pointB = (round(pointB[0], 2), round(pointB[1], 2))

    return (pointA, pointB, pointC, pointD)

# Motion of held objects
def moveTranslateObject(agent, object):

    if (agent.constantBeta == None):
        anglePosition = agent.orientation
    elif (agent.constantBeta[1] == 0):
        anglePosition = agent.constantBeta[0] + agent.orientation
    elif (agent.constantBeta[1] == 1):
        anglePosition = agent.orientation - agent.constantBeta[0]

    if (agent.constantDistanceHeldObject == None):
        agent.constantDistanceHeldObject = 0.0
    futurePosition = (agent.position[0] + (agent.constantDistanceHeldObject * math.cos(anglePosition)),
                      agent.position[1] + (agent.constantDistanceHeldObject * math.sin(anglePosition)))

    if (agent.constantGamma == None):
        futureOrientation = agent.orientation
    elif (agent.constantGamma[1] == 0):
        futureOrientation = agent.constantGamma[0] + agent.orientation
    elif (agent.constantGamma[1] == 1):
        futureOrientation = agent.orientation - agent.constantGamma[0]


    if (isinstance(object, Box)):
        futureObject = Box(futurePosition, futureOrientation, object.type, object.tangible, object.moveable, object.lockedTeam)
        futureObject.id = object.id
        intersect = intersectionWithGridObjects(agent.belongingGrid, futureObject)
        del(futureObject)
        if ((intersect[0] == 0) or ((intersect[0] == 7) and (intersect[1] == agent))):
            object.position = futurePosition
            object.orientation = futureOrientation
            if (object.type == "squared"):
                computeHitbox(object, parameters.squaredBoxSize, parameters.squaredBoxSize)
            elif (object.type == "elongated"):
                computeHitbox(object, parameters.elongatedBoxShortSide, parameters.elongatedBoxLongSide)

        else:
            return 1

    elif (isinstance(object, Ramp)):
        futureObject = Ramp(futurePosition, futureOrientation, object.tangible, object.moveable, object.lockedTeam)
        futureObject.id = object.id
        intersect = intersectionWithGridObjects(agent.belongingGrid, futureObject)
        del(futureObject)
        if (intersect[0] == 0):
            object.position = futurePosition
            object.orientation = futureOrientation
            computeHitbox(object, parameters.rampSize, parameters.rampSize)

        else:
            return 1

# Motion of hit objects
def moveObjectDueHit(agent, object, intersectionPoint):

    angleMovement = (math.atan2(object.position[1] - agent.position[1], object.position[0] - agent.position[0]))
    if (angleMovement < 0.0):
        angleMovement += 6.28319

    angleIntersection = (math.atan2(intersectionPoint[1] - object.position[1], intersectionPoint[0] - object.position[0]))
    if (angleIntersection < 0.0):
        angleIntersection += 6.28319

    angleLimits = [((object.orientation) % 6.28319),
                   ((object.orientation + 0.785398) % 6.28319),
                   ((object.orientation + 1.5708) % 6.28319),
                   ((object.orientation + 2.35619) % 6.28319),
                   ((object.orientation + 3.14159) % 6.28319),
                   ((object.orientation + 3.92699) % 6.28319),
                   ((object.orientation + 4.71239) % 6.28319),
                   ((object.orientation + 5.49779) % 6.28319)]

    for angleLimitIndex in range(0, len(angleLimits)):
        secondAngle = angleLimitIndex + 1
        if (secondAngle == len(angleLimits)):
            secondAngle = 0
        if (angleLimits[angleLimitIndex] < angleLimits[secondAngle]):
            if ((angleIntersection > angleLimits[angleLimitIndex]) and (angleIntersection <= angleLimits[secondAngle])):
                if (angleLimitIndex % 2 == 0):
                    futureOrientation = object.orientation + parameters.unitAngleRad
                else:
                    futureOrientation = object.orientation - parameters.unitAngleRad
        else:
            if ((angleIntersection >= angleLimits[angleLimitIndex]) and (angleIntersection < 6.28319)):
                if (angleLimitIndex % 2 == 0):
                    futureOrientation = object.orientation + parameters.unitAngleRad
                else:
                    futureOrientation = object.orientation - parameters.unitAngleRad
            elif ((angleIntersection >= 0.0) and (angleIntersection <= angleLimits[secondAngle])):
                if (angleLimitIndex % 2 == 0):
                    futureOrientation = object.orientation + parameters.unitAngleRad
                else:
                    futureOrientation = object.orientation - parameters.unitAngleRad

    futurePosition = (object.position[0] + (parameters.unitMotion * math.cos(angleMovement)),
                      object.position[1] + (parameters.unitMotion * math.sin(angleMovement)))

    if (isinstance(object, Box)):
        futureObject = Box(futurePosition, futureOrientation, object.type, object.tangible, object.moveable, object.lockedTeam)
        futureObject.id = object.id
        intersect = intersectionWithGridObjects(agent.belongingGrid, futureObject)
        del(futureObject)
        if (intersect[0] == 0):
            object.position = futurePosition
            object.orientation = futureOrientation
            if (object.type == "squared"):
                computeHitbox(object, parameters.squaredBoxSize, parameters.squaredBoxSize)
            elif (object.type == "elongated"):
                computeHitbox(object, parameters.elongatedBoxShortSide, parameters.elongatedBoxLongSide)
        else:
            return 1

# Print of every object in the grid
def printObjectList(grid):
    print("gridSize", grid.gridSize)
    print("------------------------------")
    print("wallAmount", grid.wallAmount)
    indexWall = 0
    for wall in grid.getWallList():
        print("--- wall n.", indexWall, " : ", wall.getWallEndPoints())
        indexWall += 1
    print("------------------------------")

    print("□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□")
    print("boxAmount", grid.boxAmount)
    indexBox = 0
    for box in grid.getBoxList():
        print("--- box n.", indexBox, " : ", box.getVertList())
        indexBox += 1
    print("□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□")

    print("◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿")
    print("rampAmount", grid.rampAmount)
    indexRamp = 0
    for ramp in grid.getRampList():
        print("--- ramp n.", indexRamp, " : ", ramp.getVertList())
        indexRamp += 1
    print("◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿◿")

    print("○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○")
    print("cylinderAmount", grid.cylinderAmount)
    indexCylinder = 0
    for cylinder in grid.getCylinderList():
        print("--- cylinder n.", indexCylinder, " : center ", cylinder.center, " radius ", cylinder.radius)
        indexCylinder += 1
    print("○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○")

    print("♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙")
    print("agentAmount", grid.agentAmount)
    indexAgent = 0
    for agent in grid.getAgentList():
        print("--- agent n.", indexAgent, " : verts ", agent.getVertList(), " orientation (rad)", round(agent.orientation, 5), " (deg) ", round(math.degrees(agent.orientation), 2))
        indexAgent += 1
    print("♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙♙")

# Random computation of position and size of a door
def randomDoorGenerator():
    # randomDoorPosition: 0 - begin
    #                     1 - middle
    #                     2 - end
    # randomDoorSize: 0 - small
    #                 1 - large
    randomDoorPosition = random.randint(0, 2)
    randomDoorSize = random.randint(0, 1)

    doorPosition = None
    doorSize = None

    if (randomDoorPosition == 0):
        doorPosition = "begin"
    elif (randomDoorPosition == 1):
        doorPosition = "middle"
    elif (randomDoorPosition == 2):
        doorPosition = "end"

    if (randomDoorSize == 0):
        doorSize = "small"
    elif (randomDoorSize == 1):
        doorSize = "large"

    return doorPosition, doorSize

