from multiagent_survival.gym_hidenseek.utils.utils import *

smallDoorSize = 2
largeDoorSize = smallDoorSize * 2

class Wall():

    startPoint = None
    endPoint = None
    wallLength = None

    vertex1 = None
    vertex2 = None
    vertex3 = None
    vertex4 = None

    def __init__(self, start, end):
        self.startPoint = start
        self.endPoint = end
        self.wallLength = distanceTwoPoints(self.startPoint, self.endPoint)

        vertices = computeWallHitbox(self)
        self.vertex1 = vertices[0]
        self.vertex2 = vertices[1]
        self.vertex3 = vertices[2]
        self.vertex4 = vertices[3]

    # doorPosition : position of the door in the wall. Can be:
    #            begin - door at the wall begin
    #            middle - door at the wall middle
    #            end - door at the wall end
    # doorSize : small - small door with size 2.0
    #            large - large door with size 4.0
    def computeDoorCoords(self, doorPosition, doorSize):
        if (doorSize == "small"):
            doorSizeNumber = smallDoorSize
        elif (doorSize == "large"):
            doorSizeNumber = largeDoorSize

        if (doorPosition == "begin"):
            doorPositionNumber = 0.8
        elif (doorPosition == "middle"):
            doorPositionNumber = 0.5
        elif (doorPosition == "end"):
            doorPositionNumber = 0.2

        percentage = doorSizeNumber * (100 / self.wallLength)

        percentage = percentage / 100

        doorStartPointX = self.startPoint[0] * (doorPositionNumber + (percentage / 2)) + self.endPoint[0] * (1 - (doorPositionNumber + (percentage / 2)))
        doorStartPointY = self.startPoint[1] * (doorPositionNumber + (percentage / 2)) + self.endPoint[1] * (1 - (doorPositionNumber + (percentage / 2)))
        doorEndPointX = self.startPoint[0] * (doorPositionNumber - (percentage / 2)) + self.endPoint[0] * (1 - (doorPositionNumber - (percentage / 2)))
        doorEndPointY = self.startPoint[1] * (doorPositionNumber - (percentage / 2)) + self.endPoint[1] * (1 - (doorPositionNumber - (percentage / 2)))
        return ((round(doorStartPointX, 2), round(doorStartPointY, 2)), (round(doorEndPointX, 2), round(doorEndPointY, 2)))

    def getWallEndPoints(self):
        return (self.startPoint, self.endPoint)

    def getVertList(self):
        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]
        return self.vertsList