from multiagent_survival.gym_hidenseek.utils import utils
import multiagent_survival.gym_hidenseek.utils.parameters as Parameters
import math

class Box():

    id = None
    belongingGrid = None

    position = None
    orientation = None
    type = None
    tangible = True
    moveable = True
    held = False

    lockedTeam = None

    side1 = None
    side2 = None

    vertex1 = None
    vertex2 = None
    vertex3 = None
    vertex4 = None

    vertsList = []

    constAngle = None

    def __init__(self, position, orientation, type, tangible, moveable, lockedTeam):
        self.position = position
        self.orientation = math.radians(orientation)
        self.type = type
        self.tangible = tangible
        self.moveable = moveable
        self.lockedTeam = lockedTeam

        if (type == "squared"):
            self.side1 = Parameters.squaredBoxSize
            self.side2 = Parameters.squaredBoxSize
            utils.computeHitbox(self, self.side1, self.side2)
        elif (type == "elongated"):
            self.side1 = Parameters.elongatedBoxLongSide
            self.side2 = Parameters.elongatedBoxShortSide
            utils.computeHitbox(self, Parameters.elongatedBoxLongSide, Parameters.elongatedBoxShortSide)

        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]

    def getVertList(self):
        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]
        return self.vertsList