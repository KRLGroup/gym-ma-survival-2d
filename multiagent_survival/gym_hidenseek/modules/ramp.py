from multiagent_survival.gym_hidenseek.utils import utils
import multiagent_survival.gym_hidenseek.utils.parameters as Parameters
import math

class Ramp():

    id = None
    belongingGrid = None

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

    def __init__(self, position, orientation, tangible, moveable, lockedTeam):
        self.position = position
        self.orientation = math.radians(orientation)
        self.tangible = tangible
        self.moveable = moveable

        self.lockedTeam = lockedTeam

        self.side1 = Parameters.rampSize
        self.side2 = Parameters.rampSize

        utils.computeHitbox(self, self.side1, self.side2)

        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]

    def getVertList(self):
        self.vertsList = [self.vertex1, self.vertex2, self.vertex3, self.vertex4]
        return self.vertsList