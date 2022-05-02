class Cylinder():

    id = None
    belongingGrid = None

    tangible = True

    center = None
    radius = None

    def __init__(self, center, radius, tangible):
        self.tangible = tangible

        self.center = center
        self.radius = radius