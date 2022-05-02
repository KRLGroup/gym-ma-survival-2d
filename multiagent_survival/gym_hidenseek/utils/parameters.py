# Parameters

# Grid size
gridSize = 15.0

# Grid tile size
tileSize = 1

# Wall thickness
wallThickness = 1.0

# Squared box side dimension
squaredBoxSize = 1.0

# Elongated box:
#       elongatedBoxLongSide - long side
#       elongatedBoxShortSide - short side
elongatedBoxLongSide = 6.0
elongatedBoxShortSide = 1.0

# Ramp side dimension
rampSize = 1.5

# Agent hitbox
#       agentSide1 - agent side 1
#       agentSide2 - agent side 2
agentSide1 = 1.0
agentSide2 = 1.0

# Amount of movement performed per step
unitMotion = 0.2

# Amount of angle rotated per step, must be expressed in degrees
unitAngle = 5.0
unitAngleRad = (unitAngle * 3.1415926535) / 180.0

# Window's sizes:
windowHeight = gridSize
windowWidth = gridSize

# Colors:
# All the colors in matplotlib are expressed as RGB in [0.0-1.0] range
gridTileColorRGB = [220.0, 220.0, 220.0]
gridTileColorPlot = [gridTileColorRGB[0] / 255.0, gridTileColorRGB[1] / 255.0, gridTileColorRGB[2] / 255.0]

gridLineColorRGB = [160.0, 160.0, 160.0]
gridLineColorPlot = [gridLineColorRGB[0] / 255.0, gridLineColorRGB[1] / 255.0, gridLineColorRGB[2] / 255.0]

wallColorRGB = [60.0, 60.0, 60.0]
wallColorPlot = [wallColorRGB[0] / 255.0, wallColorRGB[1] / 255.0, wallColorRGB[2] / 255.0]

hiderColorRGB = [90.0, 255.0, 255.0]
hiderColorPlot = [hiderColorRGB[0] / 255.0, hiderColorRGB[1] / 255.0, hiderColorRGB[2] / 255.0]

hiderArrowColorRGB = [0.0, 220.0, 220.0]
hiderArrowColorPlot = [hiderArrowColorRGB[0] / 255.0, hiderArrowColorRGB[1] / 255.0, hiderArrowColorRGB[2] / 255.0]

seekerColorRGB = [255.0, 50.0, 50.0]
seekerColorPlot = [seekerColorRGB[0] / 255.0, seekerColorRGB[1] / 255.0, seekerColorRGB[2] / 255.0]

seekerArrowColorRGB = [190.0, 30.0, 30.0]
seekerArrowColorPlot = [seekerArrowColorRGB[0] / 255.0, seekerArrowColorRGB[1] / 255.0, seekerArrowColorRGB[2] / 255.0]

boxColorRGB = [255.0, 255.0, 50.0]
boxColorPlot = [boxColorRGB[0] / 255.0, boxColorRGB[1] / 255.0, boxColorRGB[2] / 255.0]

boxLockedColorRGB = [200.0, 180.0, 0.0]
boxLockedColorPlot = [boxLockedColorRGB[0] / 255.0, boxLockedColorRGB[1] / 255.0, boxLockedColorRGB[2] / 255.0]

boxArrowColorRGB = [200.0, 200.0, 30.0]
boxArrowColorPlot = [boxArrowColorRGB[0] / 255.0, boxArrowColorRGB[1] / 255.0, boxArrowColorRGB[2] / 255.0]

cylinderColorRGB = [0.0, 0.0, 255.0]
cylinderColorPlot = [cylinderColorRGB[0] / 255.0, cylinderColorRGB[1] / 255.0, cylinderColorRGB[2] / 255.0]

rampColorRGB = [0.0, 255.0, 30.0]
rampColorPlot = [rampColorRGB[0] / 255.0, rampColorRGB[1] / 255.0, rampColorRGB[2] / 255.0]

rampLockedColorRGB = [0.0, 200.0, 20.0]
rampLockedColorPlot = [rampLockedColorRGB[0] / 255.0, rampLockedColorRGB[1] / 255.0, rampLockedColorRGB[2] / 255.0]

rampArrowColorRGB = [0.0, 210.0, 30.0]
rampArrowColorPlot = [rampArrowColorRGB[0] / 255.0, rampArrowColorRGB[1] / 255.0, rampArrowColorRGB[2] / 255.0]

hiderObservationWedgeColorRGB = [200.0, 255.0, 250.0]
hiderObservationWedgeColorPlot = [hiderObservationWedgeColorRGB[0] / 255.0,
                                  hiderObservationWedgeColorRGB[1] / 255.0,
                                  hiderObservationWedgeColorRGB[2] / 255.0]

seekerObservationWedgeColorRGB = [255.0, 150.0, 150.0]
seekerObservationWedgeColorPlot = [seekerObservationWedgeColorRGB[0] / 255.0,
                                   seekerObservationWedgeColorRGB[1] / 255.0,
                                   seekerObservationWedgeColorRGB[2] / 255.0]

# Agent visual field parameters
angleVisualField = 135.0
angleVisualFieldRad = (angleVisualField * 3.1415926535) / 180.0
radiusVisualField = 1.8

# Lidar sensor
lidarLength = gridSize * 50

# Render args
withFieldOfView = False
withLidar = False

# Activate only if the selected environment is Sequential Lock
withNumbers = False
renderArgs = [withFieldOfView, withLidar, withNumbers]