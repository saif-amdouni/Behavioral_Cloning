"""

file containing configuration of the project

"""
key_map = {
    'Z'      : [1, 0, 0],
    'Q'      : [0, 1, 0],
    'D'      : [0, 0, 1],

}

# Directory #
training_dir = "training"  # The folder containing training data
notebooks_dir = "notebooks"  # the folder containing noteBooks
models_dir = "models"  # the folder that stores trained models

# Files #
file_name = "training"  # training data file

# Var #
height = 400
width = 650
mon = {'top': 320, 'left': 0, 'width': width, 'height': height}  # monitor
Count_Down = 5  # count-down length
training_pack = 500  # how much training data to store every time
trainingImgWidth = 300
trainingImgHeight = 300

# line detection

threshold = 0.1
minLineLength = 100
maxLineGap = 50


