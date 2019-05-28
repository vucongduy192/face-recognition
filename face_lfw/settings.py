import os
root = 'dataSet/'
SINGERS = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

MIN_FACE = 70
OUTPUT_DIR = 'output/'
TESTING_DIR = 'testing/'
TRAINING_DIR = 'training/'
DATASET_DIR = 'dataSet/'

FEATURE = 'haarcascade_frontalface_default.xml'
