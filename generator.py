import glob, os, shutil
from PIL import Image
import numpy as np
import cv2
import settings


def remove_file(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)

remove_file(settings.OUTPUT_DIR)
remove_file(settings.TESTING_DIR)
remove_file(settings.TRAINING_DIR)

singers = settings.SINGERS
for singer in singers:
    path = settings.DATASET_DIR + singer
    i = 1
    total = len(os.listdir(path))
    for filename in os.listdir(path):
        src = path + '/' + filename
        if (i < total*0.7):
            shutil.copy2(src, settings.TRAINING_DIR)
        else:
            shutil.copy2(src, settings.TESTING_DIR)
        i += 1







