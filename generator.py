import glob, os, shutil
from PIL import Image
import numpy as np
import cv2

singers = [
    'dam vinh hung',
    'dan truong',
    'ha anh tuan',
    'my tam',
    'huong tram',
    'son tung'
]

for singer in singers:
    path = 'dataSet/' + singer
    i = 1
    total = len(os.listdir(path))
    for filename in os.listdir(path):
        # Rename image
        # dst = path + '/' + singer + '.' + str(i) + '.png'
        # src = path + '/' + filename
        # os.rename(src, dst)

        # move to training/testing
        src = path + '/' + filename
        if (i < total*0.7):
            shutil.copy2(src, 'dataSet/training')
        else:
            shutil.copy2(src, 'dataSet/testing')
        i += 1







