from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import glob, os, shutil
from PIL import Image
import numpy as np

import settings
import cv2


lfw_people = fetch_lfw_people(min_faces_per_person=settings.MIN_FACE, funneled=False, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

###############################################################################
# Save to LFW images dataSet


def remove_file(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)

remove_file(settings.OUTPUT_DIR)
remove_file(settings.TESTING_DIR)
remove_file(settings.TRAINING_DIR)

index = 0
count_img = np.zeros((n_classes,), dtype=int)
for x in X_train:
    class_num = y_train[index]
    index += 1
    count_img[class_num] += 1
    if (count_img[class_num] > settings.MIN_FACE):
        continue

    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(settings.TRAINING_DIR + target_names[class_num] + '.' + str(count_img[class_num]) + '.png')

index = 0
count_img = np.zeros((n_classes,), dtype=int)
for x in X_test:
    class_num = y_test[index]
    index += 1
    count_img[class_num] += 1
    if (count_img[class_num] > 0.25*settings.MIN_FACE):
        continue

    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(settings.TESTING_DIR + target_names[class_num] + '.' + str(count_img[class_num]) + '.png')





