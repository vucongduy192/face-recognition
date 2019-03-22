import glob, os, shutil
from PIL import Image
import numpy as np
import cv2
import settings

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(settings.FEATURE)
singers = settings.SINGERS
path = settings.TRAINING_DIR


faces = []
IDs = []
def get_images_and_labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        singer = os.path.split(imagePath)[-1].split('.')[0]
        ID = singers.index(singer) + 1
        temp = faceCascade.detectMultiScale(faceNp, 1.1, 5)
        for (x, y, w, h) in faceCascade.detectMultiScale(faceNp):
            faces.append(faceNp[y:y + h, x:x + w])
            IDs.append(ID)

        cv2.imshow('Traning', faceNp)
        cv2.waitKey(50)

    return np.array(IDs), faces

IDs, faces = get_images_and_labels(path)
recognizer.train(faces,  np.array(IDs))
recognizer.save('traningData.yml')
cv2.destroyAllWindows()
