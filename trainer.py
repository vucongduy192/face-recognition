import glob, os, shutil
from PIL import Image
import numpy as np
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
singers = [
    'dam vinh hung',
    'dan truong',
    'ha anh tuan',
    'my tam',
    'huong tram',
    'son tung'
]
path = 'dataSet/training'
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

faces = []
IDs = []
def get_images_and_labels(path):
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        singer = os.path.split(imagePath)[-1].split('.')[0]
        ID = singers.index(singer) + 1
        temp = faceCascade.detectMultiScale(faceNp)
        if (temp is None):
            print (imagePath)
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

# Preprocessing image
#
# for singer in singers:
#     path = 'dataSet/' + singer
#     i = 1
#     total = len(os.listdir(path))
#     for filename in os.listdir(path):
#         # Rename image
#         # dst = path + '/' + singer + '.' + str(i) + '.png'
#         # src = path + '/' + filename
#         # os.rename(src, dst)
#
#         # move to training/testing
#         src = path + '/' + filename
#         if (i < total*0.7):
#             shutil.copy2(src, 'dataSet/training')
#         else:
#             shutil.copy2(src, 'dataSet/testing')
#         i += 1







