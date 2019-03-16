import os
import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('traningData.yml')
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

path = 'dataSet/testing/'
correct = 0
fail = 0
unknown = 0

singers = [
    'dam vinh hung',
    'dan truong',
    'ha anh tuan',
    'my tam',
    'huong tram',
    'son tung'
]

for filename in os.listdir(path):
    img_result = cv2.imread(path + filename)
    img = cv2.imread(path + filename, 0)
    faces = faceDetect.detectMultiScale(img, 1.1, 5)
    id = "unknown"
    for (x, y, w, h) in faces:
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, conf = recognizer.predict(img[y:y+h, x:x+w])
        id = singers[id - 1]
        if (conf < 150):
            if (id in filename):
                correct += 1
            else:
                fail += 1
        cv2.putText(img_result, id + ' - ' + filename, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)

    if (id == "unknown"):
        cv2.putText(img_result, id + ' - ' + filename, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        unknown += 1
    cv2.imshow('Testing', img_result)
    cv2.imwrite("output/" + filename, img_result)
    cv2.waitKey(100)
    if (cv2.waitKey(1) == ord('q')):
        break;

cv2.destroyAllWindows()

print ('Corrects : ' + str(correct))
print ('Detect Fails : ' + str(unknown) + ' --- Recognizer Fails : ' + str(fail))
print ("Accuracy : " + str(correct/(correct+unknown+fail)*100))