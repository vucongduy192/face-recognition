import os
import cv2
import numpy as np
import settings

faceDetect = cv2.CascadeClassifier(settings.FEATURE)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('traningData.yml')

path = settings.TESTING_DIR
correct = 0
fail = 0
unknown = 0

singers = settings.SINGERS
font = cv2.FONT_HERSHEY_SIMPLEX
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
        cv2.putText(img_result, id + ' - ' + filename, (x, h), font, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)

    if (id == "unknown"):
        cv2.putText(img_result, id + ' - ' + filename, (x, h), font, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        unknown += 1
    cv2.imshow('Testing', img_result)
    cv2.imwrite(settings.OUTPUT_DIR + filename, img_result)
    cv2.waitKey(100)
    if (cv2.waitKey(1) == ord('q')):
        break;

cv2.destroyAllWindows()

print ('Corrects : ' + str(correct))
print ('Detect Fails : ' + str(unknown))
print ('Recognizer Fails : ' + str(fail))
print ("Accuracy : " + str(correct/(correct+unknown+fail)*100))