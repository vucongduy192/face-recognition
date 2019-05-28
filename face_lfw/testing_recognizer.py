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
count_sample = np.zeros((len(singers),), dtype=int)
count_error = np.zeros((len(singers),), dtype=int)
font = cv2.FONT_HERSHEY_SIMPLEX

for filename in os.listdir(path):
    img_result = cv2.imread(path + filename)
    singer_name = filename.split('.')[0]
    index = singers.index(singer_name)
    count_sample[index] += 1

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
                count_error[index] += 1
        cv2.putText(img_result, id + ' - ' + filename, (x, h), font, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)

    if (id == "unknown"):
        cv2.putText(img_result, id + ' - ' + filename, (x, h), font, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        unknown += 1
    cv2.imshow('Testing', img_result)
    cv2.imwrite(settings.OUTPUT_DIR + filename, img_result)
    cv2.waitKey(100)
    if (cv2.waitKey(1) == ord('q')):
        break

cv2.destroyAllWindows()
print ('Expected results for the top 5 most represented people in the dataset')
for index, singer in enumerate(singers):
    result = str(1-count_error[index]/count_sample[index])
    print ('{:20} : {:10}'.format(singer, result))


print ('\n')

print ('==============================Summary================================')
print ('Corrects : ' + str(correct))
print ('Detect Fails : ' + str(unknown))
print ('Recognizer Fails : ' + str(fail))
print ("Accuracy : " + str(correct/(correct+unknown+fail)*100))