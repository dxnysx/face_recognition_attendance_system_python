import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# create a list that will get the images from the folder automatically
# encoding
# generate the encoding automatically then find images through webcam

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    # read current image (path) / cls (name of our image)
    currentImg = cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    # append classNames, don't want 'jpg', grab first element (name)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


# encoding process
# create function to find encoding of each image
# first convert image into RGB
def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encoded_face_train = findEncoding(images)


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        # first check if the name of the attendee is already exist in attendance.csv
        # if the name not in attendance.csv, attendee name will be recorded with a time of function call.
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')


encodeListKnown = findEncoding(images)
# print(len(encodeListKnown)) Print how many images are there
print('Encoding completed!')

# initialize webcam
capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    success, img = capture.read()  # give image
    # reduce size of image, help to speed process
    # scale down the image to 1/4 of the size
    imgSize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSize = cv2.cvtColor(imgSize, cv2.COLOR_BGR2RGB)

    facesInFrame = face_recognition.face_locations(imgSize)
    encodeFrame = face_recognition.face_encodings(imgSize, facesInFrame)

    # iterate all faces in current frame and compare all the faces that are encoded

    # grab Face Location in faces Current Frame List, then grab encode Face from encode Frame
    for encodeFace, faceLoc in zip(encodeFrame, facesInFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # lower element, best match
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # show original image
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
