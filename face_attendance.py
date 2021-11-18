# import the necessary packages
import numpy as np
from datetime import datetime
import os
import cv2
import argparse
import pickle
import face_recognition as fr

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

def Attendance(name):
    with open('Attendance_Registry.csv','r+') as f:
        DataList = f.readlines()
        names = []
        for dtl in DataList:
            ent = dtl.split(',')
            names.append(ent[0])
        if name not in names:
            curr = datetime.now()
            dt = curr.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
print("[INFO] encodings complete")
print("[INFO] turning on webcam")

cap = cv2.VideoCapture(0)
print("[INFO] Press q to exit webcam")

while True:
    _, img = cap.read()
    image = cv2.resize(img,(0,0),None,0.25,0.25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    encodeKnown = list(data["encodings"])

    facesInFrame = fr.face_locations(image)
    encodesInFrame = fr.face_encodings(image,facesInFrame)

    for encFace,faceLoc in zip(encodesInFrame,facesInFrame):
            matchList = fr.compare_faces(encodeKnown,encFace)
            faceDist = fr.face_distance(encodeKnown,encFace)
            match = np.argmin(faceDist)
            if matchList[match]:
                name = data["names"][match]
                Attendance(name)

    
    cv2.imshow("img", img)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break

# do a bit of cleanup
cv2.destroyAllWindows()
   