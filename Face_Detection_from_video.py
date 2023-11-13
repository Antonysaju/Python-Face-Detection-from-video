import cv2
import os

cascPath= "haarcascade_frontalface_default.xml"
faceCascade= cv2.CascadeClassifier(cascPath)

video= cv2.VideoCapture("99_intro.mp4")

process_this_frame= True
nfaces= 0

while True:
    ret, frame= video.read()
    gray_video= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces= faceCascade.detectMultiScale(
        gray_video,
        scaleFactor= 1.1,
        minNeighbors= 5,
        minSize= (30,30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        nfaces= nfaces+1

    cv2.imshow("Video", frame)
    if cv2.waitKey(25) & 0xFF== ord("q"):
        break

print("Found {0} faces".format(nfaces))

cv2.destroyAllWindows()
