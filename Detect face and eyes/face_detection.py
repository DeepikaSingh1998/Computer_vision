# -*- coding: utf-8 -*-
import cv2

#load series of filters(Cascades) for eyes and face
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

#a function for detecting face and eyes
def detect(gray,frame):
    #detect faces in the frame
    faces=face_cascade.detectMultiScale(gray, 1.3, 5)
    #get a rectangle around each face and its eyes
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)
    return frame

#turn on  the webcam
video_capture=cv2.VideoCapture(0) #0 for internal webcam, 1 for external
while True:
    _,frame=video_capture.read()#keep getting the latest frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Video',canvas)
    #stop on keyboard interruption
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
#turn off webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()