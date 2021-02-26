import numpy as np
import cv2
import time

cap = cv2.VideoCapture("object_video.mp4")
prev_frame_time = 0
new_frame_time = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

_,frame1 = cap.read()
_,frame2 = cap.read()
control = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
frame1 = cv2.GaussianBlur(cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY),(7,7),0)
frame2 = cv2.GaussianBlur(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY),(7,7),0)

_,frame1 = cv2.threshold(frame1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_,frame2 = cv2.threshold(frame2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

while(True):
    
    frameDiff = abs(frame2-frame1)
    erode = cv2.erode(frameDiff,kernel,iterations = 3)

    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(erode,(x,y),(x+w,y+h),(255,0,0),thickness= 2)
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    cv2.putText(erode,"FPS:"+str(int(fps)),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    Hori = np.concatenate((erode,control), axis=1) 

    cv2.imshow("frame",Hori)
    frame1 = frame2
    _,frame2 = cap.read() 
    control = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    frame2 = cv2.GaussianBlur(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY),(7,7),0)
    _,frame2 = cv2.threshold(frame2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

