import numpy as np
import cv2
import time

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

prev_frame_time = 0
new_frame_time = 0

video_path = "object_video.mp4"


def calculate_fps():
    """
    Calculates Frames per Second of the recording.

    Parameters
    ----------
    NONE

    Returns
    -------
        fps: frames per second (int) 
    """

    global prev_frame_time
    global new_frame_time

    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    return int(fps)

def preprocess_frame(frame):
    """
    Preprocesses the provided frame.

    rgb2gray-> gaussian blur-> otsu threshold

    Parameters
    ----------
    frame: frame that will be preprocessed

    Returns
    -------
    frame: preprocessed frame
    """
    frame = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(7,7),0)
    _,frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return frame

def capture_frames(cap):
    """
    Captures Frames and preprocesses them.

    Parameters
    ----------
    cap: cv2.VideoCapture

    Returns
    -------
        frame1: frame 1
        frame2: frame 2 (next frame of frame1)
        control: frame2 without preprocessed

    Calls
    -----
        preprocessframe()
    """
    _,frame1 = cap.read()
    _,frame2 = cap.read()

    control = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    frame1 = preprocess_frame(frame1)
    frame2 = preprocess_frame(frame2)
    return frame1, frame2, control

def main(): 

    cap = cv2.VideoCapture(video_path)
    print(type(cap))
    frame1,frame2,control = capture_frames(cap)
    while(True): 
        
        frameDiff = abs(frame2-frame1)
        erode = cv2.erode(frameDiff,kernel,iterations = 3)

        contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(erode,(x,y),(x+w,y+h),(255,0,0),thickness= 2)
        
        cv2.putText(erode,"FPS:"+str(calculate_fps()),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        horizontal_concat = np.concatenate((erode,control), axis=1) 

        cv2.imshow("video",horizontal_concat)
        frame1 = frame2
        _,frame2,control = capture_frames(cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
