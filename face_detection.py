from imutils.video import VideoStream
from imutils import face_utils
import imutils
import argparse
import time
import cv2
import dlib
from camera import *
# 
# Create an instance of the VideoCamera class
video_camera = VideoCamera()

# Get the frame with filters applied (make_save is an example parameter)
output_frame = video_camera.get_frame(makesave="noSave", imageformat="jpgq")
output_save = video_camera.save(imageformat="jpg")
output_savecnn = video_camera.savecnn(imageformat="jpg")

shape=output_savecnn


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, height=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Display some text in the upper left corner of the screen
    cv2.putText(frame, shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
