import cv2 as cv
import numpy as np


def readVideo():
    cap = cv.VideoCapture('videos/Vid1.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(20) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def carDetection():
    readVideo()


if __name__ == '__main__':
    readVideo()
