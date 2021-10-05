import cv2 as cv
import numpy as np


def carDetection():
    classify_body = cv.CascadeClassifier('cas4.xml')
    cap = cv.VideoCapture('videos/Vid1.mp4')

    while cap.isOpened():
        ret, frame = cap.read()  # Read the data frame
        if not ret:
            print("Can't receive frame!")
            break
        frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Turn it into grayscale for better performance
        cars_detected = classify_body.detectMultiScale(gray, 1.2, 3)
        for (x, y, w, h) in cars_detected:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow('frame', frame)

        if cv.waitKey(20) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()





if __name__ == '__main__':
    carDetection()
