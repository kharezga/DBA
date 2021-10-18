import cv2 as cv
import numpy as np


def initializeYOLO(blob, image):
    classFile = 'yolo/coco.names'
    with open(classFile, 'rt') as file:
        classNames = file.read().rstrip('\n').split('\n')  # Extract class names from coco.names file

    net = cv.dnn.readNetFromDarknet("yolo/yolov3-tiny.cfg", "yolo/yolov3-tiny.weights")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    net.setInput(blob)

    layerNames = net.getLayerNames()
    net.getUnconnectedOutLayers()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # Get output names of layer

    outputs = net.forward(outputNames)  # Send output names

    findObjects(outputs, image, classNames)


def findObjects(outputs, frame, names):
    height, width, center = frame.shape
    bounding_box = []
    class_ids = []
    conf = []
    nmsThreshold = 0.3  # The bigger it is the more boxes are reduced

    for out in outputs:
        for detection in out:
            scores = detection[5:]  # Remove first 5 elements
            class_id = np.argmax(scores)  # Get index of the max value
            confidence = scores[class_id]  # Get the maximal value

            # Filtering objects
            if confidence > 0.5:
                w, h = int(detection[2] * width), int(detection[3] * height)  # Save the width and the height
                x, y = int(detection[0] * width - w / 2), int(
                    detection[0] * height - h / 2)  # Save the center point of the detection
                bounding_box.append([x, y, w, h])
                class_ids.append(class_id)
                conf.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bounding_box, conf, 0.5, nmsThreshold)  # Determines which bounding boxes we should keep
    for i in indices:
        i = i[0]  # Remove the extra bracket
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(frame, f'{names[class_ids[i].upper()]} {int(conf[i] * 100)} %', (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def convertImageToBlob(image):
    # Create a blob for each frame of the Video
    blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    return blob


def loadVideo():
    cap = cv.VideoCapture('videos/Vid1.mp4')

    while cap.isOpened():
        ret, frame = cap.read()  # Read the data frame

        if not ret:
            print("Can't receive frame!")  # Error handling
            break
        frame = cv.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv.INTER_LINEAR)
        frame_blob = convertImageToBlob(frame)
        initializeYOLO(frame_blob, frame)
        cv.imshow('Video', frame)

        if cv.waitKey(20) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    loadVideo()

