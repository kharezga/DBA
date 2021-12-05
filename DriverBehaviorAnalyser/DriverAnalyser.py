import numpy as np
import mediapipe as mp
import cv2 as cv
import os


def extractKeypoints(result):
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() \
        if result.face_landmarks else np.zeros(1404)
    return face


def extractKeypointsWithIris(result):
    """
    Extracts values of landmark of each coord (x,y,z) and assigns it to the flattened numpy array. If results don't
    exists the array of zeros is assigned.

    result: all landmarks of the deteced face
    """

    if result.multi_face_landmarks:
        for face_detected in result.multi_face_landmarks:
            face = np.array([[res.x, res.y, res.z] for res in face_detected.landmark]).flatten()
    else:
        face = np.zeros(1434)

    return face


def drawCustomLandmarks(image, results, holistic):
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    mp_drawing.draw_landmarks(image, results.face_landmarks, holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                     circle_radius=1))


def drawCustomLandmarksWithIris(image, results, mp_face_mesh):
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = image.copy()
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            cv.imshow('Video', annotated_image)


def faceFeaturesDetection(image, model):
    """Performs face detection with use of mediapipe holistic model

     Parameters:
     image: Image on which the detection is performed
     model: Mediaholistic model

     Returns:
     image: Processed frame
     results: Landmarks

    """

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False  # Save a little bit of memory
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Color conversion
    return image, results


def collectData():
    actions = np.array(['left', 'right', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values3')

    for action in actions:
        for sequence in os.listdir('Dataset2/' + action):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        for action in actions:
            for file in os.listdir('Dataset2/' + action):
                cap = cv.VideoCapture('Dataset2/' + action + "/" + file)
                frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Count the number of frames in each video
                for frameNumber in range(frameCount):

                    ret, frame = cap.read()

                    if not ret:  # Error handling
                        print("Can't receive frame!")
                        break

                    frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                                      interpolation=cv.INTER_LINEAR)  # Resize the image for better performance
                    image, results = faceFeaturesDetection(frame, holistic)
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                     circle_radius=1))  # Draw the landmarks on the image

                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(DATA_PATH, action, str(file), str(frameNumber))
                    np.save(npyPath, keypoints)

                    cv.imshow('Video', image)

                    if cv.waitKey(10) == ord('q'):
                        break
        cap.release()
        cv.destroyAllWindows()


def collectData2():
    actions = np.array(['lchange', 'rchange', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values_Sorted')

    for action in actions:
        for sequence in os.listdir('Dataset_Sorted/' + action):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
    sequenceLength = 150

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        for action in actions:
            for file in os.listdir('Dataset_Sorted/' + action):
                cap = cv.VideoCapture('Dataset_Sorted/' + action + "/" + file)
                for frameNumber in range(sequenceLength):

                    ret, frame = cap.read()
                    if not ret:  # Error handling
                        print("Can't receive frame!")
                        break

                    frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                                      interpolation=cv.INTER_LINEAR)  # Resize the image for better performance
                    image, results = faceFeaturesDetection(frame, holistic)

                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(DATA_PATH, action, str(file), str(frameNumber))
                    np.save(npyPath, keypoints)

                    cv.imshow('Video', image)

                    if cv.waitKey(10) == ord('q'):
                        break
        cap.release()
        cv.destroyAllWindows()


def collectDataWithIris():
    actions = np.array(['lchange', 'rchange', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values_Sorted_Iris_05_12_12_59')


    for action in actions:
        for sequence in os.listdir('Dataset_Sorted/' + action):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    mp_face_mesh = mp.solutions.face_mesh  # FaceMesh model
    sequenceLength = 150

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_faces=1,
                               refine_landmarks=True) as face_mesh:
        for action in actions:
            for file in os.listdir('Dataset_Sorted/' + action):
                cap = cv.VideoCapture('Dataset_Sorted/' + action + "/" + file)

                for frameNumber in range(sequenceLength):
                    ret, frame = cap.read()
                    if not ret:  # Error handling
                        print("Can't receive frame!")
                        break
                    frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                                      interpolation=cv.INTER_LINEAR)  # Resize the image for better performance

                    image, results = faceFeaturesDetection(frame, face_mesh)
                    drawCustomLandmarksWithIris(image, results, mp_face_mesh)

                    keypoints = extractKeypointsWithIris(results)
                    npyPath = os.path.join(DATA_PATH, action, str(file), str(frameNumber))
                    np.save(npyPath, keypoints)

                    print("Processing frame number:", frameNumber)

                    # Very important knowledge!!!
                    # https://stackoverflow.com/questions/67141844/how-do-i-get-the-coordinates-of-face-mash-landmarks-in-mediapipe kurwa kozak

                    if cv.waitKey(10) == ord('q'):
                        break

        cap.release()
        cv.destroyAllWindows()
  