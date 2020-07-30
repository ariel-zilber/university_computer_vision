import cv2
import numpy as np

# Consts:
from utils import euclidian_distance, get_center, get_cords_from_box, float_cords_to_int

# distance bwteen objects in video considerd too close
TOO_CLOSE = 100

# minimal confidance for detection
MIN_CONFIDENCE = 0.3

# the coco yolo id
PERSON_CLASS_ID = 0

# the max distance where objects are considered to be same size
EPSILON = 10

# configuration
YOLO_CFG = 'yolo.cfg'
YOLO_WEIGHTS = 'yolo.weights'
VIDEO_SAMPLE_PATH = 'test_video.mp4'
OUTPUT_FILE_PATH = "./output/output.avi"


def extract_Objects(video_seq_path, config_path, weights_path):
    '''
    Extracts features
    :param video_seq_path:
    :param configPath:
    :param weightsPath:
    :return:
    Detected people locations per frame
    '''
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    cap = cv2.VideoCapture(video_seq_path)

    people_features = dict([])
    currFeature = 0

    while (cap.isOpened()):

        not_done, image = cap.read()

        if not not_done:
            break

        people_features[str(currFeature)] = []

        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        '''
        Iterate  over the output layers and add the detected objects on each layer
        that correspond to the Person class
        '''
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > MIN_CONFIDENCE and classID == PERSON_CLASS_ID:
                    box = detection[0:4] * np.array([W, H, W, H])
                    top_left, bottom_right = get_cords_from_box(box)

                    people_features[str(currFeature)].append((top_left, bottom_right))

        currFeature += 1

    return people_features

def seperate_objects_by_centers(objs_data, max_dist):
    '''
    Returns two list of objects separated by their relative distance from each other.
    The distance is configured by max_dist
    :param people_data:
    :param max_dist:
    :return:
    '''

    # get a list of center of objects
    obj_centers = []
    for cord in objs_data:
        obj_centers.append(get_center(cord[0], cord[1]))

    too_close = []
    far = []

    for i in range(0, len(obj_centers)):
        for j in range(0, len(obj_centers)):
            if i == j:
                continue

            d = euclidian_distance(obj_centers[j], obj_centers[i])

            (x1, y1) = float_cords_to_int(obj_centers[j])
            (x2, y2) = float_cords_to_int(obj_centers[i])

            if (d <= max_dist):
                too_close.append(((x1, y1), (x2, y2), objs_data[j], objs_data[i]))
            else:
                far.append(((x1, y1), (x2, y2), objs_data[j], objs_data[i]))

    return too_close, far

def remove_duplicate_objects_detections(extracted_obj, epsilon):
    '''
    Removes duplicate items.
    Items with distance less then epsilon are considered duplicates and are removed
    :param extracted_obj:
    :return:
    '''

    duplicate_free = dict([])
    current_frame = 0

    for key in extracted_obj.keys():

        duplicate_free[str(current_frame)] = []

        frame = extracted_obj[key]
        centers = []

        for item in frame:
            center = get_center(item[0], item[1])
            centers.append((center, item))

        for i in range(0, len(centers)):
            current_item = centers[i]
            is_duplicate = False

            for j in range(i + 1, len(centers)):
                other_item = centers[j]
                d = euclidian_distance(current_item[0], other_item[0])

                if d <= epsilon:
                    is_duplicate = True

            if not is_duplicate:
                duplicate_free[str(current_frame)].append(current_item[1])

        current_frame += 1

    return duplicate_free

def check_social_distancing(video_path, people_data):
    '''
    check the distance between objects in a provided video and saves the result at OUTPUT_FILE_PATH
    :param video_path:
    :param people_data:
    :return:
    '''

    cap = cv2.VideoCapture(video_path)

    # configure the video file writer
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_movie = cv2.VideoWriter(OUTPUT_FILE_PATH, fourcc, fps, (width, height))

    curr_frame = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        too_close, far = seperate_objects_by_centers(people_data[str(curr_frame)], TOO_CLOSE)

        # color scheme of objects
        color_bad = (0, 0, 255)
        color_ok = (0, 255, 0)
        color_distance = (255, 0, 0)

        # draw box from far
        for cords in far:
            frame = cv2.rectangle(frame, cords[2][1], cords[2][0], color_ok, 2)
            cv2.putText(frame, 'OK', cords[2][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 2)
            frame = cv2.rectangle(frame, cords[3][1], cords[3][0], color_ok, 2)

        # draw line between close objects
        for cords in too_close:
            frame = cv2.line(frame, cords[0], cords[1], color_distance, 3)

        # draw box of close objects
        for cords in too_close:
            frame = cv2.line(frame, cords[0], cords[1], color_distance, 3)
            frame = cv2.rectangle(frame, cords[2][1], cords[2][0], color_bad, 2)
            cv2.putText(frame, 'Too close', cords[2][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bad, 2)
            frame = cv2.rectangle(frame, cords[3][0], cords[3][1], color_bad, 2)

        # write to buffer
        output_movie.write(frame)

        # go to next frame
        curr_frame += 1

    # close cv2
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # extract the objects of a vide by frame
    extracted_obj = extract_Objects(VIDEO_SAMPLE_PATH, YOLO_CFG, YOLO_WEIGHTS)

    # remove duplicate items from false alarm
    extracted_obj = remove_duplicate_objects_detections(extracted_obj, 15)

    check_social_distancing(VIDEO_SAMPLE_PATH, people_data=extracted_obj)
