import math
import cv2

def euclidian_distance(cords_a, cords_b):
    '''
    Get the  euclidian distance of two objects identified by given cordinates
    :param person_a:
    :param person_b:
    :return:
    '''
    x_dist = cords_a[0] - cords_b[0]
    y_dist = cords_a[1] - cords_b[1]

    return math.sqrt(x_dist * x_dist + y_dist * y_dist)

def extract_frames(video_seq_path):
    '''
    Returns all frames from a video sequence
    :param video_seq_path:
    :return:
    '''
    all_frames = []

    #
    capture = cv2.VideoCapture(video_seq_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        if ret:
            all_frames.append(frame)
        else:
            capture.release()
            return all_frames

def get_center(top_left,bottom_right):
    '''
    Get the center coordinates of a box in a frame represent
    :return:
    '''

    x=(float)(top_left[0]+bottom_right[0])/2
    y=(float)(top_left[1]+bottom_right[1])/2

    return (x,y)

def get_cords_from_box(box):
    '''
    Given a box,returns the top left and bottom right cordinates
    :param box:
    :return: (topleft,bottomRight)
    '''
    (centerX, centerY, width, height) = box.astype("int")

    x1 = int(centerX - (width / 2))
    y1 = int(centerY + (height / 2))
    x2 = int(centerX + (width / 2))
    y2 = int(centerY - (height / 2))

    return (x1, y1),(x2, y2)


def float_cords_to_int(cord):
    '''
    Given cords with x,y values given as floats ,return the values as ints
    :param cord:
    :return:
    '''
    x = int(cord[0])
    y = int(cord[1])

    return (x,y)