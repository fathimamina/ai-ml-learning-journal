import cv2
import numpy as np

def resize_image(img, width, height):
    return cv2.resize(img, (width, height))


def flip_image(img):
    return cv2.flip(img, 1)


def blur_image(img):
    return cv2.GaussianBlur(
        img,
        (9,9),
        0
    )


def edge_detection(img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Canny(
        gray,
        100,
        200
    )

def resize_image(img, width, height):
    return cv2.resize(img, (width, height))

def crop_image(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def normalize_image(img):
    return cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

def threshold_image(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return thresh

def morphology_image(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    kernel = np.ones(
        (5,5),
        np.uint8
    )

    return cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        kernel
    )