import cv2
import numpy as np

def rotate(img, angle):

    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D(
        (w//2, h//2),
        angle,
        1
    )

    return cv2.warpAffine(
        img,
        M,
        (w,h)
    )

def brightness(img, beta):

    return cv2.convertScaleAbs(
        img,
        alpha=1,
        beta=beta
    )

def contrast(img, alpha):

    return cv2.convertScaleAbs(
        img,
        alpha=alpha,
        beta=0
    )

def translate_image(img, tx, ty):

    h,w = img.shape[:2]

    M = np.float32([
        [1,0,tx],
        [0,1,ty]
    ])

    return cv2.warpAffine(
        img,
        M,
        (w,h)
    )

def scale_image(img, scale):

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )

def add_noise(img):

    noise = np.random.normal(
        0,
        25,
        img.shape
    )

    noisy = img + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)