import cv2
import numpy as np


def add_text(
    img,
    text
):

    image = img.copy()

    cv2.putText(
        image,
        text,
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    return image

def blend_images(
    img1,
    img2
):

    img2 = cv2.resize(
        img2,
        (
            img1.shape[1],
            img1.shape[0]
        )
    )

    return cv2.addWeighted(
        img1,
        0.7,
        img2,
        0.3,
        0
    )

def sobel_gradient(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        1,
        ksize=3
    )


def template_match(
    img,
    template
):

    return cv2.matchTemplate(
        img,
        template,
        cv2.TM_CCOEFF_NORMED
    )


def detect_shapes(img):

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

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for cnt in contours:

        approx = cv2.approxPolyDP(
            cnt,
            0.04*cv2.arcLength(
                cnt,
                True
            ),
            True
        )

        cv2.drawContours(
            result,
            [approx],
            -1,
            (0,255,0),
            2
        )

    return result