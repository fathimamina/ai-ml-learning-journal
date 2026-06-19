import numpy as np

from tensorflow.keras.applications import (
    MobileNetV2
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)

model = MobileNetV2(
    weights="imagenet"
)

def classify(img):

    img = cv2.resize(
        img,
        (224,224)
    )

    x = np.expand_dims(
        img,
        axis=0
    )

    x = preprocess_input(x)

    preds = model.predict(x)

    return decode_predictions(
        preds,
        top=3
    )[0]