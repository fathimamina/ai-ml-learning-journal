import streamlit as st
import cv2
import numpy as np



from modules.preprocessing import *
from modules.augmentation import *

st.set_page_config(
    page_title="AI Vision Studio (opencv)",
    layout="wide"
)

st.title("📸 AI Vision Studio")

file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if file:

    bytes_data = np.asarray(
        bytearray(file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        bytes_data,
        cv2.IMREAD_COLOR
    )

    st.sidebar.header("Operations")

    operation = st.sidebar.selectbox(
        "Choose Feature",
        [
            "Resize",
            "Crop",
            "Flip",
            "Normalize",
            "Blur",
            "Threshold",
            "Morphology",
            "Edge Detection",
            "Rotation",
            "Translation",
            "Scaling",
            "Brightness",
            "Contrast",
            "Noise"
        ]
    )

    output = image.copy()

    # -------------------------
    # PREPROCESSING
    # -------------------------

    if operation == "Resize":

        width = st.sidebar.slider(
            "Width",
            100,
            2000,
            image.shape[1]
        )

        height = st.sidebar.slider(
            "Height",
            100,
            2000,
            image.shape[0]
        )

        output = resize_image(
            image,
            width,
            height
        )

    elif operation == "Crop":

        h, w = image.shape[:2]

        x1 = st.sidebar.slider(
            "x1",
            0,
            w-1,
            0
        )

        y1 = st.sidebar.slider(
            "y1",
            0,
            h-1,
            0
        )

        x2 = st.sidebar.slider(
            "x2",
            x1+1,
            w,
            w
        )

        y2 = st.sidebar.slider(
            "y2",
            y1+1,
            h,
            h
        )

        output = crop_image(
            image,
            x1,
            y1,
            x2,
            y2
        )

    elif operation == "Flip":

        output = flip_image(image)

    elif operation == "Normalize":

        output = normalize_image(image)

    elif operation == "Blur":

        output = blur_image(image)

    elif operation == "Threshold":

        output = threshold_image(image)

    elif operation == "Morphology":

        output = morphology_image(image)

    elif operation == "Edge Detection":

        output = edge_detection(image)

    # -------------------------
    # AUGMENTATION
    # -------------------------

    elif operation == "Rotation":

        angle = st.sidebar.slider(
            "Angle",
            -180,
            180,
            0
        )

        output = rotate(
            image,
            angle
        )

    elif operation == "Translation":

        tx = st.sidebar.slider(
            "Translate X",
            -500,
            500,
            0
        )

        ty = st.sidebar.slider(
            "Translate Y",
            -500,
            500,
            0
        )

        output = translate_image(
            image,
            tx,
            ty
        )

    elif operation == "Scaling":

        scale = st.sidebar.slider(
            "Scale",
            0.1,
            3.0,
            1.0
        )

        output = scale_image(
            image,
            scale
        )

    elif operation == "Brightness":

        beta = st.sidebar.slider(
            "Brightness",
            -100,
            100,
            0
        )

        output = brightness(
            image,
            beta
        )

    elif operation == "Contrast":

        alpha = st.sidebar.slider(
            "Contrast",
            0.1,
            3.0,
            1.0
        )

        output = contrast(
            image,
            alpha
        )

    elif operation == "Noise":

        output = add_noise(image)

    # -------------------------
    # DISPLAY
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(
            image,
            channels="BGR",
            use_container_width=True
        )

    with col2:
        st.subheader("Processed")
        st.image(
            output,
            channels="BGR" if len(output.shape) == 3 else None,
            use_container_width=True
        )