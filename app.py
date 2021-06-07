"""

@author: Ninad Mohite
"""

import os
import cv2
import re
import numpy as np
from google.cloud import vision
from PIL import Image

import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Model/Model_Detection/ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()


def vision_ocr(image):
    _, encoded_img = cv2.imencode('.jpg', image)
    ocr_img = vision.Image(content=encoded_img.tobytes())
    response = client.text_detection(image=ocr_img)

    block = ''
    line = []
    last_point = [0] * 8
    for i, texts in enumerate(response.text_annotations):
        if i == 0:
            continue
        points = [texts.bounding_poly.vertices[0].x, texts.bounding_poly.vertices[0].y,
                  texts.bounding_poly.vertices[1].x, texts.bounding_poly.vertices[1].y,
                  texts.bounding_poly.vertices[2].x, texts.bounding_poly.vertices[2].y,
                  texts.bounding_poly.vertices[3].x, texts.bounding_poly.vertices[3].y]

        dist = ((points[0] - last_point[2]) ** 2 + (points[1] - last_point[3]) ** 2) ** 0.5
        last_point = points
        if dist < 100:
            block = ''.join([block, texts.description])
        else:
            line.append(block)
            block = texts.description

        x1, y1 = min(points[0::2]), min(points[1::2])
        x2, y2 = max(points[0::2]), max(points[1::2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, texts.description, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return image, line[1:]


def parse_vin(texts):
    if len(texts) == 0:
        return 0
    filtered = []
    for string in texts:
        if 'ENG' in string:
            continue

        key = [(11, 6), (12, 6), (10, 6), (11, 5), (10, 5)]
        for k in key:
            if re.findall(f'[\w]{{{k[0]}}}[0-9]{{{k[1]}}}', string):
                string = string[::-1]
                span = re.search(f'[0-9]{{{k[1]}}}', string)
                string = string[span.start():span.end() + k[0]]
                string = max(re.split('[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', string[::-1]), key=len)
                filtered.append(string)
                break
    return filtered if len(filtered) != 0 else 0


if __name__ == '__main__':
    st.title("VIN Recognition")
    uploaded_image = st.file_uploader("Choose a png or jpg image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        input_image = np.asarray(Image.open(uploaded_image))
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        if st.button("make prediction"):
            out_image, out_texts = vision_ocr(input_image)
            st.image(out_image, caption="VIN recognised", use_column_width=True)
            filtered_output = parse_vin(out_texts)
            st.write(filtered_output[0])

            for out in filtered_output:
           	 st.write({'chassis_no': out})
