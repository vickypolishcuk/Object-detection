import streamlit as st
from PIL import Image
import cv2
import math
import numpy as np
from ultralytics import YOLO

st.title('Завантажте фото для розпізнавання')

# Додаємо кнопку для завантаження файлу
uploaded_file = st.file_uploader("Завантажте зображення", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Відображаємо завантажене зображення
    image = Image.open(uploaded_file)
    st.image(image, caption='Завантажене зображення', use_column_width="auto")

    # Використовуємо модель YOLO8 для розпізнавання
    model = YOLO("yolov8l.pt")
    
    # Виконуємо обробку зображення
    img = np.array(image)
    results = model(img, stream=True)
    for i in results:
        boxes = i.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)
            cls = i.names[box.cls[0].item()]

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            text = f"{cls} {confidence * 100:.1f}%"
            cv2.putText(img, text, org, font, fontScale, color, thickness)

    # Виводимо оброблене зображення
    st.image(img, caption='Оброблене зображення', use_column_width="auto")


