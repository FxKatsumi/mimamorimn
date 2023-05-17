# import streamlit as st
# import cv2
# from PIL import Image
# import numpy as np
# from streamlit_webrtc import webrtc_streamer
# import av
# from pathlib import Path

# from const import CLASSES, COLORS
# from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT

# # ダウンロードモジュール
# from sample_utils.download import download_file

from common.header import Object_ID_object
from common.objectmain import appmain

# HERE = Path(__file__).parent
# ROOT = HERE.parent

# MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
# MODEL_LOCAL_PATH = ROOT / "./model/MobileNetSSD_deploy.caffemodel"
# PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
# PROTOTXT_LOCAL_PATH = ROOT / "./model/MobileNetSSD_deploy.prototxt.txt"

# @st.cache
# def process_image(image):
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
#     )
#     net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
#     net.setInput(blob)
#     detections = net.forward()
#     return detections

# @st.cache
# def annotate_image(
#     image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
# ):
#     # loop over the detections
#     (h, w) = image.shape[:2]
#     labels = []
#     for i in np.arange(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > confidence_threshold:
#             # extract the index of the class label from the `detections`,
#             # then compute the (x, y)-coordinates of the bounding box for
#             # the object
#             idx = int(detections[0, 0, i, 1])
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # display the prediction
#             label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
#             labels.append(label)
#             cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
#             y = startY - 15 if startY - 15 > 15 else startY + 15
#             cv2.putText(
#                 image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
#             )
#     return image, labels

# # st.title("Object detection with MobileNet SSD")
# # img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# # confidence_threshold = st.slider(
# #     "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
# # )

# # if img_file_buffer is not None:
# #     image = np.array(Image.open(img_file_buffer))

# # else:
# #     demo_image = DEMO_IMAGE
# #     image = np.array(Image.open(demo_image))

# # detections = process_image(image)
# # image, labels = annotate_image(image, detections, confidence_threshold)

# # st.image(
# #     image, caption=f"Processed image", use_column_width=True,
# # )

# # st.write(labels)

# # コールバック
# def callback(frame):
#     img = frame.to_ndarray(format="bgr24")

#     detections = process_image(img)
#     # img, labels = annotate_image(img, detections, confidence_threshold)
#     # img, labels = annotate_image(img, detections, 0.5)
#     img, labels = annotate_image(img, detections, 0.1)

#     return av.VideoFrame.from_ndarray(img, format="bgr24")

# # メイン
# def appmain():
#     st.title("みまもりくん")
#     st.subheader('＜物体検知＞')

#     # ファイルダウンロード
#     download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564) # 学習モデル
#     download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353) # プロトコル

#     webrtc_streamer(
#         key="video",
#         video_frame_callback=callback,
#         rtc_configuration={
#             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         }
#     )

#コマンドプロンプト上で表示する
if __name__ == "__main__":
    #関数を呼び出す
    appmain(Object_ID_object)
