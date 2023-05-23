import numpy as np
import av
from insightface.app import FaceAnalysis
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# 初期化
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# コールバック
def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 顔認識
    faces = app.get(np.asarray(img))

    # 顔認識した部分に枠を描画
    rimg = app.draw_on(img, faces)

    return av.VideoFrame.from_ndarray(rimg, format="bgr24")

# メイン
def appmain():
    st.title("みまもりくん")
    st.subheader('＜顔検知（性別・年齢判定）＞')

    webrtc_streamer(
        key="video",
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

#コマンドプロンプト上で表示する
if __name__ == "__main__":
    #関数を呼び出す
    appmain()
