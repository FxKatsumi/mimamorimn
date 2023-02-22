import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

# カスケードファイルへのパス名
#cascadePath = "haarcascades/haarcascade_frontalface_alt.xml"
cascadePath = "./haarcascade_frontalface_alt.xml"

# 顔の検出器を作成
face_cascade = cv2.CascadeClassifier(cascadePath)

# # 画像のサイズの指定
# width = 500
# height = 300

# 色       
red = (0,0,255)

# コールバック
def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # #画像のサイズを変更（リサイズ）
    # img = cv2.resize(img, (width, height))

    #グレイスケールに変換
    gray_flame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #顔認証を実行(minNeighboreは信頼性のパラメータ)
    #face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=20)
    face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=3)

    #顔を四角で囲む
    for (x,y,w,h) in face_list:
        #赤色の枠で囲む
        cv2.rectangle(img, (x,y), (x+w,y+h), red, 1)

    #person_cnt = len(face_list)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# メイン
def appmain():
    st.title("みまもりくん")
    st.subheader('＜顔検知＞')

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
