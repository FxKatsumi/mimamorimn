# import os
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image, ImageDraw, ImageFont

from common.header import Face_ID_faceid_reg
from common.facemain import appmain

# # カスケードファイルへのパス名
# cascadePath = "./haarcascade_frontalface_alt.xml"

# # 顔の検出器を作成
# face_cascade = cv2.CascadeClassifier(cascadePath)
# face_data_path = "faceimage" # 顔データパス名

# # 色       
# red = (0,0,255)

# expand_rate = 1.5 # 拡大率
# # face_threshold = 0.7 # 顔閾値
# face_threshold = 0.8 # 顔閾値

# # 顔検出のAI
# # image_size: 顔を検出して切り取るサイズ
# # margin: 顔まわりの余白
# mtcnn = MTCNN(image_size=160, margin=10)

# # 切り取った顔を512個の数字にするAI
# # 1回目の実行では学習済みのモデルをダウンロードしますので、少し時間かかります。
# resnet = InceptionResnetV1(pretrained='vggface2').eval()

# # グローバル変数を初期化する
# if 'mode' not in st.session_state: #モード
#     st.session_state.mode = 0

# if 'outfile' not in st.session_state: #ファイル名
#     st.session_state.outfile = ""

# mode = st.session_state.mode
# outfile = st.session_state.outfile

# # 顔データID変換（イメージ）
# def GetFaceIDImage(img):
#     try:
#         # 顔データを160×160に切り抜き
#         img_crop = mtcnn(img)
#         # 切り抜いた顔データを512個の数字に
#         img_emb = resnet(img_crop.unsqueeze(0))
#         # 512個の数字にしたものはpytorchのtensorという型なので、numpyの型に変換
#         return img_emb.squeeze().to('cpu').detach().numpy().copy()

#     except Exception as e:
#         #print('GetFaceIDImage:', e)
#         return None

# # 顔データID変換（ファイル名）
# def GetFaceIDFile(fname):
#     try:
#         # イメージファイルパス名
#         image_path = os.path.join(face_data_path, fname)
#         # 画像データ取得
#         img = Image.open(image_path) 
#         # 顔データID変換（イメージ）
#         return GetFaceIDImage(img)

#     except Exception as e:
#         # print('GetFaceIDFile:', e)
#         pass

# # 顔データクラス
# class FaceDataClass:
#     # コンストラクタ 
#     def __init__(self, fname):
#         try:
#             # 名前（ファイル名）設定
#             self.name = os.path.splitext(os.path.basename(fname))[0]
#             # 顔データID変換
#             self.id = GetFaceIDFile(fname)

#         except Exception as e:
#             # print('FaceDataClass(__init__):', e)
#             pass

# # 顔データ配列
# FaceDatas = []

# # 顔データ読み込み
# def FaceDataRead():
#     try:
#         # 顔データフォルダー検索
#         folderfiles = os.listdir(face_data_path)
#         # 顔データファイル名のみ取得
#         files = [f for f in folderfiles if os.path.isfile(os.path.join(face_data_path, f))]

#         # 顔データ取得
#         for fname in files:
#             # 顔データ追加
#             FaceDatas.append(FaceDataClass(fname))

#     except Exception as e:
#         # print('FaceDataRead:', e)
#         pass

# # コサイン類似度算出
# def cos_similarity(p1, p2):
#     try:
#         return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

#     except Exception as e:
#         # print('cos_similarity:', e)
#         pass

# # 顔認識
# def faceRecognition(fid):
#     try:
#         facename = '???' # 名前
#         score = 0 # スコア

#         # 顔データ検索
#         for fd in FaceDatas:
#             # 類似度取得
#             res = cos_similarity(fid, fd.id)

#             if res >= face_threshold: # 一致？
#                 if res > score: # より類似？
#                     score = res
#                     facename = fd.name

#         return facename

#     except Exception as e:
#         # print('faceRecognition:', e)
#         pass

# # クリップ拡大
# def clipExpand(x,y,w,h, img_height, img_width):
#     try:
#         w2 = int(w * expand_rate)
#         x2 = int(x - (w2 - w) / 2)
#         h2 = int(h * expand_rate)
#         y2 = int(y - (h2 - h) / 2)

#         # 補正
#         if x2 < 0:
#             x2 = 0
#         if y2 < 0:
#             y2 = 0
#         if x2 + w2 > img_width:
#             w2 = img_width - x2
#         if y2 + h2 > img_height:
#             h2 = img_height - y2

#         return (x2, y2, w2, h2)

#     except Exception as e:
#         # print('clipExpand:', e)
#         pass

# # コールバック
# def callback(frame):
#     global mode

#     try:
#         img = frame.to_ndarray(format="bgr24")
#         img_height, img_width = img.shape[:2]

#         #グレイスケールに変換
#         gray_flame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         #顔認証を実行(minNeighboreは信頼性のパラメータ)
#         #face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=20)
#         # face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=3)
#         face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=2)

#         #顔を四角で囲む
#         for (x,y,w,h) in face_list:
#             if mode == 0: # 登録ボタンが押されていない
#                 #赤色の枠で囲む
#                 cv2.rectangle(img, (x,y), (x+w,y+h), red, 1)

#             elif mode == 1: # 登録ボタンが押された
#                 # クリップ拡大
#                 (x2, y2, w2, h2) = clipExpand(x,y,w,h, img_height, img_width)
#                 #顔のみ切り取る
#                 trim_face = img[y2:y2+h2, x2:x2+w2]

#                 # 画像を登録
#                 cv2.imwrite(os.path.join(face_data_path, outfile + ".jpg"), trim_face)

#                 #赤色の枠で囲む
#                 cv2.rectangle(img, (x,y), (x+w,y+h), red, 1)

#                 mode = 2 # 完了

#                 exit

#             # else: # 完了
#                 # #赤色の枠で囲む
#                 # cv2.rectangle(img, (x,y), (x+w,y+h), red, 1)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

#     except Exception as e:
#         # with open('zdebug.txt', 'a') as f:
#         #     # ファイルに書き込む
#         #     f.write("err: " + str(e) + "\n")
#         pass

# # メイン
# def appmain():
#     global mode
#     global outfile

#     st.title("みまもりくん")
#     st.subheader('＜顔認識（登録モード）＞')

#     # 顔データ読み込み
#     FaceDataRead()

#     # 状態表示
#     st.info("アルファベットで名前を入力し、STARTボタンで映像を表示した後、登録する顔が映ったときに登録ボタンを押してください")

#     # 名前
#     st.session_state.outfile = st.text_input("表示する名前（半角英数のみ）", "")

#     # 登録ボタン
#     if st.button("登録"):
#         if mode == 0: #未登録？
#             if st.session_state.outfile == "": #ブランク？
#                 st.error("名前を入力してください")

#             else: #入力あり
#                 outfile = st.session_state.outfile
#                 mode = 1 # 登録ボタン
#                 st.session_state.mode = mode

#     webrtc_ctx = webrtc_streamer(
#         key="video",
#         video_frame_callback=callback,
#         rtc_configuration={
#             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         }
#     )

#コマンドプロンプト上で表示する
if __name__ == "__main__":
    #関数を呼び出す
    appmain(Face_ID_faceid_reg)
