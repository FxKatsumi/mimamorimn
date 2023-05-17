import av
import cv2
import numpy as np
import queue
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import sys

#メール
from email.mime.text import MIMEText
import smtplib

# ダウンロードモジュール
from sample_utils.download import download_file

from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT

from common.header import Object_ID_object, Object_ID_human, CLASSES_E, CLASSES_J

# パス名
HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./model/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./model/MobileNetSSD_deploy.prototxt.txt"

# 検出精度初期値
# DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# 色
color_white = (255, 255, 255) # 白
color_red = (255, 0, 0) # 赤
color_blue = (0, 0, 255) # 青

# フォント
# Windows
# font_name_win = "C:\\Windows\\Fonts\\msgothic.ttc" # MSゴシック
# font_name_win = "C:\\Windows\\Fonts\\msmincho.ttc" # MS明朝
# font_name_win = "C:\\Windows\\Fonts\\meiryo.ttc" # MEIRYO
# font_name_win = "C:\\Windows\\Fonts\\meiryob.ttc" # MEIRYO（太字）
font_name_win = "msgothic.ttc" # MSゴシック
# font_name_win = "meiryo.ttc" # MEIRYO

font_name_mac = "ヒラギノ丸ゴ ProN W4.ttc" # Mac
# font_name_lnx = "/usr/share/fonts/OTF/TakaoPMincho.ttf" # Linux

# streamlit Cloud（Linux）
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_name_lnx = "DejaVuSerif.ttf"
# font_name_lnx = "DejaVuSerif-Bold.ttf"
# font_name_lnx = "DejaVuSansMono.ttf"
# font_name_lnx = "DejaVuSansMono-Bold.ttf"
# font_name_lnx = "DejaVuSans.ttf"
# font_name_lnx = "DejaVuSans-Bold.ttf"

# ラベル
label_font_size = 16 # ラベルフォントサイズ

# ロゴマーク
logo_path = "./images/forex_logo_a.png" # ロゴパス名
logo_rate = 0.15 # 倍率
logo_margin = 5 # ロゴ表示マージン

# メール設定
mail_host = st.secrets.mail_settings.mail_host
mail_port = st.secrets.mail_settings.mail_port
mail_from = st.secrets.mail_settings.mail_from
mail_pass = st.secrets.mail_settings.mail_pass

# 合計人数
total_key = "total_number"
total_num = 0
total_max = 10000

# キュー
result_queue: queue.Queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.

# グローバル変数
confidence_threshold = None
plat = None
labelfont = None
logo_width = None
logo_pil = None
net = None

# メール送信
def sendMail(mail_to, subject, msg):

    try:
        msg = MIMEText(msg, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = mail_from
        msg["To"] = mail_to

        # メールサーバー
        server = smtplib.SMTP(mail_host, mail_port)

        try:
            #TLS認証
            server.ehlo()
            server.starttls()
            server.ehlo()

            #ログイン
            server.login(mail_from, mail_pass)

            # raise ValueError("テスト") # @@Debug

            # メール送信
            server.send_message(msg)

        except Exception as e:
            raise e

        finally:
            # 閉じる
            server.quit()

    except Exception as e:
        st.error("sendMail：" + str(e))

# 合計人数設定
def setTotal(num):
    global total_num

    try:
        if num != 0: # 0以外？
            # インクリメント
            total_num += 1

            # 補正
            total_num = total_num % total_max
            if total_num == 0:
                total_num += 1

        else: # リセット
            total_num = 0

        # セッション保持
        st.session_state[total_key] = total_num

    except Exception as e:
        st.error(e)

# 物体抽出
def extractionObject(cimage, detections):
    objects = [] # 物体配列
    num = 0 # 人数

    try:
        (h, w) = cimage.shape[:2]

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2] # 精度

            if confidence > confidence_threshold: # 設定精度以上？
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                idx = int(detections[0, 0, i, 1])
                ename = CLASSES_E[idx]
                jname = CLASSES_J[idx]

                if ename == "person": # 人間？
                    num += 1
                    col = color_red
                else:
                    col = color_blue

                # 物体追加
                objects.append((startX, startY, endX, endY, ename, jname, col, confidence))

    except Exception as e:
        pass

    return objects, num

# 結果描画
def drawingResult(src, objects):
    # （参考）
    # https://note.com/npaka/n/nddb33be1b782

    try:
        # 背景をPIL形式に変換
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pil_src = Image.fromarray(src)

        try:
            draw = ImageDraw.Draw(pil_src)

            # 物体取得
            for (startX, startY, endX, endY, ename, jname, col, confidence) in objects:
                # ラベル
                if plat in ("linux", "linux2"): # Linux？（日本語フォントなし）
                    label = ename # 英語
                else:
                    label = jname # 日本語

                # ラベル描画
                y = startY - (label_font_size+1) if startY - (label_font_size+1) > (label_font_size+1) else startY + (label_font_size+1)
                draw.text(xy = (startX, y), text = label, fill = col, font = labelfont)

                # 枠描画
                draw.rectangle([(startX, startY), (endX, endY)], outline=col, width=2)

            # ロゴマークを合成
            src_height, src_width = src.shape[:2]
            logo_pos = (src_width - logo_width - logo_margin, logo_margin)
            pil_src.paste(logo_pil, logo_pos, logo_pil)

        except Exception as e:
            pass

        # OpenCV形式に変換
        return cv2.cvtColor(np.asarray(pil_src), cv2.COLOR_RGB2BGR)

    except Exception as e:
        pass

# コールバック処理（人物検知）
def callback_human(frame: av.VideoFrame) -> av.VideoFrame:

    try:
        # 画像変換
        cimage = frame.to_ndarray(format="bgr24")

        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(cimage, (300, 300)), 0.007843, (300, 300), 127.5)

            # 物体検出
            net.setInput(blob)
            detections = net.forward()

            # 物体抽出
            objects, num = extractionObject(cimage, detections)

            # 結果描画
            cimage = drawingResult(cimage, objects)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            # result_queue.put(result)  # TODO:
            result_queue.put(num)  # TODO:

        except Exception as e:
            pass

        return av.VideoFrame.from_ndarray(cimage, format="bgr24")

    except Exception as e:
        pass

# メイン処理（人物検知）
def appmain_human():
    global confidence_threshold
    global plat
    global labelfont
    global logo_width
    global logo_pil
    global net
    global total_num

    try:
        # ロガー
        # logger = logging.getLogger(__name__)

        # プラットフォーム
        plat = sys.platform

        # ファイルダウンロード
        download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564) # 学習モデル
        download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353) # プロトコル

        # Session-specific caching
        cache_key = "object_detection_dnn"
        if cache_key in st.session_state:
            net = st.session_state[cache_key]
        else:
            net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
            st.session_state[cache_key] = net

        # 合計人数取得
        if total_key in st.session_state: # セッションあり？
            # セッション取得
            total_num = st.session_state[total_key]
        else:
            setTotal(0) # リセット

        # フォント
        if plat == "win32": # Windows
            font_name = font_name_win
        if plat == "darwin": # Mac
            font_name = font_name_mac
        if plat in ("linux", "linux2"): # Linux
            font_name = font_name_lnx

        # ラベルフォント
        labelfont = ImageFont.truetype(font_name, label_font_size)

        # ロゴマーク読み込み
        logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        logo_image = cv2.resize(logo_image, dsize=None, fx=logo_rate, fy=logo_rate)
        logo_height, logo_width = logo_image.shape[:2]
        # PIL形式に変換
        logo_image = cv2.cvtColor(logo_image, cv2.COLOR_BGRA2RGBA)
        logo_pil = Image.fromarray(logo_image)

        # タイトル表示
        st.subheader("みまもりくん ＜デモ＞")

        # 状態表示
        labels_placeholder = st.empty()
        # 映像表示
        streaming_placeholder = st.empty()
        # スライダー表示
        confidence_threshold = st.slider(
            "精度", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
        )

        # メールアドレス
        mailto_placeholder = st.text_input("宛先メールアドレス", "")
        # メール送信
        send_flag_placeholder = st.checkbox("人が見つかったときにメールを送信する")

        # 映像表示
        with streaming_placeholder.container():
            # WEBカメラ
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_frame_callback=callback_human,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                translations={
                    "start": "開始",
                    "stop": "停止",
                    "select_device": "カメラ切替",
                    "media_api_not_available": "Media APIが利用できない環境です",
                    "device_ask_permission": "メディアデバイスへのアクセスを許可してください",
                    "device_not_available": "メディアデバイスを利用できません",
                    "device_access_denied": "メディアデバイスへのアクセスが拒否されました",
                },
            )

        if webrtc_ctx.state.playing: # 映像配信中？
            panels = labels_placeholder

            while webrtc_ctx.state.playing: # 配信中
                try:
                    # キューの取得
                    nums = result_queue.get(timeout=1.0) # 人数取得
                except queue.Empty:
                    nums = 0

                if nums > 0: # 人がいる？
                    panels.error("人を発見！")

                    if total_num == 0: # 初回？
                        if send_flag_placeholder: # メール送信あり？
                            if mailto_placeholder != "": # 送信メールアドレスあり？
                                subject = "みまもりくん"
                                msg = "人を発見しました。"
                                #メール送信
                                sendMail(mailto_placeholder, subject, msg)

                    if send_flag_placeholder: # メール送信あり？
                        # 合計人数更新
                        setTotal(1)
                    else:
                        # 合計人数リセット
                        setTotal(0)

                else: # 人がいない
                    panels.info("安全です")

                    # # 合計人数リセット
                    # setTotal(0)

            # 合計人数リセット
            setTotal(0)

        else: # 非配信中
            # 合計人数リセット
            setTotal(0)

    except Exception as e:
        st.error("appmain：" + str(e))


@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections

@st.cache
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels

# コールバック（物体検知）
def callback_object(frame):
    img = frame.to_ndarray(format="bgr24")

    detections = process_image(img)
    # img, labels = annotate_image(img, detections, confidence_threshold)
    # img, labels = annotate_image(img, detections, 0.5)
    img, labels = annotate_image(img, detections, 0.1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 物体検知メイン
def appmain_object():
    st.title("みまもりくん")
    st.subheader('＜物体検知＞')

    # ファイルダウンロード
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564) # 学習モデル
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353) # プロトコル

    webrtc_streamer(
        key="video",
        video_frame_callback=callback_object,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

# メイン
def appmain(id):

    if id == Object_ID_object: # 物体検知？
        appmain_object()

    elif id == Object_ID_human: # 人物検知？
        appmain_human()
