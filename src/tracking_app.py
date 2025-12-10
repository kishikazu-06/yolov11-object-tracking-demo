from ultralytics import YOLO
import cv2
import os, datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict # 追加: データの履歴保存用

# --- 設定エリア ---
SOURCE = "854100-hd_1920_1080_25fps.mp4" 
model_name = 'yolo11n.pt' 

# 翻訳辞書
CLASS_NAMES_JA = {
    "person": "人", "bicycle": "自転車", "car": "車", "motorcycle": "バイク",
    "bus": "バス", "truck": "トラック", "cat": "猫", "dog": "犬", "chair": "椅子"
    # ... 必要に応じて追加 ...
}
# ------------------

# --- 追加機能設定: 軌跡 ---
track_history = defaultdict(lambda: []) # IDごとの座標履歴を保存
MAX_TRAIL_LENGTH = 30 # 過去何フレーム分の軌跡を残すか（長くすると線が長くなる）

def put_japanese_text(img, text, position, font, text_color=(255, 255, 255), active_fill=True):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox(position, text, font=font)
    if active_fill:
        draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# フォント設定
font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
try:
    font = ImageFont.truetype(font_path, 32)
    font_small = ImageFont.truetype(font_path, 20)
except OSError:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

print(f"Loading {model_name}...")
model = YOLO(model_name)

input_source = SOURCE
if isinstance(SOURCE, str) and SOURCE.isdigit():
    input_source = int(SOURCE)

cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print(f"Error opening {input_source}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0

output_filename = f"output_trails_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

print("開始します。'q'で終了、's'でスクリーンショット保存")

# スクリーンショット保存先ディレクトリ
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 追跡実行
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    annotated_frame = frame.copy()
    
    boxes = results[0].boxes
    person_count = 0
    
    if boxes is not None and boxes.id is not None:
        # IDと座標をまとめて取得 (GPU->CPU変換含む)
        track_ids = boxes.id.int().cpu().tolist()
        boxes_xyxy = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()

        # ZIPでまとめてループ処理
        for box_id, box_xyxy, cls_id in zip(track_ids, boxes_xyxy, classes):
            
            english_name = model.names[cls_id]
            label_text = CLASS_NAMES_JA.get(english_name, english_name)
            
            # 色設定 & カウント
            if english_name == 'person':
                person_count += 1
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            
            x1, y1, x2, y2 = map(int, box_xyxy)
            
            # 中心座標を計算
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2) # 足元にしたい場合は int(y2) にする

            # --- ★ここが追加機能: 軌跡の処理 ---
            track = track_history[box_id]
            track.append((center_x, center_y))
            if len(track) > MAX_TRAIL_LENGTH:
                track.pop(0) # 古い履歴を削除

            # 軌跡を描画 (複数の点を線で結ぶ)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)
            # ----------------------------------

            label_text += f" {box_id}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            annotated_frame = put_japanese_text(annotated_frame, label_text, (x1, y1 - 30), font_small, (255, 255, 255))

    count_text = f"検出人数: {person_count}人"
    annotated_frame = put_japanese_text(annotated_frame, count_text, (10, 10), font, (0, 255, 255), active_fill=True)

    out.write(annotated_frame)
    cv2.imshow("YOLO11 Japanese Tracking & Trails", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        fname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.png')
        save_path = os.path.join(RESULTS_DIR, fname)
        cv2.imwrite(save_path, annotated_frame)
        print(f"📸 スクリーンショット保存: {save_path}", flush=True)
    elif key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()