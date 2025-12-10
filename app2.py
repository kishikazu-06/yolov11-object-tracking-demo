import torch
import cv2
import os, datetime
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# モデル選択 (yolov5s, yolov5m, yolov5l 等に変更可能)
# s=軽量, m=中程度, l=高精度だが重い
model_name = 'yolov5m' 
print(f"Loading {model_name}...")
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

# 検出対象を制限したい場合はリストで指定 (例: [0] は person)
# None にすると学習済みの全クラス(80種類)を検出
model.classes = None 

# カメラを起動
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("カメラが起動できませんでした。")
    exit()

# 動画保存の設定 (mp4形式)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0 # 取得できなかった場合のフォールバック

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4", fourcc, fps, (width, height))
print("Recording to output.mp4...")

class_colors = {
'person': (0, 255, 0),
'car': (0, 0, 255),
'dog': (255, 128, 0),
'cat': (0, 128, 255),
'bicycle': (255, 0, 255)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv5はRGB入力なので変換
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 推論
    results = model(img)
    
    # 検出結果をPandas形式で取得
    df = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, conf, class, name

    # 検出結果を描画
    for _, row in df.iterrows():
        # 名前でフィルタリングせず全て表示
        name = row['name']
        conf = row['confidence']
        
        x1, y1, x2, y2 = map(int, (row['xmin'], row['ymin'], row['xmax'], row['ymax']))
        color = class_colors.get(name, (0, 255, 0))
        thickness = max(2, int(1 + conf *3))

        # 枠とラベルを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    # 録画 (描画後のフレームを保存)
    out.write(frame)

    # 表示
    cv2.imshow("YOLOv5 Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    # 's' または 'S' (Caps/Shift) で保存
    if key == ord('s') or key == ord('S'):
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        fname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.jpg')
        save_path = os.path.join(desktop, fname)
        cv2.imwrite(save_path, frame)
        print(f"📸画像を保存しました: {save_path}", flush=True)
    elif key == ord('q'):
        break

cap.release()
out.release() # 録画ファイルの解放
cv2.destroyAllWindows()