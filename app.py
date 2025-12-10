import cv2

# HOG（Histrogram of Oriented Gradients） + SVM を用いた人物検出器を初期化
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # Windowsでも互換性のためCAP_DSHOW指定

# カメラを起動
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("カメラが起動できませんでした。")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした。")
        break
    
    # 人物検出（帰ってくるのは矩形のリスト）
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

    # 検出された人物に枠を描画
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 映像を表示
    cv2.imshow("Human Detection", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 解放処理
cap.release()
cv2.destroyAllWindows()