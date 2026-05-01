import cv2
import numpy as np
import subprocess

# 1. 모델 및 클래스 로드 (YOLOv4-Tiny)
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 가속 설정
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def speak(text):
    # 음성이 겹치지 않게 이전 음성 프로세스를 죽이고 실행 (선택사항)
    subprocess.Popen(['espeak', '-v', 'ko', text])

# 2. 카메라 설정 (USB 웹캠)
cap = cv2.VideoCapture(0)

# [중요] 5MP 카메라라도 AI 분석용으로는 640x480 혹은 320x240이 가장 빠릅니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 프레임 레이트 설정 (카메라 사양에 맞춰 안정적으로 30fps 지정)
cap.set(cv2.CAP_PROP_FPS, 30)

print("--- 5MP 컬러 카메라 최적화 AI 모드 시작 ---")
last_label = ""
frame_skip = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3프레임당 1번만 분석 (실시간성 확보)
        frame_skip += 1
        if frame_skip % 3 != 0:
            continue

        # AI 분석 (YOLOv4-Tiny 표준 입력 크기 416x416)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        best_label = ""
        max_conf = 0

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # 컬러 카메라이므로 신뢰도를 0.5로 높여서 정확하게 판정
                if confidence > 0.5:
                    label = classes[class_id]
                    if confidence > max_conf:
                        max_conf = confidence
                        best_label = label

        if best_label != "" and best_label != last_label:
            # 주요 사물 한국어 매핑
            ko_dict = {"person": "사람", "cup": "컵", "cell phone": "휴대폰", "bottle": "병", "backpack": "배낭"}
            print(f"[감지] {best_label} ({max_conf*100:.1f}%)")
            
            speak_text = ko_dict.get(best_label, best_label)
            speak(f"앞에 {speak_text}이 있습니다")
            last_label = best_label

except KeyboardInterrupt:
    print("\n시스템 종료")
finally:
    cap.release()
