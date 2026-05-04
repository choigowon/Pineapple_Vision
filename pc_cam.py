import os
import time
import subprocess
import cv2
from ultralytics import YOLO

# 1. 환경 설정
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 2. 모델 로드
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)
# 화면 폭을 640으로 설정 (좌/중/우 구분을 위해)
WIDTH = 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n📡 [지능형 경로 장애물 감지 모드] 시작")

last_msg_time = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 80종 전체 인식, conf 0.4 (약간 높여서 헛것 보는 것 방지)
        results = model.predict(frame, conf=0.4, imgsz=320, verbose=False)
        
        for box in results[0].boxes:
            # 1. 위치 분석 (화면의 x축 중심점 찾기)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            w = x2 - x1
            
            # 2. 거리 계산
            distance = (40 * 550) / w if w > 0 else 0
            
            # 3. 구역 분할 (좌 0~213 / 중 214~426 / 우 427~640)
            if center_x < WIDTH / 3:
                position = "왼쪽"
            elif center_x < (WIDTH / 3) * 2:
                position = "정면"
            else:
                position = "오른쪽"

            # 4. 경고 로직 (위험 거리 1.5m 이내)
            if distance < 150:
                curr_time = time.time()
                # 정면에 있는 물체는 더 자주, 더 민감하게 경고
                threshold = 1.5 if position == "정면" else 3.0
                
                if curr_time - last_msg_time > threshold:
                    # 이름이 틀릴 수 있으므로 "장애물"이라 칭하되 위치를 강조
                    label = model.names[int(box.cls[0])]
                    msg = f"{position}에 {label} 주의"
                    
                    print(f"⚠️ [경고] {msg} ({int(distance)}cm)")
                    # espeak로 한국어 안내 (위치 정보 포함)
                    subprocess.Popen(['espeak', '-v', 'ko', msg])
                    last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n종료")
finally:
    cap.release()
