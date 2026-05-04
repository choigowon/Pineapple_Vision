import os
import time
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 시스템 최적화
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 2. 모델 로드 (가장 효율적인 YOLOv10n)
model = YOLO('yolov10n.pt') 

cap = cv2.VideoCapture(0)
# 버퍼를 1로 설정하여 딜레이 제거
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
WIDTH, HEIGHT = 640, 480

# [세팅] 중앙 집중형 시야 (전체 640 중 200~440 구역만 집중)
LANE_L, LANE_R = 200, 440

last_msg_time = 0

print("\n💎 [Hybrid Ultimate 모드] 가동: 속도+정확도+거리보정")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [Hybrid 1] 중앙 ROI만 추출하여 연산량 감소 (MobileNet 방식)
        roi = frame[:, LANE_L:LANE_R]
        
        # [Hybrid 2] 정확도를 위해 imgsz=320 유지하되 stream 모드로 속도 보완 (YOLO 방식)
        results = model.predict(roi, conf=0.45, imgsz=320, verbose=False, stream=True)
        
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0: continue
            
            # [Hybrid 3] 가장 큰 객체(가까운 물체) 선택
            best_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]))
            
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            label = model.names[int(best_box.cls[0])]
            w = x2 - x1
            h = y2 - y1

            # [거리 정밀 보정 - 삼각법 적용]
            # 120도 광각은 물체가 작게 찍히므로 보정 계수를 정교하게 잡아야 합니다.
            # 1m 거리에서 사람/물체의 픽셀 폭이 대략 120-150px라면 아래 계수가 적당합니다.
            focal_length_factor = 320 # 1m 기준 픽셀 폭 보정값 (실측 후 조절)
            distance = (40 * focal_length_factor) / w if w > 0 else 0
            
            # 바닥 점(y2)을 활용한 추가 보정 (멀수록 y값이 작고, 가까울수록 큼)
            # 지면으로부터의 거리를 반영하여 박스 폭의 오차를 보정합니다.
            if y2 < HEIGHT * 0.5: # 화면 위쪽에 있다면 실제보다 멀리 있는 것
                distance *= 1.2

            if distance < 150: # 1.5미터 이내
                curr_time = time.time()
                # 안내 주기 최적화 (정면은 1초, 거리가 아주 가까우면 0.7초)
                msg_interval = 0.7 if distance < 80 else 1.0
                
                if curr_time - last_msg_time > msg_interval:
                    msg = f"정면 {label}"
                    print(f"📡 [DET] {msg} | {int(distance)}cm")
                    
                    # -s 180: 빠르지만 정확한 발음
                    subprocess.Popen(['espeak', '-v', 'ko', '-s', '180', msg])
                    last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n👋 종료")
finally:
    cap.release()
