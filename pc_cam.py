import os
import time
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 시스템 가속 설정
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
model = YOLO('yolov10n.pt') 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 딜레이 제거 핵심
WIDTH, HEIGHT = 640, 480

# [중앙 영역] 120도 광각 렌즈에서 왜곡이 가장 적은 구역 (중앙 40%)
LANE_L, LANE_R = 192, 448 

last_msg_time = 0

print("\n🎯 [시연 최적화: 속도+거리 정밀보정] 모드 시작")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [최적화 1] 중앙만 잘라서 AI 부담 덜어주기
        roi = frame[:, LANE_L:LANE_R]
        
        # [최적화 2] imgsz=256: 160(멍청함)과 320(무거움)의 완벽한 중간 지점
        results = model.predict(roi, conf=0.45, imgsz=256, verbose=False, stream=True)
        
        for r in results:
            if not r.boxes: continue
            
            # [최적화 3] 가장 면적이 큰(가까운) 물체 하나에만 집중
            best_box = max(r.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            
            label_name = model.names[int(best_box.cls[0])]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            w = x2 - x1
            
            # [거리 정밀 보정] 
            # 팀장님 렌즈(120도)에 맞춘 '광각 보정 계수'입니다. 
            # 1미터 앞 물체가 100cm로 나올 때까지 이 210 숫자를 조절하세요.
            distance = (40 * 210) / w if w > 0 else 0
            
            # [바닥 좌표 보정] 물체가 화면 아래쪽에 있을수록 실제로는 가깝습니다.
            y_factor = y2 / 480 
            distance = distance * (1.2 - y_factor)

            # 1.4미터 이내 장애물만 안내
            if distance < 140:
                curr_time = time.time()
                if curr_time - last_msg_time > 1.0:
                    msg = f"정면 {label_name}"
                    print(f"⚠️ {msg} | 보정거리: {int(distance)}cm")
                    
                    # 음성 속도 200 (빠릿빠릿)
                    subprocess.Popen(['espeak', '-v', 'ko', '-s', '200', msg])
                    last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n👋 시연 종료 준비 완료.")
finally:
    cap.release()
