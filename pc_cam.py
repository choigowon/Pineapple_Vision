import os
import time
import subprocess
import cv2
from ultralytics import YOLO

os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

model_base = YOLO('yolov10n.pt')   # 범용 모델
model_custom = YOLO('best.pt')     # 커스텀 모델

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

LANE_L, LANE_R = 160, 480 
last_msg_time = 0
K_FACTOR = 1800

print("🚀 [우선순위 조정 모드] 범용 사물 인식 강화")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        roi = frame[:, LANE_L:LANE_R]
        
        # [수정 1] 기본 모델의 conf는 낮추고(더 잘 찾게), 커스텀은 높임(확실할 때만)
        res_base = model_base.predict(roi, conf=0.4, imgsz=320, verbose=False, stream=True)
        res_custom = model_custom.predict(roi, conf=0.75, imgsz=448, verbose=False, stream=True)
        
        # 결과를 분리해서 관리
        base_list = list(res_base)
        custom_list = list(res_custom)
        
        final_target = None
        max_area = 0

        # [수정 2] 우선순위 로직: 기본 모델(사람 등)의 결과를 먼저 검사
        for r in base_list:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                if w < 40: continue # 너무 작으면 무시
                
                area = w * h
                if area > max_area:
                    max_area = area
                    final_target = (r.names[int(box.cls[0])], w, box.conf[0], "BASE")

        # [수정 3] 기본 모델이 찾은 게 없을 때만 커스텀 모델 결과를 반영하거나, 
        # 커스텀 모델의 면적이 압도적으로 클 때만 교체
        for r in custom_list:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                if w < 60: continue # 커스텀은 더 엄격하게(크기 필터 강화)
                
                area = w * h
                # 기본 모델이 찾은 게 없거나, 커스텀 물체가 훨씬 가까울 때만 선택
                if final_target is None or (area > max_area * 1.5):
                    if area > max_area:
                        max_area = area
                        final_target = (r.names[int(box.cls[0])], w, box.conf[0], "CUSTOM")

        if final_target:
            label, w, conf, source = final_target
            distance = (40 * K_FACTOR) / w if w > 0 else 0
            
            # 터미널에서 어떤 모델이 잡았는지 확인용 (BASE면 기존 모델, CUSTOM이면 학습 모델)
            print(f"[{source}] 인식: {label}({conf:.2f}) | 거리: {int(distance)}cm")

            if distance < 150:
                curr_time = time.time()
                if curr_time - last_msg_time > 2.0:
                    msg = f"정면 {label}"
                    subprocess.Popen(['espeak', '-v', 'ko', '-s', '180', msg])
                    last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n👋 종료")
finally:
    cap.release()
