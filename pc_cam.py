import os
import time
import subprocess
import cv2
from ultralytics import YOLO

# 1. 환경 설정
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 2. 모델 로드 (가장 가벼운 v10n)
model = YOLO('yolov10n.pt') 

cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30) # 카메라 프레임 강제 고정

# [설정] 중앙 영역 (왜곡이 적은 구간)
LANE_LEFT, LANE_RIGHT = 200, 440

last_msg_time = 0

print("\n🚀 [속도/거리 최적화 버전] 가동!")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [속도 최적화 1] 2프레임당 1번 추론 (0.06초 절약)
        # 더 빠릿한 반응을 위해 건너뛰기를 최소화
        
        # [속도 최적화 2] ROI(관심영역) 추출 - 중앙만 딱 잘라서 전달
        roi = frame[:, LANE_LEFT:LANE_RIGHT]
        
        # [속도 최적화 3] imgsz를 160~224로 낮춤 (비약적인 속도 상승)
        results = model.predict(roi, conf=0.4, imgsz=224, verbose=False, stream=True)
        
        targets = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                w = box.xyxy[0][2] - box.xyxy[0][0]
                
                # [거리 보정] 
                # 거리가 너무 크게 나오면 아래의 700(초점계수) 숫자를 400~500으로 낮추세요.
                # 120도 광각은 물체가 실제보다 작게 찍히므로 이 상수가 작아야 거리가 줄어듭니다.
                distance = (40 * 450) / w if w > 0 else 0
                
                # 1.5미터(150cm) 이내만 타겟팅
                if distance < 150:
                    targets.append((label, distance))

        if targets:
            targets.sort(key=lambda x: x[1])
            label, dist = targets[0]
            
            curr_time = time.time()
            # 안내 간격을 1초로 단축 (빠릿빠릿한 반응)
            if curr_time - last_msg_time > 1.0:
                msg = f"정면 {label}" # '주의' 빼고 핵심만 전달해서 안내 속도 상승
                print(f"⚠️ {msg} | 거리: {int(dist)}cm")
                
                # espeak 옵션 추가: -s 170 (말하기 속도 증가)
                subprocess.Popen(['espeak', '-v', 'ko', '-s', '170', msg])
                last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n종료")
finally:
    cap.release()
