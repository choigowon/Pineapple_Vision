import os
import time
import subprocess
import cv2
from ultralytics import YOLO

# 1. 환경 변수 (이건 필수!)
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 2. 모델 로딩 (이미 검증된 v10n 사용)
model = YOLO('yolov10n.pt') 

cap = cv2.VideoCapture(0)
# 해상도를 480p로 낮춰서 전송 속도 자체를 올림
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 중앙 좁은 통로 설정 (120도 광각 대응: 중앙 30%만 집중)
LANE_LEFT = 640 * 0.35
LANE_RIGHT = 640 * 0.65

last_msg_time = 0
frame_count = 0

print("\n🚀 [시연용 최종 버전: YOLOv10n 하이퍼 최적화] 가동")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        # [속도 향상 핵심] 3프레임당 1번만 연산 (부드러움 유지하면서 CPU 부하 감소)
        if frame_count % 3 != 0:
            continue

        # [정확도 향상 핵심] imgsz=320으로 속도를 챙기되, 중앙만 슬라이싱해서 추론
        # 화면 전체를 다 보지 않고 중앙 핵심 영역만 AI에게 넘깁니다.
        roi = frame[:, int(LANE_LEFT):int(LANE_RIGHT)]
        results = model.predict(roi, conf=0.45, imgsz=320, verbose=False)
        
        targets = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                w = box.xyxy[0][2] - box.xyxy[0][0]
                distance = (40 * 550) / w if w > 0 else 0
                
                if distance < 180:
                    targets.append((label, distance))

        if targets:
            targets.sort(key=lambda x: x[1])
            label, dist = targets[0]
            
            curr_time = time.time()
            if curr_time - last_msg_time > 1.5:
                msg = f"정면 {label} 주의"
                print(f"⚠️ [FINAL] {msg} ({int(dist)}cm)")
                # 음성 안내를 별도 프로세스로 던져서 화면 끊김 방지
                subprocess.Popen(['espeak', '-v', 'ko', msg])
                last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n👋 시연 종료")
finally:
    cap.release()
