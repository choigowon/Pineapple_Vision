import os
import time
import subprocess
import cv2
from ultralytics import YOLO

# 1. 환경 설정 (라즈베리파이 CPU 가속 고정)
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 2. 모델 로드 (v10n)
# 팁: 정확도가 정 아쉬우면 여기서 v10s.pt로 바꿔보세요 (단, 느려짐)
model = YOLO('yolov10n.pt') 

cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# 중앙 집중 범위 (0.35 ~ 0.65 : 광각의 왜곡을 피해 가장 정확한 정면만 응시)
LANE_LEFT = WIDTH * 0.35
LANE_RIGHT = WIDTH * 0.65

last_msg_time = 0

print("\n🚀 [최종 진화형: YOLOv10n 중앙 집중 모드] 시작")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [최적화 핵심] 
        # imgsz=320으로 연산량은 줄이되, 
        # stream=True 옵션으로 메모리 점유율을 낮춰 속도를 확보합니다.
        results = model.predict(frame, conf=0.45, imgsz=320, verbose=False, stream=True)
        
        for r in results:
            # 거리순 정렬을 위해 리스트업
            targets = []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                
                # 중앙 영역 밖은 과감히 무시
                if center_x < LANE_LEFT or center_x > LANE_RIGHT:
                    continue

                label = model.names[int(box.cls[0])]
                w = x2 - x1
                distance = (40 * 550) / w if w > 0 else 0

                if distance < 200: # 2미터 이내
                    targets.append((label, distance))

            if targets:
                # 가장 가까운 물체 선정
                targets.sort(key=lambda x: x[1])
                label, dist = targets[0]

                curr_time = time.time()
                # 안내 간격을 1.2초로 줄여 더 실시간 대응 가능하게 변경
                if curr_time - last_msg_time > 1.2:
                    msg = f"정면 {label} 주의"
                    print(f"⚠️ [ALERT] {msg} ({int(dist)}cm)")
                    # 안내는 백그라운드 실행으로 프레임 드랍 방지
                    subprocess.Popen(['espeak', '-v', 'ko', msg])
                    last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n👋 시스템을 종료합니다.")
finally:
    cap.release()
