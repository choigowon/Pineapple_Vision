import os
import time
import subprocess
import cv2
from ultralytics import YOLO

# 1. 환경 변수 최적화 (라즈베리파이 전용)
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 2. 모델 로드 (가장 가벼운 Nano 모델 사용 - 속도가 생명입니다)
# 80개 클래스 전체를 인식하기 위해 classes 옵션을 제거합니다.
model = YOLO('yolov8n.pt') 

# 3. 카메라 설정 (광각 확보를 위해 해상도 유지)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🔥 [전체 사물 무제한 인식 모드] 가동!")
print("YOLOv8의 80개 카테고리를 모두 감시합니다.")

last_msg_time = 0
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        # 성능 최적화: 4프레임당 1번 분석 (약간 더 촘촘하게 수정)
        if frame_count % 4 != 0: continue 

        # [수정] 모든 클래스(80종) 인식, conf 0.35로 낮춰서 '뭐든' 잡히게 설정
        results = model.predict(frame, conf=0.35, imgsz=320, verbose=False)
        
        for box in results[0].boxes:
            # 사물 이름 (영어)
            label = model.names[int(box.cls[0])]
            
            # 거리 계산 (Bounding Box의 가로폭 기준)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            
            # 보행 보조용 거리 환산 (대략적)
            distance = (40 * 550) / w if w > 0 else 0
            
            # 1.5미터 이내에 '무엇이든' 포착되면 경고
            if distance < 150:
                curr_time = time.time()
                if curr_time - last_msg_time > 2:
                    # 한국어 변환 없이 바로 라벨 출력 및 음성 안내
                    msg = f"Watch out, {label}"
                    print(f"⚠️ {msg} ({int(distance)}cm)")
                    
                    # espeak를 영어 모드로 하여 더 정확하고 빠르게 발음
                    subprocess.Popen(['espeak', '-v', 'en', msg])
                    last_msg_time = curr_time

except KeyboardInterrupt:
    print("\n사용자에 의해 종료되었습니다.")
finally:
    cap.release()
