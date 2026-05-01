import os
import time
import subprocess
import torch

# 1. 환경 및 보안 설정 (이건 이제 필수)
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    print("필수 라이브러리가 없습니다.")
    exit()

# [중요] PyTorch 보안 허용
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass

# 2. 모델 로드 (파일이 깨졌을 수 있으니 새로 강제 다운로드 권장)
print("AI 모델 정밀 로딩 중...")
# 기존 파일이 이상하면 yolov8n.pt를 삭제하고 실행하세요. 새로 받습니다.
model = YOLO('yolov8n.pt') 

def speak(text):
    print(f"🔊 음성: {text}")
    subprocess.Popen(['espeak', '-v', 'ko', text])

cap = cv2.VideoCapture(0)
# 인식률 향상을 위해 해상도를 고정합니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🚀 [정밀 인식 모드] 가동 시작!")

last_msg_time = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [인식률 튜닝] 
        # conf=0.6 : 60% 이상 확신할 때만 출력 (모든 걸 사람으로 보는 현상 방지)
        # iou=0.45 : 박스가 겹치는 걸 방지
        # classes=[0] : '사람' 클래스 번호만 뽑아내기 (0번이 사람입니다)
        results = model.predict(frame, conf=0.6, iou=0.45, classes=[0], verbose=False)
        
        found_person = False
        
        # results[0].boxes에 잡힌 건 이제 '확실한 사람'뿐입니다.
        for box in results[0].boxes:
            found_person = True
            
            # 거리 계산
            x1, y1, x2, y2 = box.xyxy[0]
            w = x2 - x1
            dist = (40 * 550) / w if w > 0 else 0
            
            if dist < 120: # 1.2미터 이내 감지
                current_time = time.time()
                if current_time - last_msg_time > 3:
                    print(f"✅ 사람 확인! 거리: {dist:.1f}cm")
                    speak("사람 주의")
                    last_msg_time = current_time

        # 사람이 아무도 없을 때의 로그 (디버깅용)
        if not found_person and time.time() % 5 < 0.05:
            print("대기 중... (현재 사람 없음)")

except KeyboardInterrupt:
    print("\n👋 종료합니다.")
finally:
    cap.release()
