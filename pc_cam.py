import cv2
from ultralytics import YOLO
import pyttsx3
import threading

# 1. 초기화 (모델 로드 및 TTS 엔진)
model = YOLO('yolov8n.pt')  # 가장 가벼운 Nano 모델 사용
engine = pyttsx3.init()

def speak(text):
    """음성 안내를 별도 스레드에서 실행 (병목 방지)"""
    engine.say(text)
    engine.runAndWait()

# 웹캠 연결 (0번은 내장 웹캠)
cap = cv2.VideoCapture(0)

last_announced = "" # 중복 안내 방지용 변수

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. AI 추론 (실시간 인식)
    results = model(frame, conf=0.5) # 신뢰도 50% 이상만 표시

    # 3. 결과 시각화 및 음성 안내 로직
    annotated_frame = results[0].plot()
    
    for result in results[0].boxes:
        class_id = int(result.cls[0])
        label = model.names[class_id]
        
        # 새로운 물체가 나타났을 때만 말하기 (간단한 필터링)
        if label != last_announced:
            threading.Thread(target=speak, args=(f"{label}이 감지되었습니다",), daemon=True).start()
            last_announced = label
            break # 한 번에 하나만 안내

    # 화면 표시
    cv2.imshow("Vision AI Guide - Test", annotated_frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()