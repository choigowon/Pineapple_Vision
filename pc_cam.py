import cv2
import numpy as np

# 1. 모델 로드 (OpenCV DNN 모듈 사용)
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# 2. 인식 가능한 사물 목록
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 3. 카메라 설정
cap = cv2.VideoCapture(0)

print("시각장애인용 비전 AI 시작... (종료: Ctrl+C)")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 이미지 전처리 (300x300 크기로 변환)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # AI 추론 시작
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # 신뢰도가 50% 이상인 경우만 처리
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            
            # 터미널에 결과 출력 (시각장애인용 음성/진동 모듈이 들어갈 자리)
            print(f"[감지] {label}: {confidence * 100:.2f}%")

            # 화면에 박스 그리기 (VNC 사용 시 확인용)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # VNC나 모니터 연결 시에만 창 띄우기
    # cv2.imshow("Frame", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
