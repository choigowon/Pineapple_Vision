import os
import cv2
from ultralytics import YOLO

# 1. 환경 설정
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 2. 모델 로드 (가장 가벼운 모델들로 구성)
model_base = YOLO('yolov10n.pt')  # 사람, 차 등 기존 인식 담당
model_custom = YOLO('best.pt')    # 킥보드, 꼬깔 등 우리 학습 사물 담당

cap = cv2.VideoCapture(0)
# 화면 크기가 작아 보였다면 640x480으로 복구합니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("🚀 통합 인식 모드 시작 (기본 사물 + 커스텀 사물)")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 화면 회전 (세로 모드)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # [핵심] 속도를 위해 imgsz를 320으로 낮추되, 
        # 두 모델을 동시에 돌리는 부하를 줄이기 위해 '객체 수 제한'을 겁니다.
        
        # 1. 기존 사물 인식 (사람, 차 등)
        res_base = model_base.predict(frame, conf=0.45, imgsz=320, verbose=False)[0]
        
        # 2. 우리 사물 인식 (스쿠터, 꼬깔 등)
        res_custom = model_custom.predict(frame, conf=0.35, imgsz=320, verbose=False)[0]
        
        # 시각화 통합
        for r in [res_base, res_custom]:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # 전동 킥보드 특화 필터
                if 'scooter' in label.lower() and conf < 0.6:
                    continue

                color = (0, 255, 0) if r == res_base else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Pineapple Vision Full", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

except Exception as e:
    print(f"❌ 에러 발생: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
