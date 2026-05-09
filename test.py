import os
import cv2
from ultralytics import YOLO

# 1. 환경 변수 설정 (라즈베리 파이 가속)
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 2. 모델 로드 (YOLOv10n은 가볍지만, 두 개를 돌리면 여전히 느립니다)
model_base = YOLO('yolov10n.pt')
model_custom = YOLO('best.pt')

cap = cv2.VideoCapture(0)
# 카메라 해상도 자체를 낮게 잡으면 회전 및 전처리 속도가 빨라집니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

K_FACTOR = 1800

print("🚀 최적화 모드 시작 (imgsz=320, augment=OFF)")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [최적화 1] 90도 회전
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # [최적화 2] imgsz 하향 (320), augment 제거, stream=True로 메모리 효율화
        # 두 모델을 동시에 돌리는 것은 여전히 부하가 크므로, 
        # 만약 너무 느리면 하나를 주석 처리하세요.
        res_base = model_base.predict(frame, conf=0.5, imgsz=320, verbose=False)[0]
        res_custom = model_custom.predict(frame, conf=0.3, imgsz=320, verbose=False)[0]
        
        detections = []

        # 결과 처리 통합 (Base 모델)
        if res_base.boxes:
            for box in res_base.boxes:
                detections.append({
                    'box': box.xyxy[0].tolist(), # cpu().numpy()보다 가벼운 리스트 변환
                    'conf': float(box.conf[0]),
                    'label': res_base.names[int(box.cls[0])],
                    'source': 'BASE'
                })

        # 결과 처리 통합 (Custom 모델)
        if res_custom.boxes:
            for box in res_custom.boxes:
                label = res_custom.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if label == 'lamp' and conf < 0.6: continue
                    
                detections.append({
                    'box': box.xyxy[0].tolist(),
                    'conf': conf,
                    'label': label,
                    'source': 'CUSTOM'
                })

        # 면적 계산 및 정렬 (NMS와 유사한 효과)
        detections.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]), reverse=True)

        final_target = None
        seen_boxes = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            
            # 간단한 중복 체크
            is_overlap = False
            for seen in seen_boxes:
                # 겹치는 영역 계산을 생략하고 중심점 포함 여부 등으로 간소화 가능하나 일단 유지
                if x1 < seen[2] and x2 > seen[0] and y1 < seen[3] and y2 > seen[1]:
                    is_overlap = True
                    break
            
            if is_overlap and det['source'] == 'CUSTOM': continue

            # 시각화 (느릴 경우 이 부분을 생략하면 더 빨라집니다)
            color = (0, 255, 0) if det['source'] == 'BASE' else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            if final_target is None:
                final_target = (det['label'], (x2-x1))
            seen_boxes.append([x1, y1, x2, y2])

        if final_target:
            label, w = final_target
            distance = (40 * K_FACTOR) / w if w > 0 else 0
            cv2.putText(frame, f"{label} {int(distance)}cm", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Fast View", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

except KeyboardInterrupt:
    print("\n👋 종료")
finally:
    cap.release()
    cv2.destroyAllWindows()
