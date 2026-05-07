import os
import time
import subprocess
import cv2
from ultralytics import YOLO

os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 모델 로드
model_base = YOLO('yolov10n.pt')
model_custom = YOLO('best.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_msg_time = 0
K_FACTOR = 1800

print("🚀 전체 화면 + 탐지 최적화 모드 시작")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [수정 2] 잘린 물체도 인식하도록 imgsz 유지 및 유연한 탐지
        res_base = model_base.predict(frame, conf=0.25, imgsz=320, verbose=False)
        res_custom = model_custom.predict(frame, conf=0.5, imgsz=320, verbose=False)
        
        final_target = None
        max_priority_score = 0

        # 모든 탐지 결과를 하나의 리스트에 담기
        detections = []

        # 기존 모델 결과 담기
        if res_base[0].boxes:
            for box in res_base[0].boxes:
                detections.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'conf': float(box.conf[0]),
                    'label': res_base[0].names[int(box.cls[0])],
                    'source': 'BASE'
                })

        # 커스텀 모델 결과 담기
        if res_custom[0].boxes:
            for box in res_custom[0].boxes:
                label = res_custom[0].names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # [수정 4] lamp 확실도 조정: lamp일 경우 훨씬 더 엄격한 기준 적용
                if label == 'lamp' and conf < 0.75:
                    continue
                    
                detections.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'conf': conf,
                    'label': label,
                    'source': 'CUSTOM'
                })

        # [수정 1 & 3] 중복 제거 및 가장 가까운(큰) 물체 선정
        # 면적(Area) 기준으로 정렬하여 가장 큰 것부터 검사
        detections.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]), reverse=True)

        seen_boxes = [] # 겹침 확인용

        for det in detections:
            x1, y1, x2, y2 = det['box']
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            # [수정 1] 기존 인식(BASE)과 학습 인식(CUSTOM)이 겹치면 BASE 우선
            # 간단하게 이미 처리된 박스 영역과 50% 이상 겹치면 무시
            is_overlap = False
            for seen in seen_boxes:
                # 겹치는 영역 계산 (IoU 방식)
                ix1 = max(x1, seen[0])
                iy1 = max(y1, seen[1])
                ix2 = min(x2, seen[2])
                iy2 = min(y2, seen[3])
                
                if ix2 > ix1 and iy2 > iy1:
                    i_area = (ix2 - ix1) * (iy2 - iy1)
                    if i_area / area > 0.5: # 50% 이상 겹치면
                        is_overlap = True
                        break
            
            if is_overlap and det['source'] == 'CUSTOM':
                continue # 이미 BASE가 잡았으므로 CUSTOM은 무시

            # 시각화 (BASE: 초록, CUSTOM: 빨강)
            color = (0, 255, 0) if det['source'] == 'BASE' else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{det['label']}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # [수정 3] 가장 앞에 있는(면적이 가장 큰) 물체를 타겟으로 선정
            # 정렬을 이미 해두었으므로 첫 번째 유효한 박스가 타겟이 됨
            if final_target is None:
                final_target = (det['label'], w, det['conf'], det['source'])
            
            seen_boxes.append(det['box'])

        if final_target:
            label, w, conf, source = final_target
            distance = (40 * K_FACTOR) / w if w > 0 else 0
            
            print(f"🎯 [{source}] {label}({conf:.2f}) | {int(distance)}cm")
            cv2.putText(frame, f"TARGET: {label} ({int(distance)}cm)", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Laptop View", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\n👋 종료")
finally:
    cap.release()
    cv2.destroyAllWindows()
