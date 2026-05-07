import os
import time
import subprocess
import cv2
from ultralytics import YOLO

os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# 모델 로드 (imgsz 설정은 추론 시 적용)
model_base = YOLO('yolov10n.pt')   # 범용 모델
model_custom = YOLO('best.pt')     # 커스텀 모델

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_msg_time = 0
K_FACTOR = 1800

print("🚀 전체 화면 모드 (탐지 강화)")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [수정 1] ROI 제거, 전체 화면 사용
        # 이제 frame 전체를 분석합니다.

        # [수정 2] imgsz 320 통일, conf 임계값 대폭 낮춤 (많이 찾도록)
        res_base = model_base.predict(frame, conf=0.25, imgsz=320, verbose=False)
        res_custom = model_custom.predict(frame, conf=0.5, imgsz=320, verbose=False)
        
        detections = []
        
        final_target = None
        max_area = 0

        # [수정 3] 박스 그리기 좌표 보정 제거 (+LANE_L 삭제)

        # 기본 모델 분석
        if res_base[0].boxes:
            for box in res_base[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                # [수정 4] 크기 필터 완화
                if w < 30: continue
                
                # 초록색 박스
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                area = w * h
                # [수정 5] 우선순위 로직 단순화: 가장 큰 것 우선
                if area > max_area:
                    max_area = area
                    final_target = (res_base[0].names[int(box.cls[0])], w, float(box.conf[0]), "BASE")

        # 커스텀 모델 분석
        if res_custom[0].boxes:
            for box in res_custom[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                # [수정 4] 크기 필터 완화
                if w < 30: continue
                
                # 빨간색 박스
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                area = w * h
                # 가장 큰 것 우선
                if area > max_area:
                    max_area = area
                    final_target = (res_custom[0].names[int(box.cls[0])], w, float(box.conf[0]), "CUSTOM")

        if final_target:
            label, w, conf, source = final_target
            distance = (40 * K_FACTOR) / w if w > 0 else 0
            # [수정 6] 터미널 출력에 신뢰도 표시
            print(f"🎯 [{source}] {label}({conf:.2f}) | {int(distance)}cm")

            # 최종 타겟 노란색 박스
            cv2.putText(frame, f"{label} {int(distance)}cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Laptop View", frame)
        
        # ESC 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\n👋 종료")
finally:
    cap.release()
    cv2.destroyAllWindows()
