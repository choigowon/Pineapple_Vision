import cv2
from ultralytics import YOLOWorld

def run_velog_optimized_camera():
    print("YOLOv8s-worldv2 모델 로드 중...")
    model = YOLOWorld('yolov8s-worldv2.pt')

    # [2번 방법: 서술형 프롬프트 최적화] 
    # 단어 하나가 아닌 문맥(Context)을 제공하여 YOLO-World의 인지 능력을 극대화합니다.
    custom_classes = [
        # --- Door 강화 서술형 프롬프트 ---
        "a door inside a room", 
        "an entrance door of a building", 
        "a closed wooden door", 
        "a glass door with a frame",
        
        # --- Threshold 강화 서술형 프롬프트 ---
        "a door sill on the floor", 
        "a raised threshold between rooms",
        
        # --- 기존 요청 클래스 ---
        "person", "bicycle", "car", "motorcycle", "refrigerator",
        "bus", "truck", "traffic light", "sink", "stop sign", "bench",
        "chair", "potted plant", "bed", "dining table", "tv", 
        "laptop", "cell phone", "microwave",
        "bollard", "traffic cone", "utility pole", "water puddle", "shelf", "stair", "tree",
        
        # --- Box 강화 서술형 프롬프트 ---
        "a cardboard box on the floor"
    ]
    
    # 모델에 텍스트 임베딩 주입
    model.set_classes(custom_classes)
    print("서술형 프롬프트 클래스 지정 완료!")

    # 화면 표시용 맵핑 정의 (서술형 프롬프트로 검출된 결과를 원래 단어로 변환)
    door_prompts = [
        "a door inside a room", 
        "an entrance door of a building", 
        "a closed wooden door", 
        "a glass door with a frame"
    ]
    threshold_prompts = [
        "a door sill on the floor", 
        "a raised threshold between rooms"
    ]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 연산량 감소를 위한 해상도 조정 (640x480) -> 끊김 현상 대폭 완화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("실시간 인식을 시작합니다. 종료하려면 창에서 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [블로그 꿀팁 반영] 
        # 1. 절대 RGB로 변환하지 않고 frame(BGR)을 그대로 넣어서 정확도 유지!
        # 2. conf를 0.20으로 낮춰 잡기 힘든 Door를 먼저 걸려들게 한 후, iou로 억제
        results = model.predict(frame, conf=0.20, iou=0.50, stream=True, verbose=False)

        # 결과 커스텀 시각화 (인식 정확도를 눈으로 확인하기 위함)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 좌표 및 검출 정보 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                
                # 검출된 서술형 프롬프트 원본 이름
                class_name = custom_classes[cls_id]
                
                # 유사 문장들을 대표 이름으로 치환하여 화면 표시용 이름 확정
                if class_name in door_prompts:
                    class_name = "door"
                elif class_name in threshold_prompts:
                    class_name = "threshold"
                elif class_name == "a cardboard box on the floor":
                    class_name = "box"

                # 화면에 바운딩 박스와 라벨 표시
                # 문(door) 계열은 눈에 잘 띄게 파란색(255, 0, 0), 나머지는 초록색
                color = (255, 0, 0) if class_name == "door" else (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 화면 출력
        cv2.imshow("YOLOv8s-World (Prompt Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_velog_optimized_camera()
