import cv2
from ultralytics import YOLOWorld

def run_velog_optimized_camera():
    print("YOLOv8s-worldv2 모델 로드 중...")
    model = YOLOWorld('yolov8s-worldv2.pt')

    # [프롬프트 마스터 튜닝]
    # 1. 70%만 보여도 인식하도록 'partially visible' 접두어 활용
    # 2. 문턱(threshold) 단어 다각화
    # 3. 요청하신 wall, window, wardrobe, clothes hanger rack 배치
    custom_classes = [
        # --- Door 강화 (일부 노출 대응) ---
        "a room door with a handle", 
        "a partially visible entrance door", 
        "a closed wooden door frame", 
        "a glass door with a metal handle",
        
        # --- 문턱(Threshold) 극한 강화 ---
        "a door sill on the floor", 
        "a floor transition strip",            # 바닥 경계 분리대 (문턱의 또 다른 이름)
        "a raised threshold between rooms",
        "a small step on the room floor",      # 바닥의 작은 단차
        
        # --- 부피가 큰 물체들 (70%만 보여도 인식하도록 프롬프트 분할/강화) ---
        "person", "bicycle", "motorcycle", "sink", "stop sign", "bench", "chair", 
        "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard", 
        "traffic cone", "utility pole", "water puddle", "shelf", "stair", "tree",
        
        "a car or a part of a car",            # 차의 일부
        "a bus or a part of a bus",            # 버스의 일부
        "a truck or a part of a truck",        # 트럭의 일부
        "a large refrigerator",                # 냉장고
        "a bed in a bedroom",                  # 침대
        "a dining table",                      # 식탁
        "a cardboard box on the floor",        # 박스
        
        # --- 요청하신 오인식 방지 및 신규 클래스 (화면 표시용) ---
        "wall", 
        "window", 
        "wardrobe",                            # 옷장 하나만
        "clothes hanger rack"                  # 행거
    ]
    
    model.set_classes(custom_classes)
    print("정밀 프롬프트 지정 완료!")

    # [화면 표시를 위한 대표 이름 매핑 테이블]
    door_prompts = [
        "a room door with a handle", "a partially visible entrance door", 
        "a closed wooden door frame", "a glass door with a metal handle"
    ]
    threshold_prompts = [
        "a door sill on the floor", "a floor transition strip", 
        "a raised threshold between rooms", "a small step on the room floor"
    ]
    large_vehicle_prompts = {
        "a car or a part of a car": "car",
        "a bus or a part of a bus": "bus",
        "a truck or a part of a truck": "truck",
        "a large refrigerator": "refrigerator",
        "a bed in a bedroom": "bed",
        "a cardboard box on the floor": "box"
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 해상도 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("실시간 인식을 시작합니다. 종료하려면 창에서 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 잘린 문과 부피 큰 물체들을 70% 선에서 채낚기 위해 conf를 0.24로 미세하게 내렸습니다.
        # 대신 iou를 0.45로 조여 중복 박스를 방지합니다.
        results = model.predict(frame, conf=0.24, iou=0.45, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                
                class_name = custom_classes[cls_id]
                
                # 긴 문장형 프롬프트들을 요청하신 심플한 이름으로 세탁
                if class_name in door_prompts:
                    class_name = "door"
                elif class_name in threshold_prompts:
                    class_name = "threshold"
                elif class_name in large_vehicle_prompts:
                    class_name = large_vehicle_prompts[class_name]
                elif class_name == "clothes hanger rack":
                    class_name = "hanger"

                # 색상 커스텀 (문과 문턱은 파란색 계열, 신규 가구류는 보라색, 나머지는 초록색)
                if class_name in ["door", "threshold"]:
                    color = (255, 100, 0) # 파란색 계열
                elif class_name in ["wall", "window", "wardrobe", "hanger"]:
                    color = (200, 0, 200) # 보라색 계열
                else:
                    color = (0, 255, 0)   # 기본 초록색
                
                # 화면에 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv8s-World (Advanced Partial & Threshold)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_velog_optimized_camera()
