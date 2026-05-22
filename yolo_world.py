import cv2
from ultralytics import YOLOWorld

def run_velog_optimized_camera():
    print("YOLOv8s-worldv2 모델 로드 중...")
    model = YOLOWorld('yolov8s-worldv2.pt')

    # [프롬프트 마스터 튜닝] 엘리베이터 문 제외, 신규 회전 및 라벨 위치 최적화 적용
    custom_classes = [
        # --- Door 강화 (일부 노출 대응) ---
        "a room door with a handle", 
        "a partially visible entrance door", 
        "a closed wooden door frame", 
        "a glass door with a metal handle",
        
        # --- 문턱(Threshold) 극한 강화 ---
        "a door sill on the floor", 
        "a floor transition strip",            # 바닥 경계 분리대
        "a raised threshold between rooms",
        "a small step on the room floor",      # 바닥의 작은 단차
        
        # --- [매트 오인식 해결용 방어막 클래스] ---
        "floor mat", "area rug", "door mat",   # 매트를 문턱으로 오해하지 않도록 따로 지정
        
        # --- 신규 추가 클래스 (서술형 최적화) ---
        "concrete curb on the street",         # 연석 (curb)
        "stairs leading downwards",            # 내려가는 계단
        "stairs leading upwards",              # 올라가는 계단
        "wheelchair ramp or slope",            # 경사로 (ramp)
        "electric kick scooter",               # 킥보드 (scooter)
        
        # --- 기존 부피가 큰 물체 및 기타 사물들 ---
        "person", "bicycle", "motorcycle", "sink", "stop sign", "bench", "chair", 
        "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard", 
        "traffic cone", "utility pole", "water puddle", "shelf", "stair", "tree",
        
        "a car or a part of a car", 
        "a bus or a part of a bus", 
        "a truck or a part of a truck", 
        "a large refrigerator", 
        "a bed in a bedroom", 
        "a dining table", 
        "a cardboard box on the floor", 
        
        # --- 오인식 방지 및 가구류 ---
        "wall", "window", "wardrobe", "clothes hanger rack"
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
    
    # 문턱 오인식을 막아주되, 화면에는 굳이 띄우지 않을 클래스 리스트
    ignored_classes = ["floor mat", "area rug", "door mat"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 해상도 최적화 (회전 전 원본 해상도 세팅)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("실시간 인식을 시작합니다. 종료하려면 창에서 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [광각 카메라 90도 회전] 
        # 좌우 광각 카메라를 물리적으로 돌렸으므로 영상도 회전시켜 상하 광각으로 만듭니다.
        # 시계방향 90도: cv2.ROTATE_90_CLOCKWISE
        # 반시계방향 90도 회전이 필요하다면 cv2.ROTATE_90_COUNTERCLOCKWISE 로 변경하세요.
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 회전된 프레임을 기반으로 YOLO-World 추론을 수행합니다.
        results = model.predict(frame, conf=0.24, iou=0.45, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                
                class_name = custom_classes[cls_id]
                
                # 매트 등으로 인식된 방어용 박스는 화면 출력을 생략
                if class_name in ignored_classes:
                    continue
                
                # 이름 세탁 로직
                if class_name in door_prompts:
                    class_name = "door"
                elif class_name in threshold_prompts:
                    class_name = "threshold"
                elif class_name == "concrete curb on the street":
                    class_name = "curb"
                elif class_name == "stairs leading downwards":
                    class_name = "stairs down"
                elif class_name == "stairs leading upwards":
                    class_name = "stairs up"
                elif class_name == "wheelchair ramp or slope":
                    class_name = "ramp"
                elif class_name == "electric kick scooter":
                    class_name = "scooter"
                elif class_name in large_vehicle_prompts:
                    class_name = large_vehicle_prompts[class_name]
                elif class_name == "clothes hanger rack":
                    class_name = "hanger"

                # 색상 구분
                if class_name in ["threshold", "curb", "stairs down", "stairs up", "ramp"]:
                    color = (0, 0, 255) # Red
                elif class_name == "door":
                    color = (255, 100, 0) # Blue
                elif class_name in ["wall", "window", "wardrobe", "hanger"]:
                    color = (200, 0, 200) # Purple
                else:
                    color = (0, 255, 0) # Green
                
                # 사물 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # [라벨 위치 최적화 알고리즘 적용]
                # 박스 상단(y1)이 화면 맨 위(15픽셀 이하)에 바짝 붙은 경우 글씨를 박스 안쪽으로 내림
                if y1 < 15:
                    text_y = y1 + 15
                else:
                    text_y = y1 - 10
                    
                label = f"{class_name} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv8s-World (Rotated & Label Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_velog_optimized_camera()
