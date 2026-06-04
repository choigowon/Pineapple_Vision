import cv2
import time
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

from ultralytics import YOLOWorld
model = YOLOWorld("fixed_model.pt")

# 변환 스크립트의 튜닝된 프롬프트와 1:1 대응 매핑
clean_labels = {
    "a room door with a handle": "door",
    "an entrance door": "door",
    "a glass door with frame": "door",
    
    "concrete curb on the street": "curb",
    "stairs leading downwards": "stairs",
    "stairs leading upwards": "stairs",
    "wheelchair ramp or slope": "ramp",
    "electric kick scooter": "scooter",
    
    "person": "person", "bicycle": "bicycle", "motorcycle": "motorcycle",
    "sink": "sink", "stop sign": "stop_sign", "bench": "bench", "chair": "chair",
    "potted plant": "plant", "tv": "tv", "laptop": "laptop", "cell phone": "phone",
    "microwave": "microwave", "bollard": "bollard", "traffic cone": "cone",
    "utility pole": "pole", "water puddle": "puddle",
    
    "a storage shelf with open racks": "shelf", 
    "stair": "stair", "tree": "tree",
    
    "a passenger car": "car", "a large passenger bus": "bus", "a cargo truck": "truck",
    "a kitchen refrigerator": "refrigerator",
    "a bed": "bed", "a dining table": "table", "a cardboard box": "box",
    "wall": "wall", "window": "window",
    
    "a wooden wardrobe cabinet for clothes storage": "wardrobe", 
    "clothes hanger rack": "hanger"
}

korean_names = {
    "door": "문", "stairs": "계단", "stair": "계단", "ramp": "경사로", "scooter": "킥보드",
    "person": "사람", "bicycle": "자전거", "motorcycle": "오토바이", "bench": "벤치", "chair": "의자",
    "plant": "화분", "tv": "TV", "laptop": "노트북", "phone": "휴대폰", "microwave": "전자레인지",
    "bollard": "볼라드", "cone": "라바콘", "pole": "전신주", "shelf": "선반", "tree": "나무",
    "car": "자동차", "bus": "버스", "truck": "트럭", "refrigerator": "냉장고", "bed": "침대",
    "table": "테이블", "box": "상자", "window": "창문", "wardrobe": "옷장", "hanger": "행거",
    "curb": "연석", "sink": "싱크대", "stop_sign": "정지표지판"
}

# 💡 임계값 최적화 마진 밸런싱
class_conf_thresholds = {
    'stairs': 0.27, 'stair': 0.27, 'ramp': 0.25,
    'door': 0.15,        # 문 인식력을 확실하게 높여 선행 검출 유도
    'refrigerator': 0.35,
    'window': 0.30,      
    'scooter': 0.20, 'bicycle': 0.22, 'motorcycle': 0.25,
    'bench': 0.50, 'table': 0.35, 'box': 0.35,

    'wardrobe': 0.75,    # 옷장 기준선을 현실적인 상한선인 0.75로 재세팅
    'shelf': 0.55,       # 선반의 기준선을 적정 수준으로 유지
    'chair': 0.35,

    'person': 0.30, 'plant': 0.30, 'tv': 0.35, 'laptop': 0.35, 'phone': 0.30, 'microwave': 0.40,
    'bollard': 0.35, 'cone': 0.35, 'pole': 0.35, 'tree': 0.30, 'car': 0.35, 'bus': 0.35, 'truck': 0.35,
    'bed': 0.40, 'hanger': 0.40
}

dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
model.predict(dummy_img, verbose=False)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_spoken_times = {}
AUDIO_COOLDOWN = 1.5

print("🔥 문/선반 ➔ 가구 역오탐 방지 보정 시스템 가동...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.predict(
        frame,
        imgsz=320,
        conf=0.10,  
        iou=0.45,
        verbose=False
    )

    boxes = []
    for result in results:
        if result.boxes:
            boxes.extend(result.boxes)

    # 현재 프레임에 올라온 탐지물 레이블 전수 체크용
    raw_present_classes = []
    for b in boxes:
        r_name = model.names[int(b.cls[0])]
        raw_present_classes.append(clean_labels.get(r_name, r_name))

    boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)

    current_frame_objects = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        raw_name = model.names[cls_id]
        class_name = clean_labels.get(raw_name, raw_name)

        box_w = x2 - x1
        box_h = y2 - y1

        # 💥 [문과 선반을 옷장이라 부르는 현상 하드웨어 제어 카운터]
        if class_name == "door":
            # 종횡비 필터: 세로형태가 선명할 경우 강력 버프
            if box_w >= box_h * 1.0:
                conf -= 0.20
            else:
                conf = min(conf + 0.30, 1.0)

        elif class_name == "shelf":
            # 문이 함께 잡힌 상황에서 선반을 옷장으로 우회 인식하는 것을 방지
            if "wardrobe" in raw_present_classes:
                conf += 0.10

        elif class_name == "wardrobe":
            # 💥 상호 가로채기 파괴: 문(door)이나 선반(shelf) 성분이 프레임에 걸려있는데 
            # 가구 평면 혼선으로 wardrobe 점수가 올라온 경우, 강제 숙청 처리 (-1.0)
            if "door" in raw_present_classes or "shelf" in raw_present_classes:
                conf = -1.0
            else:
                conf -= 0.25

        # 임계값 필터 적용
        if class_name in class_conf_thresholds:
            if conf < class_conf_thresholds[class_name]:
                continue

        if class_name not in korean_names:
            continue

        # 방향 판별
        frame_h, frame_w = frame.shape[:2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if cx < frame_w * 0.33: horizontal = "좌측"
        elif cx > frame_w * 0.66: horizontal = "우측"
        else: horizontal = "정면"

        if cy < frame_h * 0.40: vertical = "상단"
        elif cy > frame_h * 0.60: vertical = "하단"
        else: vertical = ""

        direction = f"{vertical} {horizontal}".strip()
        ko_name = korean_names[class_name]
        
        text = f"{direction} {ko_name}"
        current_frame_objects.append(text)

        # 시각화 드로잉
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # 터미널 실시간 출력
    now = time.time()
    for obj_text in current_frame_objects:
        if obj_text not in last_spoken_times or (now - last_spoken_times.get(obj_text, 0) > AUDIO_COOLDOWN):
            print(f"[탐지]: {obj_text}")
            last_spoken_times[obj_text] = now

    cv2.imshow("YOLOWorld", frame)

    key = cv2.waitKey(1)
    if key == 27:  
        break

cap.release()
cv2.destroyAllWindows()
