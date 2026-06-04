from ultralytics import YOLOWorld

model = YOLOWorld('yolov8s-worldv2.pt')

# 💡 오리지널의 직관적인 명칭 구조를 참고하되, 가구와 문의 경계를 확실히 가른 최적화 프롬프트
custom_classes = [
    # --- 문 영역 ---
    "a room door with a handle", 
    "an entrance door", 
    "a glass door with frame",  # 유리문이 옷장으로 튀는 것을 막기 위한 프레임 강조
    
    # --- 도로 및 계단 영역 ---
    "concrete curb on the street", "stairs leading downwards", "stairs leading upwards", 
    "wheelchair ramp or slope", "stair",
    
    # --- 이동수단 및 사람 ---
    "electric kick scooter", "person", "bicycle", "motorcycle", 
    
    # --- 가구 및 소품 ---
    "sink", "stop sign", "bench", "chair", "potted plant", 
    "tv", "laptop", "cell phone", "microwave", "bed", "a dining table", "a cardboard box",
    "wall", "window", "clothes hanger rack",
    
    # --- 실외 장애물 ---
    "bollard", "traffic cone", "utility pole", "water puddle", "tree",
    "a passenger car", "a large passenger bus", "a cargo truck",
    "a kitchen refrigerator", 
    
    # --- 💥 옷장 / 선반 영역 오탐 방지 종결 프롬프트 ---
    "a storage shelf with open racks", # 앞면이 뚫린 격자 구조 강조 (옷장과 분리)
    "a wooden wardrobe cabinet for clothes storage" # 의류 보관용 거대 가구 속성 강조 (문과 분리)
]

model.set_classes(custom_classes)
model.save('fixed_model.pt')
print("🎉 [프롬프트 밸런스 튜닝 완료] 'fixed_model.pt' 빌드가 완료되었습니다.")
