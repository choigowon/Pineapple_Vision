from ultralytics import YOLOWorld

# 1. 순정 가중치 로드
model = YOLOWorld('yolov8s-worldv2.pt')

# 2. 사용할 40개 클래스 딱 한 번만 정의
custom_classes = [
    "a room door with a handle", "a partially visible entrance door", 
    "a closed wooden door frame", "a glass door with a metal handle",
    "concrete curb on the street", "stairs leading downwards", "stairs leading upwards", 
    "wheelchair ramp or slope", "electric kick scooter", 
    "person", "bicycle", "motorcycle", "sink", "stop sign", "bench", "chair", 
    "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard", 
    "traffic cone", "utility pole", "water puddle", "shelf", "stair", "tree",
    "a car or a part of a car", "a bus or a part of a bus", "a truck or a part of a truck", 
    "a large refrigerator", "a bed in a bedroom", "a dining table", "a cardboard box on the floor", 
    "wall", "window", "wardrobe", "clothes hanger rack"
]

# 3. 모델 내부에 고정 (텍스트 인코더 연산 수행)
model.set_classes(custom_classes)

# 4. 🌟 아주 중요: 이 상태 그대로 완전히 독립된 pt 파일로 저장합니다.
model.save('fixed_model.pt')
print("🎉 고정형 모델 'fixed_model.pt' 생성 완료!")
