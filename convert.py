from ultralytics import YOLOWorld

model = YOLOWorld('yolov8s-worldv2.pt')

custom_classes = [
    "automatic glass sliding door with silver metal frame",
    "framed glass door panel for entrance",
    "room door with a handle",
    "door handle or door knob",

    "pedestrian stairs with multiple continuous vertical steps, not a single road curb or flat ramp",
    "pedestrian stairs leading upwards with sequential levels",

    "electric kick scooter with a vertical handlebar and a flat board to stand on",
    "person",
    "riding bicycle",
    "motorcycle with heavy engine",

    "bench",
    "furniture chair with backrest",
    "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard",
    "traffic cone", "utility pole", "water puddle",

    # ⚠️ [테이블 vs 선반] 칸칸이 나누어진 수직 수납장/랙 구조임을 명시하여 평평한 테이블과 분리
    "vertical storage shelf with multiple grid racks for holding items, not a flat table",
    "stair", "tree",
    "passenger car on the road",
    "large passenger bus", "cargo truck",
    "kitchen refrigerator appliance",
    "bed",

    # ⚠️ [테이블 정의 강화] 다리 몇 개 위에 평평한 상판 딱 하나만 있는 식사 및 작업용 가구 강조
    "flat dining table or desk supported by legs with a single flat surface for working, NO multiple shelves",
    "cardboard box", "wall", "window fixed in a wall",
    "wooden wardrobe", "clothes hanger rack",

    "safety handrail or metallic grab bar mounted along stairs"
]
model.set_classes(custom_classes)
model.save('fixed_model.pt')
print("🎉 테이블과 선반의 형태적 독립 패치가 완료된 'fixed_model.pt' 빌드 완료!")
