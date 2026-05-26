from ultralytics import YOLOWorld

model = YOLOWorld('yolov8s-worldv2.pt')

custom_classes = [
    "automatic glass sliding door with silver metal frame",
    "framed glass door panel for entrance", 
    "room door with a handle", 
    "door handle or door knob",
    
    # ⚠️ [계단 과탐지 브레이크] 단발성 연석이나 경사로를 거르기 위해 '연속된 여러 개의 단차'를 필수 조건으로 주입
    "pedestrian stairs with multiple continuous vertical steps, not a single road curb or flat ramp", 
    "pedestrian stairs leading upwards with sequential levels", 
    
    # ⚠️ [킥보드 심폐소생] 자전거와 경합에서 이기도록 'T자형 수직 핸들바와 서서 타는 보드' 특징을 직관적으로 명시
    "electric kick scooter with a vertical handlebar and a flat board to stand on", 
    "person", 
    "riding bicycle", 
    "motorcycle with heavy engine", 
    
    "bench", 
    "furniture chair with backrest", 
    "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard", 
    "traffic cone", "utility pole", "water puddle", 
    "storage shelf with racks", "stair", "tree",
    "passenger car on the road",
    "large passenger bus", "cargo truck",
    "kitchen refrigerator appliance", 
    "bed", 
    "flat dining table for eating",
    "cardboard box", "wall", "window fixed in a wall", 
    "wooden wardrobe", "clothes hanger rack",
    
    "safety handrail or metallic grab bar mounted along stairs"
]
model.set_classes(custom_classes)
model.save('fixed_model.pt')
print("🎉 계단 과탐지 억제 및 킥보드 인지 보완이 완료된 'fixed_model.pt' 빌드 완료!")
