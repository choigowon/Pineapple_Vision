from ultralytics import YOLOWorld

model = YOLOWorld('yolov8s-worldv2.pt')

# 💡 검증 완료된 오리지널 최적화 프롬프트 리스트 그대로 고정
custom_classes = [
    "a room door with a handle", "an entrance door", "a glass door",
    "concrete curb on the street", "stairs leading downwards", "stairs leading upwards", 
    "wheelchair ramp or slope", "electric kick scooter", 
    "person", "bicycle", "motorcycle", "sink", "stop sign", "bench", "chair", 
    "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard", 
    "traffic cone", "utility pole", "water puddle", 
    "a storage shelf with racks", 
    "stair", "tree",
    "a passenger car", "a large passenger bus", "a cargo truck",
    "a kitchen refrigerator", 
    "a bed", "a dining table", "a cardboard box", 
    "wall", "window", "a wooden wardrobe", "clothes hanger rack"
]

model.set_classes(custom_classes)
model.save('fixed_model.pt')
print("🎉 오리지널 검증 프롬프트 'fixed_model.pt' 복구 완료!")
