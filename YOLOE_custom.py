from ultralytics import YOLOE

model=YOLOE("yoloe-26n-seg.pt")

custom_classes=[
    "a standard room door",
    "a glass door",
    "a wardrobe cabinet",
    "an open storage shelf",
    "a refrigerator",
    "stairs",
    "ramp",

    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",

    "sink",
    "stop sign",
    "bench",
    "chair",
    "potted plant",
    "tv",
    "laptop",
    "cell phone",
    "microwave",
    "bed",
    "table",
    "cardboard box",

    "wall",
    "window",
    "clothes rack",

    "bollard",
    "traffic cone",
    "utility pole",
    "water puddle",
    "tree"
]

model.set_classes(custom_classes)
model.save("yoloe_custom_26n.pt")

print("✅ YOLOE-26N 커스텀 모델 생성 완료")
print("📁 저장 파일: yoloe_custom_26n.pt")
