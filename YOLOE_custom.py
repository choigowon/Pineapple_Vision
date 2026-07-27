from ultralytics import YOLOE

model=YOLOE("yoloe-26n-seg.pt")

custom_classes=[
    "a closed room door",
    "an open room door",
    "a glass door",
    "a wardrobe closet",
    "a freestanding garment rack",
    "an electrical power strip",
    "a storage shelf",
    "an open shelving unit",

    "refrigerator",
    "stairs",
    "ramp",

    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",

    "sink",
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

    "window",

    "bollard",
    "traffic cone",
    "utility pole",
    "water puddle",
    "tree"
]

model.set_classes(custom_classes)
model.save("yoloe_custom_26n.pt")

print("YOLOE-26N custom model created")
print("Saved: yoloe_custom_26n.pt")
