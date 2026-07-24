from ultralytics import YOLOE

# ============================================================
# 1. YOLOE 모델 불러오기
# ============================================================
model = YOLOE("yoloe-11s-seg.pt")

# ============================================================
# 2. 인식하고 싶은 객체 목록
# ============================================================
custom_classes = [
    "a standard room door",
    "a glass door",
    "a wardrobe cabinet",
    "an open storage shelf",
    "a refrigerator",

    "person",
    "chair",
    "table",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "stairs",
    "ramp",
    "bollard",
    "traffic cone"
]

# ============================================================
# 3. YOLOE에 텍스트 프롬프트 설정
# ============================================================
model.set_classes(custom_classes)

# ============================================================
# 4. 커스텀 클래스가 적용된 모델 저장
# ============================================================
model.save("yoloe_custom.pt")

print("======================================")
print("YOLOE 커스텀 클래스 설정 완료")
print("저장 파일 : yoloe_custom.pt")
print("클래스 개수 :", len(custom_classes))
print("======================================")
