import os
import time
import threading
import queue
from collections import deque

try:
    import pytesseract
except ImportError:
    pytesseract = None

import cv2
import numpy as np
from ultralytics import YOLOE

try:
    import serial
except ImportError:
    serial = None


# ============================================================
# YOLOE 기반 상황형 보행 보조 Raspberry Pi 터미널 프로토타입
#
# 횡단보도 안전 상태 안내 개선 사항
# - 차량의 방향을 반복 안내하지 않고 횡단 가능 상태를 판단
# - 차량이 새로 감지되면 즉시 "멈추세요"
# - 횡단 상태는 "건너세요" 또는 "멈추세요"로 간결하게 안내
# - 차량 중심이 2.5초 이상 안정적으로 유지되면 횡단 가능으로 판단
# - 횡단보도 상태는 1프레임만 감지되어도 즉시 진입
# ============================================================

MODEL_PATH = os.getenv("WALKING_ASSISTANT_MODEL", "yoloe-26n-seg.pt")
CAMERA_INDEX = int(os.getenv("WALKING_ASSISTANT_CAMERA_INDEX", "0"))
IMU_PORT = os.getenv("WALKING_ASSISTANT_IMU_PORT") or None
IMU_BAUDRATE = 115200

# 1차 추론은 낮은 threshold로 후보를 확보한 뒤, 아래 후처리에서
# 클래스별 기준과 기하학 기준을 적용한다.
CONFIDENCE = 0.20
YOLO_INFERENCE_CONFIDENCE = 0.08  # 멀리 있는 작은 차량 후보를 확보한 뒤 클래스별 필터링
NMS_IOU = 0.40
# 속도와 작은 객체 인식의 균형을 위해 512로 시작한다. 환경변수로 조정할 수 있다.
INFERENCE_SIZE = int(os.getenv("WALKING_ASSISTANT_INFERENCE_SIZE", "512"))
PROFILE_TIMING = os.getenv("WALKING_ASSISTANT_PROFILE", "0").strip().lower() not in {"0", "false", "no"}
PROFILE_INTERVAL_SECONDS = 5.0
# CUDA 비동기 추론의 실제 시간을 재려면 1로 설정(측정 중 동기화 오버헤드가 생길 수 있음).
PROFILE_CUDA_SYNC = os.getenv("WALKING_ASSISTANT_PROFILE_CUDA_SYNC", "0").strip().lower() in {"1", "true", "yes"}
STATE_CONFIRM_FRAMES = 4
AUTO_CLAHE = True
SPEECH_COOLDOWN = 0.22
URGENT_SPEECH_COOLDOWN = 0.12

# 횡단보도 안전 판단 설정
CROSSWALK_VEHICLE_MIN_CONF = 0.16
CROSSWALK_CAR_CONFIRM_FRAMES = 1
CROSSWALK_CLEAR_SECONDS = 2.0
CROSSWALK_MIN_CONF = 0.36
CROSSWALK_ENTER_CONFIRM_SECONDS = 0.6
CROSSWALK_EXIT_HOLD_SECONDS = 2.0
CROSSWALK_DETECTION_GAP_SECONDS = 0.8
# 차량 중심 추적: 검출 즉시 정지, 2~3초 안정 확인 후에만 횡단 허용
VEHICLE_FRAME_HOLD_SECONDS = 0.70
VEHICLE_ABSENCE_CLEAR_SECONDS = 2.0
VEHICLE_STATIONARY_SECONDS = 2.5  # 중심이 허용 오차 내에서 2~3초 안정되면 횡단 허용
VEHICLE_CENTER_TOLERANCE = 0.035  # 화면 대각선 정규화 기준 중심 이동 허용치
VEHICLE_ASSOCIATION_GAP = 0.22
CROSSWALK_ZONE_COUNT = 4
CROSSWALK_REPEAT_SECONDS = 5.0
DOOR_OCR_INTERVAL = 1.0
DOOR_OCR_REPEAT_SECONDS = 5.0

# 차량이 보행자 쪽으로 접근하는지 판정하는 기준
VEHICLE_APPROACH_MIN_FRAMES = 3
VEHICLE_APPROACH_EVIDENCE = 2
VEHICLE_AREA_GROWTH_RATIO = 0.025
VEHICLE_BOTTOM_APPROACH_RATIO = 0.006

# 정지 차량 판별 설정
VEHICLE_MOTION_MIN_FRAMES = 3
VEHICLE_STOPPED_FRAMES = 6
VEHICLE_MOTION_PIXEL_RATIO = 0.015
VEHICLE_AREA_CHANGE_RATIO = 0.12

# 횡단보도 차량 추적: 짧은 미검출은 같은 트랙의 일시적 누락으로 처리
VEHICLE_TRACK_MAX_MISSED = 3
VEHICLE_TRACK_MAX_CENTER_GAP = 0.16
VEHICLE_MIN_MOTION_RATIO = 0.0025
VEHICLE_FUTURE_STEPS = 12
# 화면상 횡단보도/보행 위험 구역(정규화 좌표): 중앙 하단 영역
VEHICLE_RISK_X1 = 0.28
VEHICLE_RISK_X2 = 0.72
VEHICLE_RISK_Y1 = 0.58
VEHICLE_RISK_Y2 = 1.00

# 계단·엘리베이터 확정 기준
STAIR_CONFIRM_FRAMES = 2
STAIR_MIN_AREA_RATIO = 0.04
STAIR_MIN_WIDTH_RATIO = 0.18
# 엘리베이터는 오인식 방지를 위해 매우 보수적으로 확정한다.
ELEVATOR_MIN_CONF = 0.96
ELEVATOR_CONFIRM_FRAMES = 5
ELEVATOR_MIN_AREA_RATIO = 0.08
ELEVATOR_MIN_WIDTH_RATIO = 0.25
ELEVATOR_STICKER_MIN_CONF = 0.72
ELEVATOR_STICKER_CONFIRM_FRAMES = 3

# Define prompt keys before CLASS_MIN_CONF references them.
CAR_PROMPT = "passenger car driving on a road, viewed from a distance"
SCOOTER_PROMPT = "electric kick scooter on a road"
MOTORCYCLE_PROMPT = "motorcycle on a road"
PLAIN_SCOOTER_PROMPT = "scooter on a road"
CAR_DISTANT_PROMPT = "small distant car driving on a road"
CAR_FAR_PROMPT = "far away passenger vehicle on a road"
MOTORCYCLE_RIDER_PROMPT = "motorcycle with a rider on a road"
MOTORCYCLE_DISTANT_PROMPT = "small distant motorcycle with rider"
MOTORCYCLIST_PROMPT = "motorcyclist riding a motorcycle on a road"
SCOOTER_RIDER_PROMPT = "person riding an electric scooter on a road"

GLASS_DOOR_LABELS = {
    "glass door", "transparent glass door", "clear glass door",
    "glass entrance door", "large glass entrance door",
    "commercial glass entrance door", "glass storefront door",
    "glass panel door", "framed glass door", "glass door with metal frame",
    "transparent sliding glass door", "sliding glass entrance door",
    "front view of a glass door", "glass door at a building entrance",
    "automatic glass door",
}

CLASS_MIN_CONF = {
    "chair": 0.65,
    "crosswalk": CROSSWALK_MIN_CONF,
    "zebra crossing": CROSSWALK_MIN_CONF,
    "painted zebra crossing stripes": CROSSWALK_MIN_CONF,
    "pedestrian crossing stripes on pavement": CROSSWALK_MIN_CONF,
    "marked pedestrian crosswalk": CROSSWALK_MIN_CONF,
    CAR_PROMPT: 0.17,
    CAR_DISTANT_PROMPT: 0.15,
    CAR_FAR_PROMPT: 0.15,
    SCOOTER_PROMPT: 0.17,
    MOTORCYCLE_PROMPT: 0.17,
    MOTORCYCLE_RIDER_PROMPT: 0.15,
    MOTORCYCLE_DISTANT_PROMPT: 0.15,
    MOTORCYCLIST_PROMPT: 0.15,
    PLAIN_SCOOTER_PROMPT: 0.17,
    SCOOTER_RIDER_PROMPT: 0.15,
    "motorcycle": 0.17,
    "motorbike": 0.17,
    "car": 0.17,
    "scooter": 0.17,
    "kick scooter": 0.17,
    "electric scooter": 0.17,
    "electric kick scooter": 0.17,
    "person": 0.40,
    "stairs": 0.45,
    "staircase": 0.45,
    "stair landing": 0.50,
    "door": 0.48,
    "sliding door": 0.42,
    "glass sliding door": 0.32,
    "sliding glass door": 0.30,
    "front view of a glass sliding door": 0.28,
    "front facing glass sliding door": 0.28,
    "sliding glass door viewed straight on": 0.28,
    "transparent sliding glass entrance door": 0.28,
    "sliding glass door handle": 0.44,
    "automatic door": 0.60,
    "glass door": 0.60,
    "hinged door": 0.50,
    "elevator": ELEVATOR_MIN_CONF,
    "elevator safety warning sticker": ELEVATOR_STICKER_MIN_CONF,
    "do not lean on elevator door sticker": ELEVATOR_STICKER_MIN_CONF,
    "fall hazard sticker on elevator door": ELEVATOR_STICKER_MIN_CONF,
    "elevator safety notice": ELEVATOR_STICKER_MIN_CONF,
    "doorknob": 0.62,
    "door knob": 0.62,
    "door handle": 0.50,
    "lever handle": 0.50,
    "lever door handle": 0.50,
    "pull handle": 0.50,
    "pull door handle": 0.50,
    "sliding glass door handle": 0.52,
    "push bar": 0.68,
    "door push button": 0.70,
    "recessed door handle": 0.56,
    **{label: 0.18 for label in GLASS_DOOR_LABELS},
}

AUDIO_DIR = "audio"

PROMPTS = [
    "box", "desk", "table", "chair", "bed", "shelf", "wardrobe",
    "refrigerator", "tree", CAR_PROMPT, "car", SCOOTER_PROMPT, "scooter",
    MOTORCYCLE_PROMPT, "motorcycle", "bollard", "traffic cone",
    "utility pole", "bicycle", "person", "curb", "railing", "guardrail",
    "door", "sliding door", "glass sliding door", "sliding glass door",
    "automatic door", "glass door", "hinged door", "door handle", "doorknob",
    "lever handle", "pull handle", "push bar", "stairs", "staircase",
    "elevator", "elevator safety warning sticker", "elevator safety notice",
    "crosswalk", "zebra crossing",
]

# 횡단보도 모드에서는 차량·킥보드·오토바이와 횡단보도만 추론한다.
# 모드가 바뀔 때만 set_classes를 호출해 매 프레임 프롬프트 인코딩을 반복하지 않는다.
TRAFFIC_PROMPTS = [
    CAR_PROMPT, "car", SCOOTER_PROMPT, "scooter",
    MOTORCYCLE_PROMPT, "motorcycle", "crosswalk", "zebra crossing",
]

DOOR_LABELS = {
    "door", "sliding door", "glass sliding door", "sliding glass door",
    "front view of a glass sliding door", "front facing glass sliding door",
    "sliding glass door viewed straight on", "transparent sliding glass entrance door",
    "automatic door", "glass door", "hinged door",
    "transparent glass door", "clear glass door",
    "glass entrance door", "large glass entrance door",
    "commercial glass entrance door", "glass storefront door",
    "glass panel door", "framed glass door", "glass door with metal frame",
    "transparent sliding glass door", "sliding glass entrance door",
    "front view of a glass door", "glass door at a building entrance",
    "automatic glass door",
}

DOOR_COMPONENT_LABELS = {
    "doorknob", "door knob", "door handle", "lever handle",
    "lever door handle", "pull handle", "pull door handle",
    "sliding glass door handle", "push bar",
    "door push button", "recessed door handle",
}

STAIR_LABELS = {
    "stairs", "staircase", "stair landing",
}
# 버튼 프롬프트는 제거한다. 현재 모델에서는 일반 표지판·벽면·문손잡이를
# 버튼으로 오인할 가능성이 커서, 엘리베이터가 확정된 뒤에도 버튼 안내를 하지 않는다.
ELEVATOR_LABELS = {"elevator"}
ELEVATOR_COMPONENT_LABELS = set()
ELEVATOR_SUPPORT_LABELS = {
    "elevator safety warning sticker",
    "do not lean on elevator door sticker",
    "fall hazard sticker on elevator door",
    "elevator safety notice",
}
CROSSWALK_LABELS = {
    "crosswalk", "zebra crossing",
    "painted zebra crossing stripes",
    "pedestrian crossing stripes on pavement",
    "marked pedestrian crosswalk",
}
VEHICLE_LABELS = {
    CAR_PROMPT, CAR_DISTANT_PROMPT, CAR_FAR_PROMPT,
    SCOOTER_PROMPT, SCOOTER_RIDER_PROMPT,
    MOTORCYCLE_PROMPT, MOTORCYCLE_RIDER_PROMPT,
    MOTORCYCLE_DISTANT_PROMPT, MOTORCYCLIST_PROMPT, PLAIN_SCOOTER_PROMPT,
    "car", "motorcycle", "motorbike", "scooter", "kick scooter",
    "electric scooter", "electric kick scooter",
}

HANDLE_MIN_CONF = 0.50
DOOR_MIN_CONF = 0.18

AUDIO_FILES = {
    "vehicle_left": "vehicle_left.wav",
    "vehicle_right": "vehicle_right.wav",
    "vehicle_front": "vehicle_front.wav",
}

KOREAN_LABELS = {
    "box": "상자",
    "desk": "책상",
    "table": "테이블",
    "chair": "의자",
    "bed": "침대",
    "shelf": "선반",
    "wardrobe": "옷장",
    "closet": "옷장",
    "refrigerator": "냉장고",
    "tree": "나무",
    "car": "차량",
    CAR_PROMPT: "차량",
    CAR_DISTANT_PROMPT: "차량",
    CAR_FAR_PROMPT: "차량",
    SCOOTER_PROMPT: "킥보드",
    SCOOTER_RIDER_PROMPT: "킥보드",
    MOTORCYCLE_PROMPT: "오토바이",
    MOTORCYCLE_RIDER_PROMPT: "오토바이",
    MOTORCYCLE_DISTANT_PROMPT: "오토바이",
    MOTORCYCLIST_PROMPT: "오토바이",
    PLAIN_SCOOTER_PROMPT: "스쿠터",
    "motorcycle": "오토바이",
    "motorbike": "오토바이",
    "curb": "연석",
    "railing": "난간",
    "guardrail": "가드레일",
    "scooter": "킥보드",
    "kick scooter": "킥보드",
    "electric scooter": "전동 킥보드",
    "electric kick scooter": "전동 킥보드",
    "bollard": "볼라드",
    "traffic cone": "콘",
    "utility pole": "전봇대",
    "bicycle": "자전거",
    "person": "사람",
    "doorknob": "문고리",
    "door knob": "문고리",
    "door handle": "문손잡이",
    "lever handle": "레버 손잡이",
    "lever door handle": "레버 손잡이",
    "pull handle": "당김 손잡이",
    "pull door handle": "당김 손잡이",
    "sliding glass door handle": "유리문 손잡이",
    "push bar": "푸시바",
    "door push button": "문 버튼",
    "recessed door handle": "매립 손잡이",
    "front view of a glass sliding door": "유리 미닫이문",
    "front facing glass sliding door": "유리 미닫이문",
    "sliding glass door viewed straight on": "유리 미닫이문",
    "transparent sliding glass entrance door": "유리 미닫이문",
    **{label: "유리문" for label in GLASS_DOOR_LABELS},
}


class OptionalIMU:
    def __init__(self, port=None, baudrate=115200):
        self.ser = None
        self.moving = False
        self.turning = False

        if port and serial:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.02)
                threading.Thread(target=self._reader, daemon=True).start()
                print(f"IMU 연결: {port}")
            except Exception as exc:
                print(f"IMU 연결 실패: {exc}")
        else:
            print("IMU 없이 실행합니다. 카메라 기반 상태 추정만 사용합니다.")

    def parse_line(self, line):
        try:
            values = [float(x.strip()) for x in line.split(",")]
            if len(values) >= 6:
                ax, ay, az, gx, gy, gz = values[:6]
                self.moving = abs(ax) + abs(ay) + abs(az - 9.8) > 2.0
                self.turning = abs(gz) > 35.0
        except ValueError:
            pass

    def _reader(self):
        while self.ser and self.ser.is_open:
            raw = self.ser.readline().decode(errors="ignore").strip()
            if raw:
                self.parse_line(raw)


class Speech:
    """음성 엔진 대신 안내 결과를 터미널에만 출력한다."""

    def __init__(self):
        self.last_key = None
        self.last_submit_time = 0.0
        self.vehicle_priority = False
        print("출력 모드: 터미널 텍스트 전용 (음성 출력 비활성화)", flush=True)

    def set_vehicle_priority(self, active):
        self.vehicle_priority = bool(active)

    def say(self, key, text, force=False):
        if self.vehicle_priority and not key.startswith(("crosswalk_", "traffic_")):
            return
        now = time.time()
        if key == self.last_key and now - self.last_submit_time < 2.0:
            return
        self.last_key = key
        self.last_submit_time = now
        print(f"[결과] {text}", flush=True)

    def say_urgent(self, key, text):
        self.last_key = key
        self.last_submit_time = time.time()
        print(f"[긴급 결과] {text}", flush=True)

    def cancel(self):
        pass

    def say_crosswalk_status(self, status, force=True):
        if status in {"WAIT", "UNKNOWN"}:
            self.say("crosswalk_wait", "멈추세요", force=force)
        elif status == "CLEAR":
            self.say("crosswalk_clear", "건너세요", force=force)


def label_of(result, cls_id):
    names = result.names
    return names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, "unknown")


def zone(x1, x2, width):
    center = (x1 + x2) / 2
    if center < width * 0.38:
        return "왼쪽"
    if center > width * 0.62:
        return "오른쪽"
    return "정면"


def roi(width, height):
    return int(width * 0.37), int(height * 0.12), int(width * 0.63), height


def walking_candidate(det, width, height):
    _, _, x1, y1, x2, y2 = det
    rx1, ry1, rx2, ry2 = roi(width, height)
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)


def is_walk_obstacle(det):
    """보행 ROI 안에서 즉시 안내해야 하는 장애물인지 판단한다."""
    label = det[0]
    return label not in VEHICLE_LABELS and label not in CROSSWALK_LABELS


def box_iou(a, b):
    _, _, ax1, ay1, ax2, ay2 = a
    _, _, bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / (area_a + area_b - inter)


def same_target(a, b):
    return a is not None and b is not None and a[0] == b[0] and box_iou(a, b) >= 0.12


def proximity_score(det, width, height):
    """실제 거리가 아닌 영상상 우선순위 점수."""
    _, _, x1, y1, x2, y2 = det
    bottom = y2 / max(1, height)
    area = ((x2 - x1) * (y2 - y1)) / max(1, width * height)
    center = abs(((x1 + x2) / 2) - width / 2) / max(1, width)
    return 0.55 * bottom + 0.35 * min(area * 4.0, 1.0) + 0.10 * (1.0 - center)


def detection_priority(det, width, height):
    return (-proximity_score(det, width, height), -det[1])


def expanded_box(x1, y1, x2, y2, width, height, vertical_ratio=0.18):
    box_h = y2 - y1
    pad = int(box_h * vertical_ratio)
    return (
        max(0, x1),
        max(0, y1 - pad),
        min(width - 1, x2),
        min(height - 1, y2 + pad),
    )





def deduplicate_vehicle_detections(detections):
    """서로 다른 유사 프롬프트가 같은 차량에 만든 중복 박스를 하나로 합친다."""
    kept = []
    for det in sorted(detections, key=lambda d: d[1], reverse=True):
        _, _, x1, y1, x2, y2 = det
        area = max(1, (x2-x1)*(y2-y1))
        duplicate = False
        for old in kept:
            _, _, ox1, oy1, ox2, oy2 = old
            iw = max(0, min(x2, ox2)-max(x1, ox1))
            ih = max(0, min(y2, oy2)-max(y1, oy1))
            inter = iw*ih
            old_area = max(1, (ox2-ox1)*(oy2-oy1))
            iou = inter / max(1, area+old_area-inter)
            containment = inter / max(1, min(area, old_area))
            if iou >= 0.30 or containment >= 0.70:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept

def deduplicate_detections(detections):
    """동일 클래스의 겹치거나 안쪽에 중복된 박스를 합친다.

    YOLO NMS는 박스가 한 박스 안에 작게 들어간 경우 IoU가 낮아 둘 다
    남길 수 있다. 완전히 포함된 작은 동일 클래스 박스는 중복으로 제거한다.
    """
    kept = []
    ordered = sorted(detections, key=lambda d: d[1], reverse=True)
    for det in ordered:
        label, _, x1, y1, x2, y2 = det
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for existing in kept:
            if existing[0] != label:
                continue
            ex1, ey1, ex2, ey2 = existing[2:6]
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            existing_area = max(1, (ex2 - ex1) * (ey2 - ey1))
            if inter / max(1, min(area, existing_area)) >= 0.82:
                duplicate = True
                break
            union_iou = inter / max(1, area + existing_area - inter)
            if union_iou >= 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def class_min_conf(label):
    return CLASS_MIN_CONF.get(label, CONFIDENCE)


def valid_geometry(det, width, height):
    """클래스별 최소 크기·종횡비 필터. 작은 점/벽면 반사를 제거한다."""
    label, _, x1, y1, x2, y2 = det
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / max(1, width * height)
    aspect = bw / bh

    if label in VEHICLE_LABELS:
        # 원거리 차량/오토바이는 박스가 작으므로 일반 사물보다 작은 크기까지 허용
        return (bw >= 5 and bh >= 5 and area_ratio >= 0.00005
                and area_ratio <= 0.45 and 0.18 <= aspect <= 8.0)
    if label in CROSSWALK_LABELS:
        # 도로 면 전체가 아니라 실제 횡단보도 표시로 보이는 유효 박스만 허용
        return area_ratio >= 0.00020 and bw >= width * 0.025 and bh >= height * 0.015
    if label in DOOR_LABELS:
        # 정면 유리문은 투명해 박스가 작고 얇게 잡힐 수 있다.
        if label in GLASS_DOOR_LABELS or label in {
            "glass sliding door", "sliding glass door",
            "front view of a glass sliding door", "front facing glass sliding door",
            "sliding glass door viewed straight on", "transparent sliding glass entrance door",
        }:
            return area_ratio >= 0.00025 and bh >= height * 0.025 and aspect <= 10.0
        return area_ratio >= 0.0010 and bh >= height * 0.08 and aspect <= 5.0
    if label == "elevator":
        return area_ratio >= ELEVATOR_MIN_AREA_RATIO and bw >= width * ELEVATOR_MIN_WIDTH_RATIO and bh >= height * 0.20
    if label in ELEVATOR_SUPPORT_LABELS:
        return area_ratio >= 0.00015 and area_ratio <= 0.08
    if label in DOOR_COMPONENT_LABELS:
        return area_ratio <= 0.035 and bw >= 6 and bh >= 6
    if label in STAIR_LABELS:
        return area_ratio >= STAIR_MIN_AREA_RATIO or bw >= width * STAIR_MIN_WIDTH_RATIO
    return area_ratio >= 0.0002


def prepare_inference_frame(frame):
    """어두운/저대비 화면에서만 자동 CLAHE를 적용한다."""
    if not AUTO_CLAHE:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) >= 75.0 and float(np.std(gray)) >= 25.0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def guidance_for_detection(det, width):
    label, _, x1, _, x2, _ = det
    pos = zone(x1, x2, width)

    if label in {"doorknob", "door knob"}:
        return f"{pos} 문고리. 돌려서 여세요."
    if label in {"door handle", "lever handle", "lever door handle"}:
        return f"{pos} 레버 손잡이. 아래로 눌러 여세요."
    if label == "sliding glass door handle":
        return f"{pos} 유리 미닫이문 손잡이. 잡고 옆으로 여세요."
    if label in {"pull handle", "pull door handle"}:
        return f"{pos} 당김 손잡이. 당겨서 여세요."
    if label == "recessed door handle":
        return f"{pos} 매립 손잡이. 잡고 밀어 여세요."
    if label == "door push button":
        return f"{pos} 문 버튼. 눌러서 여세요."
    if label == "push bar":
        return f"{pos} 푸시바. 앞으로 미세요."

    sliding_glass_labels = {
        "glass sliding door", "sliding glass door",
        "front view of a glass sliding door", "front facing glass sliding door",
        "sliding glass door viewed straight on", "transparent sliding glass entrance door",
        "transparent sliding glass door", "sliding glass entrance door",
    }
    if label in GLASS_DOOR_LABELS and label not in sliding_glass_labels:
        return f"{pos} 유리문. 손잡이를 확인하세요."

    door_text = {
        "sliding door": "미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "glass sliding door": "수동 유리 미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "sliding glass door": "수동 유리 미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "front view of a glass sliding door": "수동 유리 미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "front facing glass sliding door": "수동 유리 미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "sliding glass door viewed straight on": "수동 유리 미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "transparent sliding glass entrance door": "수동 유리 미닫이문. 손잡이를 잡고 옆으로 여세요.",
        "automatic door": "자동문. 잠시 기다리세요.",
        "glass door": "유리문. 손잡이를 확인하세요.",
        "hinged door": "여닫이문. 당기거나 미세요.",
        "door": "문. 손잡이를 당기거나 미세요.",
    }

    if label in door_text:
        return f"{pos} {door_text[label]}"
    if label in STAIR_LABELS:
        return f"{pos} 계단. 방향을 확인하세요."
    if label == "elevator":
        return f"{pos} 엘리베이터로 보입니다. 문과 버튼을 확인하세요."
    if label in CROSSWALK_LABELS:
        return "정면 횡단보도. 멈추고 확인하세요."

    return f"{pos} {KOREAN_LABELS.get(label, label)}"


class SituationMachine:
    def __init__(self):
        self.state = "WALK"
        self.forced = False
        self.candidate = None
        self.count = 0
        self.last_crosswalk_time = None
        self.crosswalk_candidate_since = None

    def set_forced(self, state):
        self.state = state
        self.forced = True
        if state == "CROSSWALK":
            self.last_crosswalk_time = time.monotonic()

    def auto(self):
        self.forced = False
        self.last_crosswalk_time = None
        self.crosswalk_candidate_since = None

    def update(self, detections):
        labels = {d[0] for d in detections}
        now = time.monotonic()
        crosswalk_detected = bool(labels & CROSSWALK_LABELS)

        if crosswalk_detected:
            self.last_crosswalk_time = now
            if self.crosswalk_candidate_since is None:
                self.crosswalk_candidate_since = now

            # 횡단보도 검출이 일정 시간 확인된 뒤 모드에 진입한다.
            # 이미 진입한 경우에는 잠깐의 검출 누락에도 교통 모드를 유지한다.
            if (self.state == "CROSSWALK"
                    or now - self.crosswalk_candidate_since >= CROSSWALK_ENTER_CONFIRM_SECONDS):
                self.state = "CROSSWALK"
                self.candidate = "CROSSWALK"
                self.count = 1
                self.forced = False
                return self.state
        elif self.crosswalk_candidate_since is not None:
            # 한두 프레임의 검출 흔들림은 같은 확인 구간으로 허용한다.
            if (self.last_crosswalk_time is None
                    or now - self.last_crosswalk_time > CROSSWALK_DETECTION_GAP_SECONDS):
                self.crosswalk_candidate_since = None

        if self.state == "CROSSWALK":
            if (self.last_crosswalk_time is not None
                    and now - self.last_crosswalk_time <= CROSSWALK_EXIT_HOLD_SECONDS):
                return self.state
            # 횡단보도가 일정 시간 보이지 않으면 일반 인식 모드로 복귀한다.
            self.state = "WALK"
            self.forced = False
            self.candidate = "WALK"
            self.count = 1
            self.last_crosswalk_time = None
            self.crosswalk_candidate_since = None

        if self.forced:
            return self.state

        if labels & ELEVATOR_LABELS:
            candidate = "ELEVATOR"
        elif labels & STAIR_LABELS:
            candidate = "STAIRS"
        elif labels & DOOR_LABELS:
            candidate = "DOOR"
        else:
            candidate = "WALK"

        if candidate == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = candidate, 1

        if candidate == "WALK":
            self.state = "WALK"
        elif self.count >= STATE_CONFIRM_FRAMES:
            self.state = candidate
        return self.state


class VehicleMotionTracker:
    """차량/킥보드의 이동 방향과 횡단보도 위험 구역 진입 가능성을 추적한다.

    최근 3프레임까지의 일시적 검출 누락은 트랙을 유지한다. 차량이
    화면 어디서든 움직인다는 이유만으로 경고하지 않고, 현재 위치 또는
    이동 방향의 연장선이 중앙 하단 가상 위험 영역을 통과할 때만 관심 대상으로 둔다.
    """

    def __init__(self):
        self.tracks = []
        self.frame_index = 0
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.frame_index = 0
        self.next_id = 1

    @staticmethod
    def _center(det):
        return ((det[2] + det[4]) / 2.0, (det[3] + det[5]) / 2.0)

    @staticmethod
    def _area(det):
        return max(1.0, (det[4] - det[2]) * (det[5] - det[3]))

    @staticmethod
    def _norm_point(point, width, height):
        return point[0] / max(1, width), point[1] / max(1, height)

    @staticmethod
    def _inside_risk(point):
        x, y = point
        return VEHICLE_RISK_X1 <= x <= VEHICLE_RISK_X2 and VEHICLE_RISK_Y1 <= y <= VEHICLE_RISK_Y2

    @staticmethod
    def _segment_hits_risk(start, end):
        # 선분을 샘플링해 가상 위험 사각형을 통과하는지 확인한다.
        for step in range(13):
            ratio = step / 12.0
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            if VehicleMotionTracker._inside_risk(point):
                return True
        return False

    def _match(self, track, detections, used, width, height):
        history = track["history"]
        last = history[-1]
        last_det = last["det"]
        old_cx, old_cy = last["center"]
        gap = self.frame_index - last["frame"]

        # 짧은 누락에서도 최근 이동 벡터로 다음 위치를 예측
        predicted = (old_cx, old_cy)
        if len(history) >= 2:
            prev = history[-2]
            dt = max(1, last["frame"] - prev["frame"])
            vx = (old_cx - prev["center"][0]) / dt
            vy = (old_cy - prev["center"][1]) / dt
            predicted = (old_cx + vx * gap, old_cy + vy * gap)

        best_i, best_score = None, float("inf")
        diag = max(1.0, (width * width + height * height) ** 0.5)
        for i, det in enumerate(detections):
            if i in used or det[0] != track["label"]:
                continue
            cx, cy = self._center(det)
            dist = ((cx - predicted[0]) ** 2 + (cy - predicted[1]) ** 2) ** 0.5 / diag
            overlap = box_iou(last_det, det)
            if dist <= VEHICLE_TRACK_MAX_CENTER_GAP or overlap >= 0.01:
                score = dist - min(overlap, 0.5) * 0.25
                if score < best_score:
                    best_i, best_score = i, score
        return best_i

    def _track_hits_risk(self, track, width, height):
        history = track["history"]
        current = history[-1]
        current_norm = self._norm_point(current["center"], width, height)
        if self._inside_risk(current_norm):
            return True
        if len(history) < 2:
            return False

        prev = history[-2]
        dt = max(1, current["frame"] - prev["frame"])
        p0 = self._norm_point(prev["center"], width, height)
        p1 = current_norm
        vx, vy = (p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < VEHICLE_MIN_MOTION_RATIO:
            return False

        future = (p1[0] + vx * VEHICLE_FUTURE_STEPS, p1[1] + vy * VEHICLE_FUTURE_STEPS)
        # 관심 대상은 위험 영역을 향하거나 그 영역을 통과하는 트랙만
        return self._segment_hits_risk(p1, future) or self._segment_hits_risk(p0, p1)

    def _is_approaching(self, track, width, height):
        history = track["history"]
        if len(history) < 2:
            return False
        last = history[-1]
        prev = history[-2]
        dt = max(1, last["frame"] - prev["frame"])
        p0 = self._norm_point(prev["center"], width, height)
        p1 = self._norm_point(last["center"], width, height)
        vx, vy = (p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt
        speed = (vx * vx + vy * vy) ** 0.5
        area_growth = (last["area"] - prev["area"]) / max(1.0, prev["area"]) / dt
        bottom_delta = (last["det"][5] - prev["det"][5]) / max(1, height) / dt
        trajectory_to_risk = self._track_hits_risk(track, width, height)
        # 위험 가상 영역을 향하거나 통과하는 움직임이면, 아래 방향/박스 확대를
        # 추가로 기다리지 않고 즉시 접근 차량으로 분류한다(수평 통과 차량 포함).
        return trajectory_to_risk and speed >= VEHICLE_MIN_MOTION_RATIO

    def update(self, detections, width, height):
        self.frame_index += 1
        used = set()

        for track in self.tracks:
            match_i = self._match(track, detections, used, width, height)
            if match_i is None:
                track["missed"] += 1
                continue
            used.add(match_i)
            det = detections[match_i]
            track["history"].append({
                "frame": self.frame_index,
                "center": self._center(det),
                "area": self._area(det),
                "det": det,
            })
            track["missed"] = 0

        for i, det in enumerate(detections):
            if i in used:
                continue
            self.tracks.append({
                "id": self.next_id,
                "label": det[0],
                "history": deque([{
                    "frame": self.frame_index,
                    "center": self._center(det),
                    "area": self._area(det),
                    "det": det,
                }], maxlen=8),
                "missed": 0,
            })
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= VEHICLE_TRACK_MAX_MISSED][-20:]

        relevant = [
            t for t in self.tracks
            if t["missed"] <= VEHICLE_TRACK_MAX_MISSED
            and self._track_hits_risk(t, width, height)
        ]
        approaching_tracks = [t for t in relevant if self._is_approaching(t, width, height)]
        approaching = [t["history"][-1]["det"] for t in approaching_tracks]

        stationary = [
            t for t in relevant
            if len(t["history"]) >= VEHICLE_MOTION_MIN_FRAMES
            and t["missed"] == 0
            and not self._is_approaching(t, width, height)
        ]
        stationary_confirmed = bool(relevant) and len(stationary) == len(relevant)
        return approaching, bool(relevant), stationary_confirmed


def crosswalk_alignment_box(width, height):
    """중앙 정렬 안내 영역. 횡단보도 중심이 이 상자 안으로 들어오면 정렬 완료."""
    return (int(width * 0.25), int(height * 0.20),
            int(width * 0.75), int(height * 0.82))


class CrosswalkAlignment:
    def __init__(self):
        self.aligned = False
        self.last_detection_time = None

    def reset(self):
        self.aligned = False
        self.last_detection_time = None

    def update(self, detections, width, height, now=None):
        now = time.monotonic() if now is None else now
        if detections:
            self.last_detection_time = now
            x1, y1, x2, y2 = crosswalk_alignment_box(width, height)
            for det in detections:
                cx = (det[2] + det[4]) / 2
                cy = (det[3] + det[5]) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    self.aligned = True
                    return True
            self.aligned = False
            return False

        # 횡단보도가 검출되지 않으면 정렬 상태도 즉시 초기화한다.
        self.reset()
        return False


class CrosswalkSafety:
    """차량별 중심을 추적한다. 2.5초 동안 위치가 허용 오차 내로 안정되면 CLEAR."""

    def __init__(self):
        self.status = None
        self.tracks = []
        self.next_track_id = 1
        self.last_stop_reason = None

    def reset(self):
        self.status = None
        self.tracks.clear()
        self.last_stop_reason = None

    @staticmethod
    def _center(det):
        return ((det[2]+det[4])/2.0, (det[3]+det[5])/2.0)

    @staticmethod
    def _distance(a, b, width, height):
        dx = (a[0]-b[0]) / max(1, width)
        dy = (a[1]-b[1]) / max(1, height)
        return (dx*dx + dy*dy) ** 0.5

    def update(self, vehicles, width, height, now=None):
        now = time.monotonic() if now is None else now
        unmatched_tracks = set(range(len(self.tracks)))
        urgent = False
        moving = False

        # 먼저 오래 미검출된 트랙을 정리한다.
        self.tracks = [t for t in self.tracks
                       if now - t["last_seen"] < VEHICLE_ABSENCE_CLEAR_SECONDS]
        unmatched_tracks = set(range(len(self.tracks)))

        for det in sorted(vehicles, key=lambda d: proximity_score(d, width, height), reverse=True):
            center = self._center(det)
            candidates = [
                (self._distance(center, self.tracks[i]["center"], width, height), i)
                for i in unmatched_tracks
                if self.tracks[i]["label"] == det[0]
                or self._distance(center, self.tracks[i]["center"], width, height) <= VEHICLE_ASSOCIATION_GAP
            ]
            match = min(candidates, default=None)
            if match is None or match[0] > VEHICLE_ASSOCIATION_GAP:
                self.tracks.append({
                    "id": self.next_track_id,
                    "label": det[0],
                    "center": center,
                    "anchor": center,
                    "stable_since": now,
                    "last_seen": now,
                    "zone": self.zone_index(center[0], width),
                })
                self.next_track_id += 1
                urgent = True
                moving = True
                continue

            _, i = match
            unmatched_tracks.discard(i)
            track = self.tracks[i]
            drift = self._distance(center, track["anchor"], width, height)
            gap = now - track["last_seen"]
            if drift > VEHICLE_CENTER_TOLERANCE or gap > VEHICLE_ABSENCE_CLEAR_SECONDS:
                # 이동 또는 긴 검출 공백: 안정 시간 다시 계산
                track["anchor"] = center
                track["stable_since"] = now
                moving = True
            track["center"] = center
            track["label"] = det[0]
            track["last_seen"] = now
            track["zone"] = self.zone_index(center[0], width)

        # 미검출 차량은 잠시 유지하되, 안정 차량으로 간주하지 않는다.
        active_tracks = [t for t in self.tracks
                         if now - t["last_seen"] < VEHICLE_ABSENCE_CLEAR_SECONDS]
        all_visible_and_stable = bool(active_tracks) and all(
            now - t["last_seen"] <= VEHICLE_FRAME_HOLD_SECONDS
            and now - t["stable_since"] >= VEHICLE_STATIONARY_SECONDS
            for t in active_tracks
        )

        if vehicles and all_visible_and_stable and not moving:
            self.status = "CLEAR"
        elif active_tracks:
            self.status = "WAIT"
        else:
            self.status = "CLEAR"

        self.last_stop_reason = "new_or_moving_vehicle" if (urgent or moving) else None
        return self.status, urgent

    @staticmethod
    def zone_index(center_x, width):
        return min(CROSSWALK_ZONE_COUNT - 1,
                   max(0, int((center_x / max(1, width)) * CROSSWALK_ZONE_COUNT)))


class StairConfirm:
    """계단 박스가 충분히 보일 때만 계단으로 확정한다."""

    def __init__(self):
        self.last = None
        self.count = 0

    def reset(self):
        self.last = None
        self.count = 0

    def is_complete(self, det, width, height):
        _, _, x1, y1, x2, y2 = det
        box_area = max(1, (x2 - x1) * (y2 - y1))
        frame_area = max(1, width * height)
        width_ratio = (x2 - x1) / max(1, width)
        # 계단 전체가 화면에 들어와야 한다는 조건은 사용하지 않는다.
        # 화면에 일부만 보여도 보행 영역에 있고 최소 크기만 넘으면 허용한다.
        overlaps_walk = walking_candidate(det, width, height)
        large_enough = (
            box_area / frame_area >= STAIR_MIN_AREA_RATIO
            or width_ratio >= STAIR_MIN_WIDTH_RATIO
        )
        return overlaps_walk and large_enough

    def update(self, detections, width, height):
        candidates = [
            d for d in detections
            if d[0] in STAIR_LABELS and self.is_complete(d, width, height)
        ]
        if not candidates:
            self.reset()
            return []

        current = max(candidates, key=lambda d: d[1])
        if self.last is not None and same_target(self.last, current):
            self.count += 1
        else:
            self.last = current
            self.count = 1

        if self.count >= STAIR_CONFIRM_FRAMES:
            return [current]
        return []


class ElevatorConfirm:
    """엘리베이터 자체 또는 일반 문에 붙은 안전 경고 스티커를
    함께 확인할 때만 엘리베이터로 확정한다."""

    def __init__(self):
        self.last = None
        self.count = 0
        self.from_sticker = False

    def reset(self):
        self.last = None
        self.count = 0
        self.from_sticker = False

    @staticmethod
    def plausible(det, width, height):
        _, _, x1, y1, x2, y2 = det
        area_ratio = max(1, (x2-x1)*(y2-y1)) / max(1, width*height)
        width_ratio = (x2-x1) / max(1, width)
        height_ratio = (y2-y1) / max(1, height)
        return area_ratio >= ELEVATOR_MIN_AREA_RATIO and width_ratio >= ELEVATOR_MIN_WIDTH_RATIO and height_ratio >= 0.20

    @staticmethod
    def sticker_near_door(sticker, doors):
        _, _, sx1, sy1, sx2, sy2 = sticker
        scx, scy = (sx1+sx2)/2, (sy1+sy2)/2
        for door in doors:
            _, _, dx1, dy1, dx2, dy2 = door
            if box_iou(sticker, door) > 0.01:
                return True
            if dx1-50 <= scx <= dx2+50 and dy1-50 <= scy <= dy2+50:
                return True
        return False

    def update(self, detections, width, height, door_candidates):
        direct = [
            d for d in detections
            if d[0] == "elevator" and d[1] >= ELEVATOR_MIN_CONF
            and self.plausible(d, width, height)
        ]
        stickers = [
            d for d in detections
            if d[0] in ELEVATOR_SUPPORT_LABELS
            and d[1] >= ELEVATOR_STICKER_MIN_CONF
            and self.sticker_near_door(d, door_candidates)
        ]
        candidates = direct + stickers
        if not candidates:
            self.reset()
            return []
        current = max(candidates, key=lambda d: d[1])
        from_sticker = current[0] in ELEVATOR_SUPPORT_LABELS
        if self.last is not None and same_target(self.last, current):
            self.count += 1
        else:
            self.last = current
            self.count = 1
            self.from_sticker = from_sticker
        required = ELEVATOR_STICKER_CONFIRM_FRAMES if from_sticker else ELEVATOR_CONFIRM_FRAMES
        if self.count < required:
            return []
        if from_sticker and door_candidates:
            nearest = min(
                door_candidates,
                key=lambda d: abs(((d[2]+d[4])/2)-((current[2]+current[4])/2)),
            )
            return [("elevator", current[1], nearest[2], nearest[3], nearest[4], nearest[5])]
        return [current]


class DoorComponentConfirm:
    """손잡이는 문 검출 누락을 고려해 독립적으로 2프레임 확인한다.

    문과 같은 프레임에서 가까이 검출되면 바로 후보화하고, 문 박스가
    누락되어도 같은 손잡이가 2회 연속 검출되면 허용한다.
    """

    def __init__(self):
        self.previous = []

    def reset(self):
        self.previous = []

    @staticmethod
    def near_door(component, doors):
        _, _, cx1, cy1, cx2, cy2 = component
        cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        for door in doors:
            _, _, x1, y1, x2, y2 = door
            if box_iou(component, door) > 0.01:
                return True
            if x1 - 60 <= cx <= x2 + 60 and y1 - 60 <= cy <= y2 + 60:
                return True
        return False

    @staticmethod
    def _same_handle_region(a, b):
        if a[0] not in DOOR_COMPONENT_LABELS or b[0] not in DOOR_COMPONENT_LABELS:
            return False
        if box_iou(a, b) >= 0.03:
            return True
        acx, acy = (a[2]+a[4])/2, (a[3]+a[5])/2
        bcx, bcy = (b[2]+b[4])/2, (b[3]+b[5])/2
        return ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5 <= 35

    def update(self, detections, confirmed_doors):
        candidates = [
            d for d in detections
            if d[0] in DOOR_COMPONENT_LABELS
            and d[1] >= CLASS_MIN_CONF.get(d[0], HANDLE_MIN_CONF)
        ]
        accepted = []
        for det in candidates:
            if self.near_door(det, confirmed_doors):
                accepted.append(det)
                continue
            if any(self._same_handle_region(old, det) for old in self.previous):
                accepted.append(det)
        self.previous = candidates
        return accepted


class DoorConfirm:
    """문은 단일 프레임 검출만으로 안내하지 않는다."""

    def __init__(self):
        self.last_box = None
        self.count = 0
        self.confirmed = None

    def reset(self):
        self.last_box = None
        self.count = 0
        self.confirmed = None

    @staticmethod
    def _same_door_region(a, b):
        if a[0] not in DOOR_LABELS or b[0] not in DOOR_LABELS:
            return False
        if box_iou(a, b) >= 0.04:
            return True
        acx, acy = (a[2]+a[4])/2, (a[3]+a[5])/2
        bcx, bcy = (b[2]+b[4])/2, (b[3]+b[5])/2
        scale = max(a[4]-a[2], a[5]-a[3], b[4]-b[2], b[5]-b[3], 1)
        return ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5 <= scale * 0.30

    def update(self, doors):
        if not doors:
            self.reset()
            return []

        door = max(doors, key=lambda d: d[1])
        if self.last_box is not None and self._same_door_region(self.last_box, door):
            self.count += 1
        else:
            self.last_box = door
            self.count = 1
            self.confirmed = None

        if self.count >= 2:
            self.confirmed = door
            return [door]
        return []


class TargetSelector:
    """새 대상이 한 프레임 튄 것만으로 음성이 끊기지 않게 한다."""

    def __init__(self):
        self.current = None
        self.pending = None
        self.pending_count = 0
        self.missing_count = 0

    def reset(self):
        self.current = None
        self.pending = None
        self.pending_count = 0
        self.missing_count = 0

    @staticmethod
    def _key(det, width):
        return None if det is None else (det[0], zone(det[2], det[4], width))

    def update(self, candidates, width, height):
        candidates = sorted(
            candidates,
            key=lambda d: detection_priority(d, width, height),
        )
        candidate = candidates[0] if candidates else None

        if self.current is None:
            self.current = candidate
            self.missing_count = 0
            return self.current

        if candidate is None:
            # 검출이 끊기면 이전 대상을 즉시 폐기한다.
            # 특히 문이 사라졌는데 문 안내가 계속되는 현상을 방지한다.
            self.reset()
            return None

        current_key = self._key(self.current, width)
        candidate_key = self._key(candidate, width)

        if same_target(self.current, candidate) or current_key == candidate_key:
            self.current = candidate
            self.missing_count = 0
            self.pending = None
            self.pending_count = 0
            return self.current

        self.missing_count = 0
        if self.pending is not None and self._key(self.pending, width) == candidate_key:
            self.pending_count += 1
        else:
            self.pending = candidate
            self.pending_count = 1

        # 새 대상은 2프레임 연속 확인 후 전환한다.
        if self.pending_count >= 2:
            self.current = candidate
            self.pending = None
            self.pending_count = 0

        return self.current



class DoorNoticeOCR:
    """문 박스에서만 OCR을 수행하고 '당기세요/미세요'만 음성으로 전달한다."""

    def __init__(self, speech):
        self.speech = speech
        self.jobs = queue.Queue(maxsize=1)
        self.enabled = os.getenv("WALKING_ASSISTANT_OCR", "0").strip().lower() in {"1", "true", "yes"}
        self.running = self.enabled and pytesseract is not None
        self.last_submit = 0.0
        self.last_spoken = {}
        self.thread = None
        if self.running:
            self.thread = threading.Thread(target=self._worker, name="door-ocr", daemon=True)
            self.thread.start()
            print("문 안내문 OCR: 활성화")
        elif pytesseract is None:
            print("문 안내문 OCR: pytesseract가 없어 비활성화")
        else:
            print("문 안내문 OCR: 기본 비활성화 (WALKING_ASSISTANT_OCR=1로 켤 수 있음)")

    def submit(self, frame, doors):
        if not self.running or pytesseract is None or not doors:
            return
        now = time.monotonic()
        if now - self.last_submit < DOOR_OCR_INTERVAL:
            return
        # OCR은 문 검출 박스만 받으며, 전체 프레임의 다른 글자는 처리하지 않는다.
        crops = []
        height, width = frame.shape[:2]
        for det in doors:
            _, conf, x1, y1, x2, y2 = det
            if conf < DOOR_MIN_CONF:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2].copy())
        if not crops:
            return
        self.last_submit = now
        try:
            if self.jobs.full():
                self.jobs.get_nowait()
                self.jobs.task_done()
            self.jobs.put_nowait((now, crops))
        except queue.Full:
            pass

    def _worker(self):
        while self.running:
            try:
                timestamp, crops = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                for crop in crops:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    try:
                        text = pytesseract.image_to_string(gray, lang="kor+eng", config="--psm 11")
                    except Exception as exc:
                        print("문 안내문 OCR 오류:", exc)
                        continue
                    compact = text.replace(" ", "").replace("\n", "")
                    notice = None
                    if "당기세요" in compact or "당기" in compact:
                        notice = "당기세요"
                    elif "미세요" in compact or "미시" in compact:
                        notice = "미세요"
                    if notice:
                        now = time.monotonic()
                        if now - self.last_spoken.get(notice, 0.0) >= DOOR_OCR_REPEAT_SECONDS:
                            self.last_spoken[notice] = now
                            self.speech.say(f"door_notice:{notice}", f"문 안내문. {notice}")
            finally:
                self.jobs.task_done()

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class LatestFrameCapture:
    """카메라 수집과 추론을 분리하고 최신 프레임 1장만 유지한다.

    추론이 느려도 캡처 프레임을 큐에 쌓지 않는다. 추론 스레드는 이전
    프레임 분석이 끝난 뒤, 그 시점의 가장 최신 프레임을 받아 처리한다.
    """

    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.latest_frame = None
        self.sequence = 0
        self.running = True
        self.reader = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self.reader.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self.condition:
                self.latest_frame = frame
                self.sequence += 1
                self.condition.notify_all()

    def isOpened(self):
        return self.capture.isOpened()

    def set(self, prop, value):
        return self.capture.set(prop, value)

    def read(self):
        # 각 호출은 이전에 소비한 프레임보다 새로운 프레임을 기다린다.
        with self.condition:
            previous_sequence = getattr(self, "_last_read_sequence", 0)
            self.condition.wait_for(
                lambda: not self.running or self.sequence > previous_sequence,
                timeout=2.0,
            )
            if self.sequence <= previous_sequence or self.latest_frame is None:
                return False, None
            self._last_read_sequence = self.sequence
            # 프레임 버퍼를 이후 캡처가 덮어쓰지 않도록 복사해 반환
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        # 먼저 카메라를 닫아 블로킹된 read()가 있으면 풀어준다.
        self.capture.release()
        self.reader.join(timeout=2.0)

def sync_cuda_for_profile(model):
    """선택적으로 CUDA 비동기 추론을 동기화해 단계별 시간을 정확히 잰다."""
    if not PROFILE_CUDA_SYNC:
        return
    try:
        import torch
        device = getattr(model, "device", None)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    except Exception:
        pass


def main():
    print("YOLOE 모델 로딩 중... (Raspberry Pi 터미널 모드)", flush=True)
    print(f"카메라: {CAMERA_INDEX} | 추론 크기: {INFERENCE_SIZE} | 단계별 시간 측정: {'ON' if PROFILE_TIMING else 'OFF'}", flush=True)
    try:
        import torch
        torch.set_num_threads(max(1, int(os.getenv("WALKING_ASSISTANT_CPU_THREADS", "3"))))
    except Exception as exc:
        print(f"PyTorch CPU 스레드 설정을 건너뜁니다: {exc}")
    model = YOLOE(MODEL_PATH)
    model.set_classes(PROMPTS)
    active_prompt_mode = "GENERAL"

    raw_cap = cv2.VideoCapture(CAMERA_INDEX)
    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # 드라이버 내부 버퍼도 가능한 한 작게 설정해 오래된 프레임 누적을 줄인다.
    raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not raw_cap.isOpened():
        raw_cap.release()
        raise RuntimeError("웹캠을 열 수 없습니다.")

    cap = LatestFrameCapture(raw_cap)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    imu = OptionalIMU(IMU_PORT, IMU_BAUDRATE)
    speech = Speech()
    door_ocr = DoorNoticeOCR(speech)
    machine = SituationMachine()
    crosswalk_safety = CrosswalkSafety()
    crosswalk_alignment = CrosswalkAlignment()
    target_selector = TargetSelector()
    door_confirm = DoorConfirm()
    door_component_confirm = DoorComponentConfirm()
    stair_confirm = StairConfirm()
    elevator_confirm = ElevatorConfirm()
    last_guidance_key = None
    last_traffic_status = None
    last_crosswalk_clear_announce = 0.0
    last_crosswalk_align_announce = 0.0
    profile_samples = deque(maxlen=120)
    last_profile_log = time.monotonic()

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            capture_finished = time.perf_counter()
            if not ok:
                break

            h, w = frame.shape[:2]
            inference_frame = prepare_inference_frame(frame)
            preprocess_finished = time.perf_counter()
            sync_cuda_for_profile(model)
            inference_started = time.perf_counter()
            result = model.predict(
                inference_frame,
                conf=YOLO_INFERENCE_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=INFERENCE_SIZE,
                max_det=30,
                verbose=False,
            )[0]
            sync_cuda_for_profile(model)
            inference_finished = time.perf_counter()
            urgent_detect_to_submit_ms = None
            urgent_speech_call_ms = None

            all_dets = []
            crosswalk_vehicles = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = label_of(result, cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det = (label, conf, x1, y1, x2, y2)
                if conf >= class_min_conf(label) and valid_geometry(det, w, h):
                    all_dets.append(det)
                    if label in VEHICLE_LABELS and conf >= CROSSWALK_VEHICLE_MIN_CONF:
                        crosswalk_vehicles.append(det)

            crosswalk_vehicles = deduplicate_vehicle_detections(crosswalk_vehicles)
            all_dets = deduplicate_detections(all_dets)

            # 엘리베이터 자체 또는 문에 붙은 안전 경고 스티커가
            # 일반 문과 함께 확인될 때만 엘리베이터로 확정한다.
            door_candidates_for_elevator = [
                d for d in all_dets
                if d[0] in DOOR_LABELS and d[1] >= DOOR_MIN_CONF
            ]
            confirmed_elevators = elevator_confirm.update(
                all_dets, w, h, door_candidates_for_elevator
            )

            def near_elevator(det):
                return False

            elevator_components = []

            raw_doors = [
                d for d in all_dets
                if d[0] in DOOR_LABELS
                and d[1] >= DOOR_MIN_CONF
                and not near_elevator(d)
            ]
            confirmed_doors = door_confirm.update(raw_doors)

            valid_components = door_component_confirm.update(all_dets, confirmed_doors)

            filtered = [
                d for d in all_dets
                if d in confirmed_doors
                or d in valid_components
                or d in confirmed_elevators
                or d in elevator_components
                or d[0] not in (
                    DOOR_LABELS | DOOR_COMPONENT_LABELS | ELEVATOR_LABELS
                    | ELEVATOR_COMPONENT_LABELS | ELEVATOR_SUPPORT_LABELS
                )
            ]
            confirmed_stairs = stair_confirm.update(all_dets, w, h)

            # 계단은 최소 크기와 연속 검출이 확인된 경우에만 안내 후보에 포함한다.
            filtered_for_guidance = [
                d for d in filtered
                if (
                    (d[0] not in STAIR_LABELS or d in confirmed_stairs)
                    and (d[0] not in ELEVATOR_LABELS or d in confirmed_elevators)
                )
            ]

            # 일반 안내 후보는 기존처럼 중앙 보행 ROI를 사용
            # 높이·기하학 조건을 한 번 더 적용해 바닥/벽면 오탐을 줄인다.
            active = [
                d for d in filtered_for_guidance
                if walking_candidate(d, w, h)
                and valid_geometry(d, w, h)
            ]

            # 횡단보도는 ROI 밖에 있어도 상태 전환에 사용
            crosswalk_dets = [
                d for d in filtered_for_guidance
                if d[0] in CROSSWALK_LABELS
            ]
            crosswalk_aligned = crosswalk_alignment.update(crosswalk_dets, w, h)
            elevator_dets = confirmed_elevators

            state = machine.update(active + crosswalk_dets + confirmed_stairs + elevator_dets)

            # 차량 기반 멈춤/건너기 안내는 횡단보도 모드에서만 작동한다.
            # 일반 보행 모드에서는 차량이 보여도 기존 음성을 끊거나 정지 안내를 하지 않는다.
            if state == "CROSSWALK":
                traffic_status, urgent_stop = crosswalk_safety.update(crosswalk_vehicles, w, h)
                speech.set_vehicle_priority(traffic_status == "WAIT")
                if urgent_stop:
                    speech_request_started = time.perf_counter()
                    urgent_detect_to_submit_ms = (speech_request_started - inference_finished) * 1000.0
                    speech.say_urgent("traffic_stop", "멈추세요")
                    urgent_speech_call_ms = (time.perf_counter() - speech_request_started) * 1000.0
                    last_guidance_key = "traffic:WAIT"
            else:
                crosswalk_safety.reset()
                traffic_status, urgent_stop = "CLEAR", False
                speech.set_vehicle_priority(False)

            # 자동/수동 횡단보도 모드에서는 교통 전용 프롬프트로 전환한다.
            # 전체 프롬프트 복귀도 모드 전환 시에만 수행한다.
            desired_prompt_mode = "TRAFFIC" if state == "CROSSWALK" else "GENERAL"
            if desired_prompt_mode != active_prompt_mode:
                model.set_classes(TRAFFIC_PROMPTS if desired_prompt_mode == "TRAFFIC" else PROMPTS)
                active_prompt_mode = desired_prompt_mode
                print(f"추론 프롬프트 모드: {active_prompt_mode} ({INFERENCE_SIZE}px)")

            # ----------------------------------------------------
            # 최우선 정지 게이트: 신규 차량 검출 시 다른 안내를 중단하고 즉시 "멈추세요".
            # 횡단보도 상태뿐 아니라 일반 보행 모드에서도 동일하게 적용한다.
            # ----------------------------------------------------
            primary = None

            # 횡단보도 자체가 인식된 때만 정렬 안내/횡단 상태를 말한다.
            # 중심 정렬 전에는 가운데 정렬을 유도하고, 정렬되면 교통 판단 단계로 넘어간다.
            if state == "CROSSWALK" and crosswalk_dets:
                status_changed = traffic_status != last_traffic_status
                if urgent_stop:
                    last_guidance_key = "traffic:WAIT"
                elif not crosswalk_aligned:
                    now = time.monotonic()
                    if now - last_crosswalk_align_announce >= 3.0:
                        speech.say("crosswalk_align", "횡단보도를 가운데로 맞춰주세요", force=True)
                        last_crosswalk_align_announce = now
                    last_crosswalk_clear_announce = 0.0
                    last_guidance_key = "crosswalk:ALIGN"
                elif status_changed and traffic_status == "WAIT":
                    speech.say("crosswalk_wait", "멈추세요", force=True)
                    last_guidance_key = "traffic:WAIT"
                elif traffic_status == "CLEAR":
                    # 정렬 후 횡단보도 차량 판단이 끝난 동안 반복 안내
                    now = time.monotonic()
                    if now - last_crosswalk_clear_announce >= CROSSWALK_REPEAT_SECONDS:
                        speech.say("crosswalk_clear", "건너세요", force=True)
                        last_crosswalk_clear_announce = now
                    last_guidance_key = "traffic:CLEAR"
                else:
                    last_crosswalk_clear_announce = 0.0
                target_selector.reset()
            elif urgent_stop or traffic_status == "WAIT":
                last_crosswalk_align_announce = 0.0
                last_crosswalk_clear_announce = 0.0
                # stop 상태가 활성화된 동안 일반 사물·문·계단 음성은 내보내지 않는다.
                if urgent_stop:
                    pass  # 즉시 정지 음성은 결과 직후 이미 출력함
                elif last_guidance_key != "traffic:WAIT":
                    speech.say("traffic_wait", "멈추세요", force=True)
                last_guidance_key = "traffic:WAIT"
                target_selector.reset()
            else:
                # 횡단보도가 현재 프레임에서 확인되지 않으면 횡단 안내를 하지 않는다.
                last_crosswalk_align_announce = 0.0
                last_crosswalk_clear_announce = 0.0
                obstacle_candidates = [
                    d for d in active
                    if is_walk_obstacle(d)
                    and (d[0] not in DOOR_LABELS or d in confirmed_doors)
                ]
                # ROI 안에 안내 후보가 없더라도 화면에서 유효하게 인식된 장애물을
                # 무음으로 버리지 않는다. ROI 후보가 있으면 기존처럼 그 대상을 우선한다.
                if not obstacle_candidates:
                    obstacle_candidates = [
                        d for d in filtered_for_guidance
                        if is_walk_obstacle(d)
                        and valid_geometry(d, w, h)
                        and (d[0] not in DOOR_LABELS or d in confirmed_doors)
                    ]
                primary = target_selector.update(obstacle_candidates, w, h)

                if primary is not None:
                    guidance_key = f"obstacle:{primary[0]}:{zone(primary[2], primary[4], w)}"
                    text = guidance_for_detection(primary, w)
                    speech.say(guidance_key, text, force=(guidance_key != last_guidance_key))
                    last_guidance_key = guidance_key
                else:
                    last_guidance_key = None

            last_traffic_status = traffic_status
            # OCR은 확인된 문 영역만 비동기 전달한다. 차량 감지 중에는 문 안내를 보류한다.
            if not crosswalk_vehicles:
                door_ocr.submit(frame, confirmed_doors)

            loop_finished = time.perf_counter()
            if PROFILE_TIMING:
                profile_samples.append({
                    "capture": (capture_finished - frame_started) * 1000.0,
                    "preprocess": (preprocess_finished - capture_finished) * 1000.0,
                    "inference": (inference_finished - inference_started) * 1000.0,
                    "frame": (loop_finished - frame_started) * 1000.0,
                    "detect_to_stop": urgent_detect_to_submit_ms,
                    "speech_call": urgent_speech_call_ms,
                })
                now_profile = time.monotonic()
                if now_profile - last_profile_log >= PROFILE_INTERVAL_SECONDS and profile_samples:
                    def avg_ms(key):
                        values = [sample[key] for sample in profile_samples if sample[key] is not None]
                        return (sum(values) / len(values)) if values else None

                    urgent_count = sum(sample["detect_to_stop"] is not None for sample in profile_samples)
                    detect_to_stop = avg_ms("detect_to_stop")
                    speech_call = avg_ms("speech_call")
                    urgent_summary = (
                        f"차량감지→음성요청={detect_to_stop:.1f}ms | 음성요청처리={speech_call:.1f}ms"
                        if detect_to_stop is not None else "차량 긴급 이벤트 없음"
                    )
                    print(
                        "[성능 평균 최근 {}프레임] 캡처대기={:.1f}ms | 전처리={:.1f}ms | "
                        "YOLO추론={:.1f}ms | 전체루프={:.1f}ms | 긴급횟수={} | {}".format(
                            len(profile_samples), avg_ms("capture"), avg_ms("preprocess"),
                            avg_ms("inference"), avg_ms("frame"), urgent_count, urgent_summary
                        )
                    )
                    profile_samples.clear()
                    last_profile_log = now_profile


    finally:
        door_ocr.close()
        cap.release()


if __name__ == "__main__":
    main()
