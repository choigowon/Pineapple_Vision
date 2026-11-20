import os
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "False")
os.environ.setdefault("WALKING_ASSISTANT_CPU_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import threading
import queue
import shutil
import subprocess
import warnings
import wave
import hashlib
from pathlib import Path
from collections import deque

import cv2
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.load` is deprecated.*",
    category=FutureWarning,
)

from ultralytics import YOLOE

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

# ============================================================
# YOLOE 기반 상황형 보행 보조 Raspberry Pi 터미널 프로토타입
# ============================================================

MODEL_PATH = os.getenv("WALKING_ASSISTANT_MODEL", "yoloe-26n-seg.pt")
CAMERA_INDEX = int(os.getenv("WALKING_ASSISTANT_CAMERA_INDEX", "0"))
CONFIDENCE = 0.14
YOLO_INFERENCE_CONFIDENCE = float(os.getenv("WALKING_ASSISTANT_YOLO_CONF", "0.08"))
NMS_IOU = 0.40
INFERENCE_SIZE = int(os.getenv("WALKING_ASSISTANT_INFERENCE_SIZE", "320"))
PROFILE_TIMING = os.getenv("WALKING_ASSISTANT_PROFILE", "0").strip().lower() not in {"0", "false", "no"}
PROFILE_INTERVAL_SECONDS = 5.0
PROFILE_CUDA_SYNC = os.getenv("WALKING_ASSISTANT_PROFILE_CUDA_SYNC", "0").strip().lower() in {"1", "true", "yes"}
STATE_CONFIRM_FRAMES = 4
AUTO_CLAHE = os.getenv("WALKING_ASSISTANT_AUTO_CLAHE", "1").strip().lower() in {"1", "true", "yes"}
SPEECH_COOLDOWN = 0.22
URGENT_SPEECH_COOLDOWN = 0.12

CROSSWALK_VEHICLE_MIN_CONF = 0.10
CROSSWALK_CLEAR_SECONDS = 2.0
CROSSWALK_MIN_CONF = 0.30
CROSSWALK_CONFIRM_FRAMES = 2
CROSSWALK_ENTER_CONFIRM_SECONDS = 0.5
CROSSWALK_EXIT_HOLD_SECONDS = 2.7
CROSSWALK_DETECTION_GAP_SECONDS = 2.0

VEHICLE_FRAME_HOLD_SECONDS = 0.5
VEHICLE_ABSENCE_CLEAR_SECONDS = VEHICLE_FRAME_HOLD_SECONDS
VEHICLE_STATIONARY_SECONDS = 0.7
VEHICLE_CENTER_TOLERANCE = 0.035
VEHICLE_ASSOCIATION_GAP = 0.22
CROSSWALK_ZONE_COUNT = 4
CROSSWALK_REPEAT_SECONDS = 3.0
CROSSWALK_STOP_TO_CROSS_DELAY_SECONDS = 2.0
CROSSWALK_ALIGNMENT_HOLD_SECONDS = 2.0
PIPER_SPEED_FACTOR = float(os.getenv("WALKING_ASSISTANT_PIPER_SPEED", "1.12"))
SHOW_CROSSWALK_DEBUG = os.getenv("WALKING_ASSISTANT_SHOW_DEBUG", "0").strip().lower() in {"1", "true", "yes"}

STAIR_CONFIRM_FRAMES = 2
STAIR_MIN_AREA_RATIO = 0.004
STAIR_MIN_WIDTH_RATIO = 0.045
ELEVATOR_MIN_CONF = 0.96
ELEVATOR_CONFIRM_FRAMES = 5
ELEVATOR_MIN_AREA_RATIO = 0.08
ELEVATOR_MIN_WIDTH_RATIO = 0.25
ELEVATOR_STICKER_MIN_CONF = 0.72
ELEVATOR_STICKER_CONFIRM_FRAMES = 3

CROSSWALK_PROMPTS = [
    "white striped pedestrian crosswalk pattern on a floor",
    "white parallel zebra crossing stripes on dark surface",
    "miniature white zebra pedestrian crosswalk model",
    "white stripe lines of crosswalk pattern",
    "black and white striped crosswalk mat on floor",
    "small model crosswalk road with white stripes",
    "white stripes on black mat",
    "zebra stripes pattern on floor",
]
CROSSWALK_LABELS = set(CROSSWALK_PROMPTS)

STAIR_PROMPTS = [
    "indoor architectural staircase with steps leading up or down",
    "outdoor concrete flight of stairs with handrail",
    "building entrance staircase with vertical risers",
    "descending or ascending staircase structure",
]

TEXT_SIGN_PROMPTS = [
    "push or pull sign sticker on a door",
    "push pull door sign sticker with text",
    "push text sign on door",
    "pull text sign on door",
    "door sign with English text",
]

CAR_PROMPT = "automobile car"
SCOOTER_PROMPT = "electric kick scooter on a road"
MOTORCYCLE_PROMPT = "motorcycle on a road"
PLAIN_SCOOTER_PROMPT = "scooter on a road"
CAR_DISTANT_PROMPT = "small passenger car with wheels"
CAR_FAR_PROMPT = "far away vehicle"
TOY_CAR_PROMPTS = {
    "toy car",
    "miniature car",
    "small car",
    "car",
}
MOTORCYCLE_RIDER_PROMPT = "motorcycle with a rider on a road"
MOTORCYCLE_DISTANT_PROMPT = "small distant motorcycle with rider"
MOTORCYCLIST_PROMPT = "motorcyclist riding a motorcycle on a road"
SCOOTER_RIDER_PROMPT = "person riding an electric scooter on a road"
HANGER_PROMPTS = {
    "freestanding clothes rack with hanging clothes",
    "clothing rack with several garments on hangers",
    "indoor garment rack with clothes hanging from it",
}
HANGER_LABELS = HANGER_PROMPTS
RAILING_LABELS = {"railing"}
GLASS_DOOR_LABELS = {
    "glass door", "transparent glass door", "clear glass door",
    "glass entrance door", "large glass entrance door",
    "commercial glass entrance door", "glass storefront door",
    "glass panel door", "framed glass door", "glass door with metal frame",
    "transparent sliding glass door", "sliding glass entrance door",
    "front view of a glass door", "glass door at a building entrance",
    "glass entrance door with visible handle",
    "transparent glass door in a metal frame",
    "glass door panel with vertical pull handle",
    "glass entrance door viewed from outside with a doorway and frame",
    "exterior glass entrance door with visible door frame",
    "storefront glass entrance door viewed from outside",
    "commercial building glass entry door with a visible doorway",
    "outside view of transparent sliding glass entrance door",
    "full height transparent glass entrance door",
    "clear glass door with faint reflections and a visible frame",
    "glass entry door with dark metal frame",
    "transparent glass door with vertical pull handle",
    "glass door with a vertical metal handle",
    "transparent glass entrance door with a visible handle",
}

BED_LABELS = {
    "bed with mattress and pillow",
    "bed with visible mattress, pillow and blanket in a bedroom",
}
WALL_SURFACE_LABELS = {
    "tiled wall", "ceramic wall tiles", "wall covered with square ceramic tiles",
    "close-up of a tiled wall", "wall with a grid of square tiles",
}
DOOR_LOCK_LABELS = {
    "door lock", "door lock with handle", "keypad door lock",
    "electronic keypad door lock", "deadbolt lock on a door", "door lever lock",
}

STEEL_DOOR_LABELS = {
    "steel door", "metal door", "metal entrance door", "metal security door",
    "steel entrance door with visible handle",
    "metal security door in building entrance",
    "closed metal entrance door", "metal door with a handle",
}

SLIDING_DOOR_LABELS = {
    "sliding door", "glass sliding door", "sliding glass door", "manual sliding glass door",
    "automatic glass sliding door",
}

GLASS_DOOR_TYPE_LABELS = {
    "hinged glass door with a frame",
    "glass hinged entrance door",
    "glass door with a vertical pull handle",
    *SLIDING_DOOR_LABELS
}

LEVER_HANDLE_PROMPTS = {
    "door lever handle",
    "lever door handle",
    "silver or black door handle lever",
    "black plastic lever door handle",
    "black horizontal door lever handle",
    "dark plastic door handle lever",
    "matte black lever handle on door",
    "horizontal door lever handle",
    "black door handle",
    "plastic door handle",
}
ROUND_KNOB_PROMPT = "round spherical doorknob"
PULL_BAR_PROMPT = "long vertical metallic door handle bar"
PUSH_BAR_PROMPT = "horizontal exit push bar installed across a door"
DOOR_LOCK_PROMPT = "digital keypad smart door lock on a door panel"
DOOR_BUTTON_PROMPT = "door push button switch"

HANDLE_TYPE_LABELS = {
    ROUND_KNOB_PROMPT,
    *LEVER_HANDLE_PROMPTS,
    PULL_BAR_PROMPT,
    PUSH_BAR_PROMPT,
    "recessed handle flush with a sliding door surface",
}

LOCK_TYPE_LABELS = {
    DOOR_LOCK_PROMPT,
}

CLASS_MIN_CONF = {
    "bag": 0.48,
    "backpack": 0.48,

    # 의자 및 자전거 인식 빡세게 조정 (임계값 상향)
    "chair": 0.55,
    "bicycle": 0.58,

    # 문버튼 및 오만 사물 오인식 방지 (임계값 상향)
    DOOR_BUTTON_PROMPT: 0.58,
    **{label: 0.25 for label in LEVER_HANDLE_PROMPTS},
    ROUND_KNOB_PROMPT: 0.65,
    PULL_BAR_PROMPT: 0.38,
    PUSH_BAR_PROMPT: 0.42,

    # 미닫이문 감지 민감도 유지
    "sliding glass door": 0.20,
    "glass sliding door": 0.20,
    "manual sliding glass door": 0.20,
    "sliding door": 0.20,

    "bed with mattress and pillow": 0.40,
    "bed with visible mattress, pillow and blanket in a bedroom": 0.40,
    "tiled wall": 0.42,
    "ceramic wall tiles": 0.42,
    "wall covered with square ceramic tiles": 0.44,
    "close-up of a tiled wall": 0.44,
    "wall with a grid of square tiles": 0.44,
    "refrigerator": 0.60,
    "refrigerator in a kitchen": 0.60,
    "large household refrigerator": 0.65,
    "shelf": 0.70,
    "bookshelf": 0.70,
    "storage shelf": 0.70,
    "freestanding storage shelf": 0.72,
    "wardrobe": 0.32,
    "closet": 0.32,
    **{label: 0.56 for label in HANGER_PROMPTS},
    "box": 0.45,
    "desk": 0.60,
    "door": 0.22,
    "wooden door": 0.35,
    **{label: 0.68 for label in STEEL_DOOR_LABELS},
    "glass entrance door": 0.30,
    "automatic glass door": 0.42,
    "digital keypad smart door lock on a door panel": 0.28,
    "electronic keypad door lock": 0.28,
    "door lock": 0.32,
    CAR_PROMPT: 0.30,
    CAR_DISTANT_PROMPT: 0.30,
    CAR_FAR_PROMPT: 0.25,
    SCOOTER_PROMPT: 0.25,
    MOTORCYCLE_PROMPT: 0.17,
    MOTORCYCLE_RIDER_PROMPT: 0.15,
    MOTORCYCLE_DISTANT_PROMPT: 0.15,
    MOTORCYCLIST_PROMPT: 0.15,
    PLAIN_SCOOTER_PROMPT: 0.25,
    SCOOTER_RIDER_PROMPT: 0.15,
    "motorcycle": 0.17,
    "motorbike": 0.17,
    "car": 0.25,
    "scooter": 0.25,
    "kick scooter": 0.25,
    "electric scooter": 0.17,
    "electric kick scooter": 0.17,
    "person": 0.42,
    "stairs": 0.34,
    "staircase": 0.34,
    "stairway steps": 0.34,
    "stair steps on a staircase": 0.34,
    "indoor staircase with visible steps": 0.34,
    "staircase with visible steps": 0.34,
    "stairs with visible steps": 0.34,
    "flight of stairs": 0.34,
    "stair steps viewed from the front": 0.34,
    "staircase with several visible steps": 0.34,
    "downward stairs": 0.28,
    "down staircase": 0.28,
    "descending staircase": 0.28,
    "stairs going down": 0.28,
    "staircase descending from above": 0.32,
    "stair landing": 0.46,
    "elevator": ELEVATOR_MIN_CONF,
    "elevator safety warning sticker": ELEVATOR_STICKER_MIN_CONF,
    "do not lean on elevator door sticker": ELEVATOR_STICKER_MIN_CONF,
    "fall hazard sticker on elevator door": ELEVATOR_STICKER_MIN_CONF,
    "elevator safety notice": ELEVATOR_STICKER_MIN_CONF,
}

for prompt in CROSSWALK_PROMPTS:
    CLASS_MIN_CONF[prompt] = CROSSWALK_MIN_CONF

for prompt in TEXT_SIGN_PROMPTS:
    CLASS_MIN_CONF[prompt] = 0.14

for prompt in TOY_CAR_PROMPTS:
    CLASS_MIN_CONF[prompt] = 0.18

PROMPTS = [
    "bag", "backpack",
    "box", "desk", "chair",
    "bed with mattress and pillow",
    "bed with visible mattress, pillow and blanket in a bedroom",
    "shelf", "bookshelf", "storage shelf", "freestanding storage shelf", "wardrobe", "closet",
    "refrigerator", "refrigerator in a kitchen", "large household refrigerator",
    *sorted(HANGER_PROMPTS),
    *sorted(WALL_SURFACE_LABELS),
    "tree", CAR_PROMPT, "car", CAR_DISTANT_PROMPT, CAR_FAR_PROMPT,
    *sorted(TOY_CAR_PROMPTS),
    SCOOTER_PROMPT, PLAIN_SCOOTER_PROMPT, SCOOTER_RIDER_PROMPT, "scooter",
    MOTORCYCLE_PROMPT, MOTORCYCLE_RIDER_PROMPT, MOTORCYCLE_DISTANT_PROMPT,
    MOTORCYCLIST_PROMPT, "motorcycle", "bollard",
    "utility pole", "bicycle", "person", "curb", "railing",
    "door", "entrance door", "interior door", "wooden door", "closed door", "open door",
    "steel door", "glass entrance door", "sliding glass door", "automatic glass door", "glass sliding door", "manual sliding glass door", "sliding door",
    
    *sorted(LEVER_HANDLE_PROMPTS),
    ROUND_KNOB_PROMPT,
    DOOR_LOCK_PROMPT,
    PULL_BAR_PROMPT,
    PUSH_BAR_PROMPT,
    DOOR_BUTTON_PROMPT,
    
    *STAIR_PROMPTS,
    "elevator", "elevator safety warning sticker", "elevator safety notice",
    *CROSSWALK_PROMPTS,
    *TEXT_SIGN_PROMPTS,
]

STAIR_LABELS = set(STAIR_PROMPTS) | {"stairs", "staircase"}

for prompt in STAIR_PROMPTS:
    CLASS_MIN_CONF[prompt] = 0.42

DOOR_LABELS = {
    "door", "entrance door", "interior door", "wooden door", "closed door", "open door",
    "sliding door", "glass sliding door", "sliding glass door", "manual sliding glass door",
    *sorted(GLASS_DOOR_TYPE_LABELS),
    "automatic door", "glass door", "hinged door",
    *GLASS_DOOR_LABELS, *STEEL_DOOR_LABELS,
}

DOOR_COMPONENT_LABELS = {
    *HANDLE_TYPE_LABELS,
    *LOCK_TYPE_LABELS,
    DOOR_BUTTON_PROMPT,
}

STAIR_LABELS = {
    "stairs", "staircase", "stairway steps", "stair steps on a staircase",
    "indoor staircase with visible steps", "staircase with visible steps",
    "stairs with visible steps", "flight of stairs",
    "stair steps viewed from the front", "staircase with several visible steps",
    "real staircase with clearly visible treads and risers",
    "staircase viewed from the bottom with several visible steps",
    "indoor staircase with multiple horizontal treads",
    "downward stairs", "down staircase", "descending staircase",
    "stairs going down", "staircase descending from above", "stair landing",
}

ELEVATOR_LABELS = {"elevator"}
ELEVATOR_SUPPORT_LABELS = {
    "elevator safety warning sticker",
    "do not lean on elevator door sticker",
    "fall hazard sticker on elevator door",
    "elevator safety notice",
}

VEHICLE_LABELS = {
    CAR_PROMPT, *TOY_CAR_PROMPTS, "car",
    SCOOTER_PROMPT, SCOOTER_RIDER_PROMPT,
    MOTORCYCLE_PROMPT, MOTORCYCLE_RIDER_PROMPT,
    MOTORCYCLE_DISTANT_PROMPT, MOTORCYCLIST_PROMPT, PLAIN_SCOOTER_PROMPT,
    "motorcycle", "motorbike", "scooter", "kick scooter",
    "electric scooter", "electric kick scooter", "bicycle",
}

HANDLE_MIN_CONF = 0.18
DOOR_MIN_CONF = 0.15

KOREAN_LABELS = {
    "bag": "가방",
    "backpack": "가방",
    "box": "상자",
    "desk": "책상",
    **{label: "행거" for label in HANGER_PROMPTS},
    "chair": "의자",
    "bed with mattress and pillow": "침대",
    "bed with visible mattress, pillow and blanket in a bedroom": "침대",
    "shelf": "선반",
    "bookshelf": "선반",
    "storage shelf": "선반",
    "freestanding storage shelf": "선반",
    "wardrobe": "옷장",
    "closet": "옷장",
    "refrigerator": "냉장고",
    "refrigerator in a kitchen": "냉장고",
    "large household refrigerator": "냉장고",
    **{label: "철문" for label in STEEL_DOOR_LABELS},
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
    **{label: "차량" for label in TOY_CAR_PROMPTS},
    "motorcycle": "오토바이",
    "motorbike": "오토바이",
    "curb": "연석",
    "railing": "난간",
    "scooter": "킥보드",
    "kick scooter": "킥보드",
    "electric scooter": "전동 킥보드",
    "electric kick scooter": "전동 킥보드",
    "bollard": "볼라드",
    "utility pole": "전봇대",
    "bicycle": "자전거",
    "person": "사람",
    "stairs": "계단",
    "staircase": "계단",
    "stairway steps": "계단",
    "stair steps on a staircase": "계단",
    "indoor staircase with visible steps": "계단",
    "staircase with visible steps": "계단",
    "stairs with visible steps": "계단",
    "flight of stairs": "계단",
    "stair steps viewed from the front": "계단",
    "staircase with several visible steps": "계단",
    "downward stairs": "계단",
    "down staircase": "계단",
    "descending staircase": "계단",
    "stairs going down": "계단",
    "staircase descending from above": "계단",
    "stair landing": "계단",

    ROUND_KNOB_PROMPT: "원형 문고리",
    **{label: "레버 손잡이" for label in LEVER_HANDLE_PROMPTS},
    DOOR_LOCK_PROMPT: "도어락",
    PULL_BAR_PROMPT: "문손잡이",
    PUSH_BAR_PROMPT: "푸시바",
    DOOR_BUTTON_PROMPT: "문 버튼",
    "recessed handle flush with a sliding door surface": "매립 손잡이",

    "entrance door": "출입문",
    "interior door": "실내 문",
    "wooden door": "나무 문",
    "closed door": "닫힌 문",
    "open door": "열린 문",
    "automatic door": "자동문",
    "automatic glass door": "자동 유리문",
    "automatic glass sliding door": "자동 유리문",
    "glass sliding door": "유리 미닫이문",
    "sliding glass door": "유리 미닫이문",
    "manual sliding glass door": "유리 미닫이문",
    "sliding door": "미닫이문",
    "hinged glass door with a frame": "여닫이 유리문",
    "glass hinged entrance door": "여닫이 유리문",
    "glass door with a vertical pull handle": "손잡이형 유리문",
    **{label: "유리문" for label in GLASS_DOOR_LABELS},
}

class AsyncOCRReader:
    def __init__(self, speech_instance):
        self.speech = speech_instance
        self.latest_roi = None
        self.lock = threading.Lock()
        self.has_job = threading.Event()
        self.running = True
        self.last_ocr_time = 0.0
        self.last_detected_time = 0.0
        self.last_detected_type = None
        self.thread = threading.Thread(target=self._worker, name="ocr-worker", daemon=True)
        self.thread.start()

    def request_ocr(self, roi_img):
        if not HAS_PYTESSERACT:
            return
        with self.lock:
            self.latest_roi = roi_img
        self.has_job.set()

    def _preprocess(self, roi_img):
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # 1. 적정 해상도 스케일링 (너무 작으면 3배, 크면 2배)
        h, w = gray.shape[:2]
        scale = 3.0 if max(h, w) < 120 else 2.0
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. CLAHE 대비 보정 + 가우시안 블러로 고주파 노이즈 억제
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        # 3. 정방향/반전 Otsu 이진화 교차 적용
        self._toggle = not getattr(self, "_toggle", False)
        thresh_flag = cv2.THRESH_BINARY_INV if self._toggle else cv2.THRESH_BINARY
        _, binary = cv2.threshold(sharp, 0, 255, thresh_flag + cv2.THRESH_OTSU)

        # 4. 모폴로지 연산으로 문틀 선이나 점 같은 미세 잡음 제거
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 5. 흰색 여백 추가
        return cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    @staticmethod
    def _match_door_sign(text):
        # 줄/단어 단위로 분리하여 1~2글자짜리 배경 잡음('A', 'F', 'YF', 'CD' 등) 제거
        tokens = [
            "".join(c for c in word.upper() if c.isalpha())
            for word in text.split()
        ]
        meaningful_words = [w for w in tokens if len(w) >= 3]
        joined = "".join(meaningful_words)

        if len(joined) < 3:
            return None

        # 3~4글자 이상의 신뢰도 높은 패턴만 허용 (오탐 방지)
        pull_patterns = ("PULL", "ULL", "PUL", "PLL", "FULL", "POLL")
        push_patterns = ("PUSH", "USH", "PUS", "PSH", "POSH", "BUSH")

        if any(p in joined for p in pull_patterns):
            return "pull"
        if any(p in joined for p in push_patterns):
            return "push"
        return None

    def _worker(self):
        # --psm 6(단일 텍스트 블록) 위주로 사용하고, 영문 대문자만 허용
        config = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        while self.running:
            if not self.has_job.wait(timeout=0.2):
                continue
            self.has_job.clear()

            now = time.monotonic()
            if now - self.last_ocr_time < 0.25:
                continue

            with self.lock:
                roi_img = self.latest_roi
                self.latest_roi = None

            if roi_img is None:
                continue

            self.last_ocr_time = now

            try:
                img = self._preprocess(roi_img)
                text = pytesseract.image_to_string(img, config=config).strip()
                if not text:
                    continue

                detected = self._match_door_sign(text)
                if detected == "pull":
                    print(f"[OCR 성공: PULL] {repr(text)}", flush=True)
                    self.speech.say("ocr_pull", "문 당기세요", force=True)
                elif detected == "push":
                    print(f"[OCR 성공: PUSH] {repr(text)}", flush=True)
                    self.speech.say("ocr_push", "문 미세요", force=True)

            except Exception:
                pass

    def close(self):
        self.running = False
        self.has_job.set()

class UltrasonicSensor:
    TRIG_PIN = 23
    ECHO_PIN = 24
    MAX_DISTANCE_CM = 400.0
    WARNING_DISTANCE_CM = 30.0

    def __init__(self):
        self.gpio = None
        self.ready = False
        self._beep_lock = threading.Lock()
        self._last_beep = 0.0
        self._beep_interval = 0.65
        #self._aplay = shutil.which("aplay")
        self._aplay = shutil.which("pw-play")
        self._alsa_device = os.getenv("WALKING_ASSISTANT_ALSA_DEVICE", "plughw:2,0")
        self._beep_samples = self._make_double_beep()
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.ECHO_PIN, GPIO.IN)
            time.sleep(0.05)
            self.ready = True
            print("초음파 센서 연결: Trig=GPIO23(물리 16), Echo=GPIO24(물리 18)", flush=True)
        except Exception as exc:
            print(f"초음파 센서 비활성화: {exc}", flush=True)

    @staticmethod
    def _make_double_beep():
        sample_rate = 24000
        tone_count = int(sample_rate * 0.11)
        silence_count = int(sample_rate * 0.08)
        t = np.arange(tone_count, dtype=np.float32) / sample_rate
        tone = (np.sin(2.0 * np.pi * 900.0 * t) * 0.35 * 32767).astype(np.int16)
        silence = np.zeros(silence_count, dtype=np.int16)
        return tone.tobytes() + silence.tobytes() + tone.tobytes()

    def measure_cm(self):
        if not self.ready or self.gpio is None:
            return None
        gpio = self.gpio
        try:
            gpio.output(self.TRIG_PIN, gpio.LOW)
            time.sleep(0.000002)
            gpio.output(self.TRIG_PIN, gpio.HIGH)
            time.sleep(0.00001)
            gpio.output(self.TRIG_PIN, gpio.LOW)

            deadline = time.monotonic() + 0.03
            while gpio.input(self.ECHO_PIN) == 0:
                if time.monotonic() >= deadline:
                    return self.MAX_DISTANCE_CM
            pulse_start = time.monotonic()

            deadline = pulse_start + 0.03
            while gpio.input(self.ECHO_PIN) == 1:
                if time.monotonic() >= deadline:
                    return self.MAX_DISTANCE_CM
            pulse_end = time.monotonic()
            distance_cm = (pulse_end - pulse_start) * 34300.0 / 2.0
            return min(self.MAX_DISTANCE_CM, max(0.0, distance_cm))
        except Exception as exc:
            print(f"초음파 거리 측정 오류: {exc}", flush=True)
            return None

    def _play_beep(self):
        if not self._aplay or not self._beep_lock.acquire(blocking=False):
            return
        try:
            subprocess.run(
                #[self._aplay, "-q", "-D", self._alsa_device, "-t", "raw",
                # "-f", "S16_LE", "-r", "24000", "-c", "1"],
                [self._aplay, "--raw", "--rate", "16000", "--channels", "1", "--format", "s16", "-"],
                input=self._beep_samples,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
        except Exception as exc:
            print(f"초음파 경고음 출력 오류: {exc}", flush=True)
        finally:
            self._beep_lock.release()

    def update_warning(self, distance_cm, active):
        if not active or distance_cm is None or distance_cm > self.WARNING_DISTANCE_CM:
            return
        now = time.monotonic()
        if now - self._last_beep >= self._beep_interval:
            self._last_beep = now
            threading.Thread(target=self._play_beep, name="ultrasonic-beep", daemon=True).start()

    def close(self):
        if self.gpio is not None and self.ready:
            try:
                self.gpio.cleanup((self.TRIG_PIN, self.ECHO_PIN))
            except Exception:
                pass
            self.ready = False


class ShutdownButton:
    PIN = 17
    LONG_PRESS_SECONDS = 2.0
    DEBOUNCE_SECONDS = 0.08

    def __init__(self, shutdown_requested, paused):
        self.gpio = None
        self.ready = False
        self.shutdown_requested = shutdown_requested
        self.paused = paused
        self.active_level = int(os.getenv("WALKING_ASSISTANT_BUTTON_ACTIVE_LEVEL", "0"))
        self._stop_polling = threading.Event()
        self._poll_thread = None
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            pull = GPIO.PUD_DOWN if self.active_level == 1 else GPIO.PUD_UP
            GPIO.setup(self.PIN, GPIO.IN, pull_up_down=pull)
            self.ready = True
            self._poll_thread = threading.Thread(
                target=self._poll_button, name="shutdown-button", daemon=True
            )
            self._poll_thread.start()
            print(
                f"버튼 감시 시작: GPIO{self.PIN}(물리 11), 짧게 누름=일시정지/재개, "
                f"{self.LONG_PRESS_SECONDS:.0f}초 이상=전원 종료",
                flush=True,
            )
        except Exception as exc:
            self.ready = False
            print(f"버튼 감시 비활성화: {exc}", flush=True)

    def _poll_button(self):
        released_since = None
        armed = False
        pressed_since = None
        long_press_handled = False
        while not self._stop_polling.wait(0.02):
            try:
                level = int(self.gpio.input(self.PIN))
                now = time.monotonic()
                is_pressed = level == self.active_level

                if not armed:
                    if is_pressed:
                        released_since = None
                    else:
                        if released_since is None:
                            released_since = now
                        elif now - released_since >= 0.5:
                            armed = True
                            print("버튼 준비 완료", flush=True)
                    continue

                if is_pressed:
                    if pressed_since is None:
                        pressed_since = now
                        long_press_handled = False
                    elif not long_press_handled and now - pressed_since >= self.LONG_PRESS_SECONDS:
                        long_press_handled = True
                        print("버튼 2초 이상 누름: 프로그램을 정리한 뒤 전원을 종료합니다.", flush=True)
                        self.shutdown_requested.set()
                        return
                    continue

                if pressed_since is not None:
                    press_duration = now - pressed_since
                    if press_duration >= self.DEBOUNCE_SECONDS and not long_press_handled:
                        if self.paused.is_set():
                            self.paused.clear()
                            print("프로그램 재개", flush=True)
                        else:
                            self.paused.set()
                            print("프로그램 일시정지", flush=True)
                    pressed_since = None
                    long_press_handled = False
            except Exception as exc:
                print(f"버튼 입력 확인 오류: {exc}", flush=True)
                return

    def close(self):
        self._stop_polling.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=0.5)
        if self.gpio is not None and self.ready:
            try:
                self.gpio.cleanup(self.PIN)
            except Exception:
                pass
            self.ready = False


def shutdown_system():
    commands = [
        ["systemctl", "poweroff"],
        ["shutdown", "-h", "now"],
        ["/usr/sbin/shutdown", "-h", "now"],
        ["/sbin/shutdown", "-h", "now"],
    ]
    is_root = os.geteuid() == 0

    for cmd in commands:
        full_cmd = cmd if is_root else ["sudo", "-n", *cmd]
        try:
            result = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
                check=False,
            )
            if result.returncode == 0:
                return
            detail = (result.stderr or result.stdout or "").strip()
            print(f"종료 명령 실패 ({' '.join(full_cmd)}): {detail}", flush=True)
        except Exception as exc:
            print(f"종료 명령 예외 ({' '.join(full_cmd)}): {exc}", flush=True)


class Speech:
    def __init__(self):
        self.last_key = None
        self.last_submit_time = 0.0
        self.vehicle_priority = False
        self.ultrasonic_suppressed = False
        self.last_urgent_key = None
        self.last_urgent_submit = 0.0
        self.current_text = None
        self.interrupted_texts = set()
        self.jobs = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.process_lock = threading.Lock()
        self.active_processes = set()
        self.worker = None
        
        #self.aplay = shutil.which("aplay")
        self.aplay = shutil.which("pw-play")
        self.alsa_device = os.getenv("WALKING_ASSISTANT_ALSA_DEVICE", "plughw:2,0")
        self.espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        self.cache_dir = Path(os.getenv(
            "WALKING_ASSISTANT_AUDIO_CACHE",
            str(Path(__file__).resolve().parent / "audio_cache")
        )).expanduser()

        print("음성 출력: 사전 생성된 WAV 캐시 모드 (초기화 지연 0초)", flush=True)

        if self.aplay or self.espeak:
            self.worker = threading.Thread(target=self._audio_worker, name="speech-audio", daemon=True)
            self.worker.start()

    def _cache_path_for_text(self, text):
        cache_key = f"ko_KR-kss-medium|{PIPER_SPEED_FACTOR:.3f}\0{text}"
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    def _play_file(self, path):
        devices = list(dict.fromkeys((self.alsa_device, "default")))
        for device in devices:
            proc = None
            try:
                proc = subprocess.Popen(
                    #[self.aplay, "-q", "-D", device, str(path)],
                    [self.aplay, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                with self.process_lock:
                    self.active_processes.add(proc)
                return_code = proc.wait()
                if return_code == 0:
                    return
            except Exception:
                pass
            finally:
                if proc is not None:
                    with self.process_lock:
                        self.active_processes.discard(proc)

    def _run_espeak(self, text):
        proc = subprocess.Popen(
            [self.espeak, "-v", os.getenv("WALKING_ASSISTANT_ESPEAK_VOICE", "ko"),
             "-s", os.getenv("WALKING_ASSISTANT_VOICE_SPEED", "180"), "--", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self.process_lock:
            self.active_processes.add(proc)
        try:
            proc.wait()
        finally:
            with self.process_lock:
                self.active_processes.discard(proc)

    def _speak(self, text):
        cache_path = self._cache_path_for_text(text)
        if cache_path.is_file() and cache_path.stat().st_size > 44:
            self._play_file(cache_path)
            return
            
        if self.espeak:
            try:
                self._run_espeak(text)
                return
            except Exception:
                pass

    def _audio_worker(self):
        while not self.stopping.is_set():
            try:
                text = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            self.current_text = text
            try:
                self._speak(text)
            finally:
                self.interrupted_texts.discard(text)
                self.current_text = None

    def _queue_audio(self, text, urgent=False):
        if not isinstance(text, str) or not text.strip() or self.stopping.is_set():
            return

        if urgent:
            # 1. 현재 "멈추세요"가 이미 재생 중이거나 큐에 대기 중이면 절대 끊지 않음
            if self.current_text == text:
                return

            # 2. 다른 일반 안내가 재생 중일 때만 프로세스를 종료하고 큐를 비움
            with self.process_lock:
                for proc in tuple(self.active_processes):
                    if proc.poll() is None:
                        proc.terminate()

            while not self.jobs.empty():
                try:
                    self.jobs.get_nowait()
                except queue.Empty:
                    break

            try:
                self.jobs.put_nowait(text)
            except queue.Full:
                pass
            return

        # 일반 안내는 현재 재생 중인 음성이나 대기 큐가 있으면 절대 끼어들지 않음
        if self.current_text is not None or not self.jobs.empty():
            return

        try:
            self.jobs.put_nowait(text)
        except queue.Full:
            pass

    def set_vehicle_priority(self, active):
        self.vehicle_priority = bool(active)

    def set_ultrasonic_suppressed(self, active):
        self.ultrasonic_suppressed = bool(active)

    def say(self, key, text, force=False):
        if not isinstance(text, str) or not text.strip():
            return
        if self.vehicle_priority and not key.startswith(("crosswalk_", "traffic_")):
            return
        if self.ultrasonic_suppressed and not key.startswith(("crosswalk_", "traffic_", "stairs:", "ocr_")):
            return

        # 현재 어떤 음성이든 재생 중이거나 대기 중이면 절대 말을 끊지 않음
        if self.current_text is not None or not self.jobs.empty():
            return

        now = time.time()
        if not force and key == self.last_key and now - self.last_submit_time < 2.5:
            return

        self.last_key = key
        self.last_submit_time = now
        print(f"[결과] {text}", flush=True)
        self._queue_audio(text, urgent=False)

    def say_urgent(self, key, text):
        if not isinstance(text, str) or not text.strip():
            return

        # 이미 "멈추세요"가 재생 중이면 스스로를 끊지 않고 유지
        if self.current_text == text:
            return

        now = time.monotonic()
        repeat_window = 2.0 if key == "traffic_stop" else 1.5
        if key == self.last_urgent_key and now - self.last_urgent_submit < repeat_window:
            return

        self.last_urgent_key = key
        self.last_urgent_submit = now
        self.last_key = key
        self.last_submit_time = time.time()
        print(f"[긴급 결과] {text}", flush=True)
        self._queue_audio(text, urgent=True)

    def wait_until_idle(self, timeout=8.0):
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.current_text is None and self.jobs.empty():
                return True
            time.sleep(0.05)
        return False

    def cancel(self):
        with self.process_lock:
            for proc in tuple(self.active_processes):
                if proc.poll() is None:
                    proc.terminate()
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.stopping.set()
        self.cancel()
        if self.worker is not None:
            self.worker.join(timeout=1.0)


def label_of(result, cls_id):
    names = result.names
    raw = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, "unknown")
    normalized = str(raw).strip().lower()
    if "stair" in normalized or "staircase" in normalized or "stairway" in normalized:
        return "stairs"
    return raw


def zone(x1, x2, width):
    center = (x1 + x2) / 2
    if center < width * 0.25:
        return "왼쪽"
    if center > width * 0.75:
        return "오른쪽"
    return "정면"


def roi(width, height):
    return int(width * 0.08), 0, int(width * 0.92), height


def walking_candidate(det, width, height):
    _, _, x1, y1, x2, y2 = det
    rx1, ry1, rx2, ry2 = roi(width, height)
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)


def is_walk_obstacle(det):
    label = det[0]
    return label not in CROSSWALK_LABELS


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
    _, _, x1, y1, x2, y2 = det
    bottom = y2 / max(1, height)
    area = ((x2 - x1) * (y2 - y1)) / max(1, width * height)
    center = abs(((x1 + x2) / 2) - width / 2) / max(1, width)
    if det[0] in CROSSWALK_LABELS:
        return 0.45 * bottom + 0.30 + 0.25 * (1.0 - center)
    if det[0] in DOOR_COMPONENT_LABELS:
        return 0.70 * bottom + 0.20 + 0.10 * (1.0 - center)
    return 0.55 * bottom + 0.35 * min(area * 4.0, 1.0) + 0.10 * (1.0 - center)


def detection_priority(det, width, height):
    return (-proximity_score(det, width, height), -det[1])


def deduplicate_vehicle_detections(detections):
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


def box_overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a[2:6]
    bx1, by1, bx2, by2 = b[2:6]
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def valid_geometry(det, width, height):
    label, _, x1, y1, x2, y2 = det
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / max(1, width * height)
    aspect = bw / bh
    y2_ratio = y2 / max(1, height)

    if label == "desk":
        return area_ratio >= 0.1 and bw >= width * 0.3 and bh >= height * 0.2

    if label == "chair":
        return area_ratio >= 0.015 and bh >= height * 0.12 and 0.35 <= aspect <= 2.2

    if label == "bicycle":
        return area_ratio >= 0.02 and bw >= width * 0.15 and 0.60 <= aspect <= 3.0

    if label == DOOR_BUTTON_PROMPT:
        return bw >= 3 and bh >= 3 and area_ratio <= 0.015 and 0.4 <= aspect <= 2.5

    if label == ROUND_KNOB_PROMPT:
        return (
            bw >= 8 and bh >= 8
            and 0.0003 <= area_ratio <= 0.03
            and 0.65 <= aspect <= 1.5
        )

    if label in STEEL_DOOR_LABELS:
        return (
            area_ratio >= 0.12
            and bh >= height * 0.35
            and 0.25 <= aspect <= 1.8
        )

    if label in DOOR_COMPONENT_LABELS or label in DOOR_LOCK_LABELS or label in LOCK_TYPE_LABELS:
        return bw >= 3 and bh >= 3 and 0.0001 <= area_ratio <= 0.08 and 0.15 <= aspect <= 7.5

    if label in {"bag", "backpack"}:
        return 0.008 <= area_ratio <= 0.35 and bh >= height * 0.10 and 0.35 <= aspect <= 2.0

    if label in {SCOOTER_PROMPT, PLAIN_SCOOTER_PROMPT, "scooter", "kick scooter", "electric scooter"}:
        return (
            area_ratio >= 0.002 
            and bh >= height * 0.08
            and 0.15 <= aspect <= 1.1
        )
    if label in TOY_CAR_PROMPTS:
        return (
            bw >= 12 and bh >= 10
            and 0.0005 <= area_ratio <= 0.18
            and 0.7 <= aspect <= 3.5
        )
    if label in {CAR_DISTANT_PROMPT, CAR_FAR_PROMPT}:
        return (bw >= 10 and bh >= 8 and 0.0002 <= area_ratio <= 0.10
                and 0.5 <= aspect <= 3.5)
    if label in VEHICLE_LABELS:
        return (
            bw >= 16 and bh >= 12
            and 0.001 <= area_ratio <= 0.25
            and 0.8 <= aspect <= 4.0
        )
    if label in {"shelf", "bookshelf", "storage shelf", "freestanding storage shelf"}:
        return (area_ratio >= 0.012 and bw >= width * 0.12
                and bh >= height * 0.10 and 0.20 <= aspect <= 4.0)
    if label in {"wardrobe", "closet"}:
        return (area_ratio >= 0.01 and bw >= width * 0.08
                and bh >= height * 0.15 and 0.25 <= aspect <= 3.5)
    if label == "person":
        return (area_ratio >= 0.0008 and bh >= height * 0.035
                and 0.12 <= aspect <= 2.2)
    if label in BED_LABELS:
        return (area_ratio >= 0.035 and area_ratio <= 0.65
                and bw >= width * 0.25 and bh >= height * 0.15
                and y2 >= height * 0.75 and y1 >= height * 0.08
                and 0.75 <= aspect <= 4.0)
    if label in CROSSWALK_LABELS:
        cx = (x1 + x2) / 2.0
        if cx < width * 0.18 or cx > width * 0.82:
            return False
        return (
            0.015 <= area_ratio <= 0.65
            and bw >= width * 0.15
            and bh >= height * 0.05
            and aspect >= 0.6
            and y2_ratio >= 0.35
        )
    if label in GLASS_DOOR_LABELS or label in GLASS_DOOR_TYPE_LABELS:
        return area_ratio >= 0.04 and bh >= height * 0.22 and 0.20 <= aspect <= 3.5
    if label in DOOR_LABELS:
        return area_ratio >= 0.00005 and bh >= height * 0.010 and aspect <= 14.0
    if label == "elevator":
        return area_ratio >= ELEVATOR_MIN_AREA_RATIO and bw >= width * ELEVATOR_MIN_WIDTH_RATIO and bh >= height * 0.20
    if label in ELEVATOR_SUPPORT_LABELS:
        return area_ratio >= 0.00015 and area_ratio <= 0.08
    if label in STAIR_LABELS:
        return (area_ratio >= 0.003 or bw >= width * 0.05
                or bh >= height * 0.04)
    return area_ratio >= 0.002


def prepare_inference_frame(frame):
    if not AUTO_CLAHE:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) >= 115.0 and float(np.std(gray)) >= 35.0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def guidance_for_detection(det, width, all_dets=None):
    label, _, x1, y1, x2, y2 = det
    pos = zone(x1, x2, width)

    # 1. 미닫이문 직통 판별 (유리문/자동문 판별 조건 완벽 차단)
    if label in SLIDING_DOOR_LABELS or "sliding" in label.lower():
        return f"{pos} 미닫이문, 옆으로 미세요"

    # 2. 기타 문 구조체 판별
    if label in DOOR_LABELS or label in STEEL_DOOR_LABELS or label in GLASS_DOOR_LABELS or label in GLASS_DOOR_TYPE_LABELS:
        if "glass" in label.lower() or label in GLASS_DOOR_LABELS or label in GLASS_DOOR_TYPE_LABELS:
            door_name = "유리문"
        elif label == "wooden door":
            door_name = "나무문"
        elif label in STEEL_DOOR_LABELS:
            door_name = "철문"
        else:
            door_name = "문"

        nearby_handle = None
        has_button = False

        if all_dets:
            for other in all_dets:
                olabel = other[0]
                if olabel in DOOR_COMPONENT_LABELS:
                    ox1, oy1, ox2, oy2 = other[2:6]
                    ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
                    if (x1 - 80) <= ocx <= (x2 + 80) and (y1 - 80) <= ocy <= (y2 + 80):
                        if olabel in HANDLE_TYPE_LABELS:
                            nearby_handle = olabel
                        elif olabel == DOOR_BUTTON_PROMPT:
                            has_button = True

        if nearby_handle:
            if nearby_handle in LEVER_HANDLE_PROMPTS:
                return f"{pos} {door_name}, 레버 손잡이를 내리세요"
            elif nearby_handle == ROUND_KNOB_PROMPT:
                return f"{pos} {door_name}, 문고리를 돌리세요"
            elif nearby_handle == PUSH_BAR_PROMPT:
                return f"{pos} {door_name}, 푸시바를 밀으세요"
            elif nearby_handle == PULL_BAR_PROMPT:
                return f"{pos} {door_name}, 손잡이 잡고 당기세요"

        if has_button or label in {"automatic door", "automatic glass door"}:
            if has_button:
                return f"{pos} 자동문, 버튼을 누르세요"
            return f"{pos} 자동문"

        return f"{pos} {door_name}"

    # 3. 손잡이 단독 감지 (검증 통과 건만)
    if label in HANDLE_TYPE_LABELS:
        if label in LEVER_HANDLE_PROMPTS:
            return f"{pos} 레버 손잡이, 내리세요"
        elif label == ROUND_KNOB_PROMPT:
            return f"{pos} 원형 문고리"
        elif label == PUSH_BAR_PROMPT:
            return f"{pos} 푸시바, 밀으세요"
        elif label == PULL_BAR_PROMPT:
            return f"{pos} 문손잡이"
        return f"{pos} 문손잡이"

    # 4. 기타 장애물
    if label in STAIR_LABELS:
        return f"{pos} 계단"
    if label == "elevator":
        return f"{pos} 엘리베이터"
    if label in CROSSWALK_LABELS:
        return f"{pos} 횡단보도"
    if label in WALL_SURFACE_LABELS:
        return ""

    korean_name = KOREAN_LABELS.get(label)
    if korean_name:
        return f"{pos} {korean_name}"
    return ""

class CrosswalkFrameHold:
    def __init__(self):
        self.detections = []
        self.last_seen = None
        self.streak = 0

    def reset(self):
        self.detections = []
        self.last_seen = None
        self.streak = 0

    @staticmethod
    def _has_zebra_contrast(frame, det):
        x1, y1, x2, y2 = det[2:6]
        roi = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_ratio = np.mean(gray > 160)
        dark_ratio = np.mean(gray < 115)
        return (0.10 <= bright_ratio <= 0.80) and (dark_ratio >= 0.10)

    def update(self, frame, detections, has_vehicles=False, now=None):
        now = time.monotonic() if now is None else now
        valid_dets = [d for d in detections if self._has_zebra_contrast(frame, d)]

        if valid_dets:
            self.streak += 1
            # 이미 횡단보도 유지 중이면 1프레임만 보여도 즉시 시간 연장
            if self.last_seen is not None or self.streak >= 2:
                self.detections = list(valid_dets)
                self.last_seen = now
                return list(self.detections), True
            return [], False

        self.streak = 0
        # 차량이 횡단보도 위를 가리고 있는 동안에는 모드가 절대 끊기지 않도록 5초간 유지
        if self.last_seen is not None and has_vehicles and (now - self.last_seen <= 5.0):
            return list(self.detections), False

        # 차량도 없고 횡단보도도 2.0초 이상 완전히 안 보일 때만 모드 종료
        if self.last_seen is not None and now - self.last_seen <= 2.0:
            return list(self.detections), False

        self.reset()
        return [], False

class SituationMachine:
    def __init__(self):
        self.state = "WALK"
        self.forced = False
        self.last_crosswalk_time = None

    def update(self, detections, crosswalk_held_dets, width, has_vehicles=False, has_masked_zones=False):
        now = time.monotonic()
        all_cw = list(crosswalk_held_dets) + [d for d in detections if d[0] in CROSSWALK_LABELS]
        has_front_crosswalk = any(zone(d[2], d[4], width) == "정면" for d in all_cw)

        if has_front_crosswalk:
            self.last_crosswalk_time = now
            self.state = "CROSSWALK"
            return self.state

        # 정지 차량 때문에 프레임 일부를 마스킹했거나 차량이 지나가는 중이면 횡단보도 모드 유지
        if self.state == "CROSSWALK" and (has_vehicles or has_masked_zones) and self.last_crosswalk_time is not None:
            if now - self.last_crosswalk_time <= 8.0:
                return self.state

        self.state = "WALK"
        self.last_crosswalk_time = None
        return self.state

class CrosswalkSafety:
    STATIONARY_SECONDS = 1.5      # 1.5초간 정지 유지 시 해당 프레임 영역 마스킹
    PENDING_TOLERANCE = 0.08      # 손떨림 및 박스 지터 완화 허용 오차 (8%)
    MASK_PADDING = 0.015          # 마스킹 테두리 여유분 (1.5%)
    MAX_MASKS = 2                 # 화면 내 동시 정지 마스크 최대 2개로 제한 (인식 영역 잠식 방지)

    def __init__(self):
        self.status = "CLEAR"
        self.masked_rects = []    # [(x1, y1, x2, y2), ...]
        self.pending_frame = None

    def reset(self):
        self.status = "CLEAR"
        self.masked_rects.clear()
        self.pending_frame = None

    def apply_frame_mask(self, frame):
        """YOLO 추론 전에 정지 확정된 프레임 영역을 검은색으로 가림"""
        if not self.masked_rects:
            return frame

        masked = frame.copy()
        h, w = masked.shape[:2]
        for rx1, ry1, rx2, ry2 in self.masked_rects:
            px1 = max(0, int((rx1 - self.MASK_PADDING) * w))
            py1 = max(0, int((ry1 - self.MASK_PADDING) * h))
            px2 = min(w, int((rx2 + self.MASK_PADDING) * w))
            py2 = min(h, int((ry2 + self.MASK_PADDING) * h))
            masked[py1:py2, px1:px2] = 0
        return masked

    @staticmethod
    def _norm_rect(det, w, h):
        _, _, x1, y1, x2, y2 = det
        return (
            x1 / max(1, w),
            y1 / max(1, h),
            x2 / max(1, w),
            y2 / max(1, h),
        )

    @staticmethod
    def _rect_iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _is_same_frame_region(rect_a, rect_b, tol):
        acx, acy = (rect_a[0] + rect_a[2]) / 2.0, (rect_a[1] + rect_a[3]) / 2.0
        bcx, bcy = (rect_b[0] + rect_b[2]) / 2.0, (rect_b[1] + rect_b[3]) / 2.0
        center_diff = ((bcx - acx) ** 2 + (bcy - acy) ** 2) ** 0.5
        # 박스 크기는 YOLO 지터가 심하므로 중심점 이동 거리만 엄격히 체크
        return center_diff <= tol

    def _register_mask(self, new_rect):
        """새 정지 마스크 등록 시 겹치는 기존 마스크는 교체하고 최대 2개까지만 유지"""
        self.masked_rects = [
            r for r in self.masked_rects
            if self._rect_iou(r, new_rect) < 0.10
        ]
        if len(self.masked_rects) >= self.MAX_MASKS:
            self.masked_rects.pop(0)
        self.masked_rects.append(new_rect)

    def update(self, frame, vehicles):
        now = time.monotonic()
        h, w = frame.shape[:2]

        if not vehicles:
            self.pending_frame = None
            self.status = "CLEAR"
            return "CLEAR", False

        target_det = max(vehicles, key=lambda d: (d[4] - d[2]) * (d[5] - d[3]))
        curr_rect = self._norm_rect(target_det, w, h)

        if self.pending_frame is None or not self._is_same_frame_region(
            self.pending_frame["rect"], curr_rect, self.PENDING_TOLERANCE
        ):
            self.pending_frame = {
                "rect": curr_rect,
                "start_time": now,
            }
            self.status = "WAIT"
            return "WAIT", True

        if now - self.pending_frame["start_time"] >= self.STATIONARY_SECONDS:
            self._register_mask(self.pending_frame["rect"])
            self.pending_frame = None
            self.status = "CLEAR"
            return "CLEAR", False

        self.status = "WAIT"
        return "WAIT", False

class StairConfirm:
    def __init__(self):
        self.last = None
        self.count = 0

    def reset(self):
        self.last = None
        self.count = 0

    def is_complete(self, det, width, height, crosswalk_dets=None):
        _, _, x1, y1, x2, y2 = det
        if crosswalk_dets:
            for cw in crosswalk_dets:
                if box_iou(det, cw) > 0.15:
                    return False
        box_area = max(1, (x2 - x1) * (y2 - y1))
        frame_area = max(1, width * height)
        width_ratio = (x2 - x1) / max(1, width)
        height_ratio = (y2 - y1) / max(1, height)
        aspect = (x2 - x1) / max(1, y2 - y1)
        
        overlaps_walk = (
            x2 >= width * 0.02
            and x1 <= width * 0.98
            and y2 >= height * 0.35
        )
        large_enough = (
            box_area / frame_area >= 0.0025
            or width_ratio >= 0.035
        )
        plausible_shape = 0.20 <= aspect <= 9.0 and height_ratio >= 0.25
        return overlaps_walk and large_enough and plausible_shape

    def update(self, detections, width, height, crosswalk_dets=None):
        candidates = [
            d for d in detections
            if d[0] in STAIR_LABELS and self.is_complete(d, width, height, crosswalk_dets=crosswalk_dets)
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
    def __init__(self):
        self.previous = []

    def reset(self):
        self.previous = []

    @staticmethod
    def is_inside_or_near_door(component, doors, max_pixel_margin=70):
        if not doors:
            if component[0] in LEVER_HANDLE_PROMPTS:
                return component[1] >= 0.30
            if component[0] == ROUND_KNOB_PROMPT:
                return component[1] >= 0.78
            return component[1] >= 0.50
            
        cx, cy = (component[2] + component[4]) / 2.0, (component[3] + component[5]) / 2.0
        
        for door in doors:
            if component[0] == ROUND_KNOB_PROMPT and door[1] < 0.25:
                continue
            dx1, dy1, dx2, dy2 = door[2:6]
            if (dx1 - max_pixel_margin) <= cx <= (dx2 + max_pixel_margin) and \
               (dy1 - max_pixel_margin) <= cy <= (dy2 + max_pixel_margin):
                return True
        return False

    def update(self, detections, confirmed_doors):
        candidates = []
        raw_components = [
            d for d in detections
            if d[0] in DOOR_COMPONENT_LABELS
            and d[1] >= CLASS_MIN_CONF.get(d[0], 0.38)
        ]
        
        all_doors = [d for d in detections if d[0] in DOOR_LABELS] + confirmed_doors

        for comp in raw_components:
            if self.is_inside_or_near_door(comp, all_doors):
                candidates.append(comp)

        self.previous = candidates
        return candidates


class DoorConfirm:
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
        if box_iou(a, b) >= 0.03:
            return True
        acx, acy = (a[2]+a[4])/2, (a[3]+a[5])/2
        bcx, bcy = (b[2]+b[4])/2, (b[3]+b[5])/2
        scale = max(a[4]-a[2], a[5]-a[3], b[4]-b[2], b[5]-b[3], 1)
        return ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5 <= scale * 0.35

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

        if self.count >= 1:
            self.confirmed = door
            return [door]
        return []


class TargetSelector:
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

        if self.pending_count >= 1:
            self.current = candidate
            self.pending = None
            self.pending_count = 0

        return self.current


class LatestFrameCapture:
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
        with self.condition:
            previous_sequence = getattr(self, "_last_read_sequence", 0)
            self.condition.wait_for(
                lambda: not self.running or self.sequence > previous_sequence,
                timeout=2.0,
            )
            if self.sequence <= previous_sequence or self.latest_frame is None:
                return False, None
            self._last_read_sequence = self.sequence
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        self.capture.release()
        self.reader.join(timeout=2.0)


def sync_cuda_for_profile(model):
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
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(max(1, int(os.getenv("WALKING_ASSISTANT_CPU_THREADS", "3"))))
    except Exception as exc:
        print(f"PyTorch CPU 스레드 설정을 건너뜁니다: {exc}")
    
    model_load_started = time.monotonic()
    model = YOLOE(MODEL_PATH)

    prompt_embeddings_path = Path("/home/pineapple/pi/prompt_embeddings.npz")
    embeddings_loaded = False
    if prompt_embeddings_path.is_file():
        try:
            model.load_prompt_embeddings(str(prompt_embeddings_path))
            embeddings_loaded = True
            print(f"저장된 YOLOE 프롬프트 임베딩 사용: {prompt_embeddings_path}", flush=True)
        except Exception as exc:
            print(f"저장된 프롬프트 임베딩 로드 실패: {exc}", flush=True)

    if not embeddings_loaded:
        model.set_classes(PROMPTS)
        try:
            model.save_prompt_embeddings(str(prompt_embeddings_path))
            print(f"YOLOE 프롬프트 임베딩 저장 완료: {prompt_embeddings_path}", flush=True)
        except Exception as exc:
            print(f"프롬프트 임베딩 저장 생략: {exc}", flush=True)

    warmup_started = time.monotonic()
    try:
        warmup = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        model.predict(warmup, conf=YOLO_INFERENCE_CONFIDENCE, imgsz=INFERENCE_SIZE, max_det=1, verbose=False)
        print(f"추론 워밍업 완료: {time.monotonic() - warmup_started:.2f}초", flush=True)
    except Exception as exc:
        print(f"추론 워밍업 생략: {exc}", flush=True)

    print(f"모델 준비 완료: {time.monotonic() - model_load_started:.1f}초", flush=True)

    raw_cap = cv2.VideoCapture(CAMERA_INDEX)
    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("WALKING_ASSISTANT_CAMERA_WIDTH", "640")))
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("WALKING_ASSISTANT_CAMERA_HEIGHT", "480")))
    raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not raw_cap.isOpened():
        raw_cap.release()
        raise RuntimeError("웹캠을 열 수 없습니다.")

    cap = LatestFrameCapture(raw_cap)
    speech = Speech()
    ocr_reader = AsyncOCRReader(speech)
    ultrasonic = UltrasonicSensor()
    shutdown_requested = threading.Event()
    program_paused = threading.Event()
    shutdown_button = ShutdownButton(shutdown_requested, program_paused)
    
    machine = SituationMachine()
    crosswalk_frame_hold = CrosswalkFrameHold()
    crosswalk_safety = CrosswalkSafety()
    target_selector = TargetSelector()
    door_confirm = DoorConfirm()
    door_component_confirm = DoorComponentConfirm()
    stair_confirm = StairConfirm()
    elevator_confirm = ElevatorConfirm()

    last_guidance_key = None
    last_traffic_status = None
    last_crosswalk_clear_announce = 0.0
    last_crosswalk_wait_time = None
    last_reported_state = "WALK"
    pause_cleanup_done = False
    last_stop_announce_time = 0.0
    STOP_TO_CROSS_GAP_SECONDS = 2.0

    try:
        while not shutdown_requested.is_set():
            if program_paused.is_set():
                if not pause_cleanup_done:
                    speech.cancel()
                    crosswalk_safety.reset()
                    crosswalk_frame_hold.reset()
                    target_selector.reset()
                    pause_cleanup_done = True
                time.sleep(0.05)
                continue
            if pause_cleanup_done:
                last_guidance_key = None
                last_traffic_status = None
                pause_cleanup_done = False

            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            inference_frame = prepare_inference_frame(frame)

            # [핵심] 횡단보도 모드일 때, 이미 정지 확정된 프레임 영역은 검은색으로 가려서 모델 인식 범위에서 아예 제외
            if machine.state == "CROSSWALK":
                inference_frame = crosswalk_safety.apply_frame_mask(inference_frame)

            sync_cuda_for_profile(model)

            max_dets = 12 if machine.state == "CROSSWALK" else int(os.getenv("WALKING_ASSISTANT_MAX_DET", "20"))
            result = model.predict(
                inference_frame,
                conf=YOLO_INFERENCE_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=INFERENCE_SIZE,
                max_det=max_dets,
                verbose=False,
            )[0]

            all_dets = []
            crosswalk_vehicles = []
            crosswalk_dets = []
            ocr_triggered = False

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = label_of(result, cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                aspect = bw / bh

                if label in {*LEVER_HANDLE_PROMPTS, PULL_BAR_PROMPT} and aspect >= 0.95 and (bw * bh) / (w * h) <= 0.04:
                    label = "black plastic lever door handle"

                if label in {"refrigerator", "refrigerator in a kitchen", "large household refrigerator"}:
                    if bh >= h * 0.70:
                        label = "steel door"

                if label in {"scooter", "kick scooter", SCOOTER_PROMPT, PLAIN_SCOOTER_PROMPT}:
                    if aspect >= 1.0 or bh < h * 0.08:
                        label = "toy car"
                        conf = max(conf, 0.20)

                det = (label, conf, x1, y1, x2, y2)

                if label in TEXT_SIGN_PROMPTS and conf >= 0.12:
                    if bw >= 25 and bh >= 15:
                        pad_x = max(15, int(bw * 0.15))
                        pad_y = max(15, int(bh * 0.15))
                        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                        crop_img = frame[ry1:ry2, rx1:rx2]
                        if crop_img.shape[0] > 15 and crop_img.shape[1] > 15:
                            ocr_reader.request_ocr(crop_img)
                            ocr_triggered = True
                elif label in DOOR_LABELS and conf >= 0.20:
                    rx1 = max(0, x1 + int(bw * 0.10))
                    rx2 = min(w, x2 - int(bw * 0.10))
                    ry1 = max(0, y1 + int(bh * 0.15))
                    ry2 = min(h, y2 - int(bh * 0.15))
                    crop_img = frame[ry1:ry2, rx1:rx2]
                    if crop_img.shape[0] > 20 and crop_img.shape[1] > 20:
                        ocr_reader.request_ocr(crop_img)
                        ocr_triggered = True
                
                if conf >= class_min_conf(label) and valid_geometry(det, w, h):
                    all_dets.append(det)
                    if label in VEHICLE_LABELS and conf >= CROSSWALK_VEHICLE_MIN_CONF:
                        crosswalk_vehicles.append(det)
                    elif label in CROSSWALK_LABELS and conf >= CROSSWALK_MIN_CONF:
                        crosswalk_dets.append(det)

            if not ocr_triggered and machine.state == "WALK":
                cx1, cy1 = int(w * 0.25), int(h * 0.25)
                cx2, cy2 = int(w * 0.75), int(h * 0.75)
                ocr_reader.request_ocr(frame[cy1:cy2, cx1:cx2])

            crosswalk_vehicles = deduplicate_vehicle_detections(crosswalk_vehicles)
            all_dets = deduplicate_detections(all_dets)
            rect_obstacles = [
                d for d in all_dets
                if d[0] in {"box", "desk", "shelf", "bookshelf", "wardrobe", "closet", "refrigerator", "bed with mattress and pillow"}
            ]
            if rect_obstacles:
                all_dets = [
                    d for d in all_dets
                    if not (d[0] in VEHICLE_LABELS and any(box_overlap_ratio(d, r) >= 0.35 for r in rect_obstacles))
                ]
                crosswalk_vehicles = [
                    d for d in crosswalk_vehicles
                    if not any(box_overlap_ratio(d, r) >= 0.35 for r in rect_obstacles)
                ]

            # 횡단보도가 잡혔거나 차량이 함께 있는 경우, 겹치는 'desk' 오인식 제거
            if crosswalk_dets or crosswalk_vehicles:
                all_dets = [
                    d for d in all_dets
                    if d[0] != "desk"
                ]

            door_candidates_for_elevator = [d for d in all_dets if d[0] in DOOR_LABELS and d[1] >= DOOR_MIN_CONF]
            confirmed_elevators = elevator_confirm.update(all_dets, w, h, door_candidates_for_elevator)

            raw_doors = [d for d in all_dets if d[0] in DOOR_LABELS and d[1] >= DOOR_MIN_CONF]
            confirmed_doors = door_confirm.update(raw_doors)
            valid_components = door_component_confirm.update(all_dets, confirmed_doors)

            confirmed_stairs = stair_confirm.update(all_dets, w, h, crosswalk_dets=crosswalk_dets)

            filtered_for_guidance = [
                d for d in all_dets
                if ((d[0] not in STAIR_LABELS or d in confirmed_stairs) and (d[0] not in ELEVATOR_LABELS or d in confirmed_elevators))
            ]

            active = [d for d in filtered_for_guidance if walking_candidate(d, w, h) and valid_geometry(d, w, h)]

            crosswalk_dets, crosswalk_is_live = crosswalk_frame_hold.update(
                frame, crosswalk_dets, has_vehicles=(len(crosswalk_vehicles) > 0)
            )
            state = machine.update(
                all_dets,
                crosswalk_dets,
                w,
                has_vehicles=(len(crosswalk_vehicles) > 0),
                has_masked_zones=(len(crosswalk_safety.masked_rects) > 0),
            )

            general_mode = state != "CROSSWALK"
            ultrasonic_distance_cm = ultrasonic.measure_cm() if general_mode else None
            speech.set_ultrasonic_suppressed(
                general_mode and ultrasonic_distance_cm is not None and ultrasonic_distance_cm >= UltrasonicSensor.MAX_DISTANCE_CM
            )
            ultrasonic.update_warning(ultrasonic_distance_cm, active=general_mode)

            if state == "CROSSWALK":
                traffic_status, urgent_stop = crosswalk_safety.update(frame, crosswalk_vehicles)
                speech.set_vehicle_priority(traffic_status == "WAIT")

                if traffic_status == "WAIT":
                    # 차량 감지/이동 중에는 계속 타이머를 리셋하여 성급한 "건너세요" 차단
                    last_stop_announce_time = time.monotonic()

                if last_reported_state != "CROSSWALK":
                    if traffic_status == "WAIT":
                        speech.say_urgent("traffic_stop", "멈추세요")
                        last_guidance_key = "traffic:WAIT"
                    else:
                        speech.say("crosswalk_front", "정면 횡단보도", force=False)
                        last_crosswalk_clear_announce = 0.0
                else:
                    if urgent_stop or (traffic_status == "WAIT" and last_traffic_status != "WAIT"):
                        speech.say_urgent("traffic_stop", "멈추세요")
                        last_guidance_key = "traffic:WAIT"
                    elif traffic_status == "CLEAR":
                        now = time.monotonic()
                        # "멈추세요" 이후 2.5초간 안전(CLEAR) 상태가 유지된 경우에만 "건너세요" 출력
                        if (now - last_stop_announce_time >= STOP_TO_CROSS_GAP_SECONDS) and (
                            last_crosswalk_clear_announce == 0.0
                            or now - last_crosswalk_clear_announce >= CROSSWALK_REPEAT_SECONDS
                        ):
                            speech.say("crosswalk_clear", "건너세요", force=False)
                            last_crosswalk_clear_announce = now
                        last_guidance_key = "traffic:CLEAR"
            else:
                # 횡단보도 모드가 완전히 끝났을 때만 봉인된 정지 차량 프레임 구역 초기화
                crosswalk_safety.reset()
                traffic_status = "CLEAR"
                speech.set_vehicle_priority(False)

                side_crosswalks = [cw for cw in crosswalk_dets if zone(cw[2], cw[4], w) != "정면"]
                if side_crosswalks:
                    nearest_cw = max(side_crosswalks, key=lambda d: proximity_score(d, w, h))
                    cw_pos = zone(nearest_cw[2], nearest_cw[4], w)
                    speech.say(f"crosswalk:{cw_pos}", f"{cw_pos} 횡단보도")

                obstacle_candidates = [
                    d for d in active
                    if is_walk_obstacle(d)
                    and not any(box_overlap_ratio(d, cw) >= 0.25 for cw in crosswalk_dets)
                    and (d[0] not in DOOR_LABELS or d in confirmed_doors)
                    and (d[0] not in DOOR_COMPONENT_LABELS or d in valid_components)
                ]
                primary = target_selector.update(obstacle_candidates, w, h)

                if primary is not None:
                    position = zone(primary[2], primary[4], w)
                    text = guidance_for_detection(primary, w, all_dets=all_dets)
                    key_label = "vehicle" if primary[0] in VEHICLE_LABELS else primary[0]
                    guidance_key = f"obstacle:{key_label}:{position}"
                    if text:
                        speech.say(guidance_key, text, force=(guidance_key != last_guidance_key))
                        last_guidance_key = guidance_key

            last_traffic_status = traffic_status
            if state != last_reported_state:
                print(f"상황 모드: {state}", flush=True)
                last_reported_state = state

    finally:
        ocr_reader.close()
        shutdown_button.close()
        ultrasonic.close()
        speech.close()
        if shutdown_requested.is_set():
            shutdown_system()

        cap.release()

if __name__ == "__main__":
    main()
