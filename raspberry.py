import os
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
CROSSWALK_MIN_CONF = 0.20
CROSSWALK_ENTER_CONFIRM_SECONDS = 0.4
CROSSWALK_EXIT_HOLD_SECONDS = 3.0
CROSSWALK_DETECTION_GAP_SECONDS = 2.0

VEHICLE_FRAME_HOLD_SECONDS = 0.5
VEHICLE_ABSENCE_CLEAR_SECONDS = VEHICLE_FRAME_HOLD_SECONDS
VEHICLE_STATIONARY_SECONDS = 0.7
VEHICLE_CENTER_TOLERANCE = 0.035
VEHICLE_ASSOCIATION_GAP = 0.22
CROSSWALK_ZONE_COUNT = 4
CROSSWALK_REPEAT_SECONDS = 3.0
CROSSWALK_STOP_TO_CROSS_DELAY_SECONDS = 1.5
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
]
CROSSWALK_LABELS = set(CROSSWALK_PROMPTS)

STAIR_PROMPTS = [
    "indoor architectural staircase with steps leading up or down",
    "outdoor concrete flight of stairs with handrail",
    "building entrance staircase with vertical risers",
    "descending or ascending staircase structure",
]

# 문 스티커 / 표지판 프롬프트 (미세요, 당기세요, 숫자 타겟)
TEXT_SIGN_PROMPTS = [
    "push or pull sign sticker on a door",
    "room number sign plate with numbers",
    "door sign sticker with text",
]

CAR_PROMPT = "real passenger car or automobile"
SCOOTER_PROMPT = "electric kick scooter on a road"
MOTORCYCLE_PROMPT = "motorcycle on a road"
PLAIN_SCOOTER_PROMPT = "scooter on a road"
CAR_DISTANT_PROMPT = "distant passenger car"
CAR_FAR_PROMPT = "far away vehicle"
# TOY_CAR_PROMPT = "toy car, miniature vehicle model, or diecast car"
# TOY_VEHICLE_PROMPT = "small toy car on a floor or table"
TOY_CAR_PROMPT = "miniature toy car or plastic model vehicle on floor"
TOY_VEHICLE_PROMPT = "small toy vehicle model"
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
RAILING_LABELS = {"railing", "guardrail"}
GLASS_DOOR_LABELS = {
    "glass door", "transparent glass door", "clear glass door",
    "glass entrance door", "large glass entrance door",
    "commercial glass entrance door", "glass storefront door",
    "glass panel door", "framed glass door", "glass door with metal frame",
    "transparent sliding glass door", "sliding glass entrance door",
    "front view of a glass door", "glass door at a building entrance",
    "automatic glass door", "glass entrance door with visible handle",
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
# WINDOW_LABELS = {"window", "glass window", "building window", "exterior window"}
STEEL_DOOR_LABELS = {
    "steel door", "metal door", "metal entrance door", "metal security door",
    "steel entrance door with visible handle",
    "metal security door in building entrance",
    "closed metal entrance door", "metal door with a handle",
}
GLASS_DOOR_TYPE_LABELS = {
    "automatic glass sliding door", "hinged glass door with a frame",
    "glass hinged entrance door", "manual sliding glass door",
    "glass door with a vertical pull handle",
}
HANDLE_TYPE_LABELS = {
    "round spherical metal doorknob mounted on a door",
    "horizontal lever door handle attached to a door",
    "recessed handle flush with a sliding door surface",
    "long vertical pull handle bar on a commercial glass door",
    "horizontal panic push bar across a exit door",
}
LOCK_TYPE_LABELS = {
    "digital keypad smart door lock with numbers",
    "electronic door lock mounted on a door panel",
}
DOOR_HANDLE_PROMPTS = [
    "straight horizontal lever door handle attached to a door",  # 레버 손잡이
    "round ball-shaped spherical metal doorknob on a door",      # 원형 문고리
    "digital keypad smart door lock on a door panel",            # 도어락
    "long vertical pull handle bar on a door",                    # 바 손잡이
]

CLASS_MIN_CONF = {
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
    # "window": 0.38,
    # "glass window": 0.40,
    # "building window": 0.40,
    # "exterior window": 0.42,
    "box": 0.55,
    "desk": 0.40,
    "chair": 0.48,
    "door": 0.22,
    "wooden door": 0.35,
    "steel door": 0.55,
    "glass entrance door": 0.30,
    "sliding glass door": 0.20,
    "automatic glass door": 0.30,
    "door handle": 0.35,
    "doorknob": 0.30,
    "round spherical metal doorknob mounted on a door": 0.33,
    "lever door handle": 0.25,
    "horizontal lever door handle attached to a door": 0.25,
    "straight horizontal lever door handle attached to a door": 0.25,
    "door knob": 0.30,
    "push bar": 0.50,
    "door push button": 0.50,
    "digital keypad smart door lock on a door panel": 0.25,
    "electronic keypad door lock": 0.25,
    "door lock": 0.30,
    CAR_PROMPT: 0.30,
    CAR_DISTANT_PROMPT: 0.15,
    CAR_FAR_PROMPT: 0.15,
    SCOOTER_PROMPT: 0.25,
    MOTORCYCLE_PROMPT: 0.17,
    MOTORCYCLE_RIDER_PROMPT: 0.15,
    MOTORCYCLE_DISTANT_PROMPT: 0.15,
    MOTORCYCLIST_PROMPT: 0.15,
    PLAIN_SCOOTER_PROMPT: 0.25,
    SCOOTER_RIDER_PROMPT: 0.15,
    TOY_CAR_PROMPT: 0.22,
    TOY_VEHICLE_PROMPT: 0.22,
    "motorcycle": 0.17,
    "motorbike": 0.17,
    "car": 0.30,
    "scooter": 0.25,
    "kick scooter": 0.25,
    "electric scooter": 0.17,
    "electric kick scooter": 0.17,
    "bicycle": 0.52,
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
    CLASS_MIN_CONF[prompt] = 0.18

PROMPTS = [
    "box", "desk", "chair",
    "bed with mattress and pillow",
    "bed with visible mattress, pillow and blanket in a bedroom",
    "shelf", "bookshelf", "storage shelf", "freestanding storage shelf", "wardrobe", "closet",
    "refrigerator", "refrigerator in a kitchen", "large household refrigerator",
    # "window", "glass window", "building window", "exterior window",
    *sorted(HANGER_PROMPTS),
    *sorted(WALL_SURFACE_LABELS),
    "tree", CAR_PROMPT, "car", CAR_DISTANT_PROMPT, CAR_FAR_PROMPT,
    TOY_CAR_PROMPT, TOY_VEHICLE_PROMPT,
    SCOOTER_PROMPT, PLAIN_SCOOTER_PROMPT, SCOOTER_RIDER_PROMPT, "scooter",
    MOTORCYCLE_PROMPT, MOTORCYCLE_RIDER_PROMPT, MOTORCYCLE_DISTANT_PROMPT,
    MOTORCYCLIST_PROMPT, "motorcycle", "bollard",
    "utility pole", "bicycle", "person", "curb", "railing", "guardrail",
    "door", "entrance door", "interior door", "wooden door", "closed door", "open door",
    "door", "wooden door", "steel door", "glass entrance door", "sliding glass door", "automatic glass door",
    "door handle", "doorknob", "lever door handle", "door knob",
    "push bar on a door", "automatic door push button", "electronic keypad door lock", "door lock",
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
    "sliding door", "glass sliding door", "sliding glass door",
    *sorted(GLASS_DOOR_TYPE_LABELS),
    "automatic door", "glass door", "hinged door",
    *GLASS_DOOR_LABELS, *STEEL_DOOR_LABELS,
}

DOOR_COMPONENT_LABELS = {
    *sorted(HANDLE_TYPE_LABELS),
    *sorted(LOCK_TYPE_LABELS),
    "door push button switch",
}

for handle_prompt in HANDLE_TYPE_LABELS:
    CLASS_MIN_CONF[handle_prompt] = 0.48

for lock_prompt in LOCK_TYPE_LABELS:
    CLASS_MIN_CONF[lock_prompt] = 0.45

CLASS_MIN_CONF["door push button switch"] = 0.45

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
    CAR_PROMPT, CAR_DISTANT_PROMPT, CAR_FAR_PROMPT,
    SCOOTER_PROMPT, SCOOTER_RIDER_PROMPT,
    MOTORCYCLE_PROMPT, MOTORCYCLE_RIDER_PROMPT,
    MOTORCYCLE_DISTANT_PROMPT, MOTORCYCLIST_PROMPT, PLAIN_SCOOTER_PROMPT,
    TOY_CAR_PROMPT, TOY_VEHICLE_PROMPT,
    "car", "motorcycle", "motorbike", "scooter", "kick scooter",
    "electric scooter", "electric kick scooter", "bicycle",
}

HANDLE_MIN_CONF = 0.18
DOOR_MIN_CONF = 0.15

KOREAN_LABELS = {
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
    # "window": "창문",
    # "glass window": "창문",
    # "building window": "창문",
    # "exterior window": "창문",
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
    TOY_CAR_PROMPT: "차량",
    TOY_VEHICLE_PROMPT: "차량",
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
    "doorknob": "원형 문고리",
    "door knob": "원형 문고리",
    "door handle": "문손잡이",
    "lever handle": "레버 손잡이",
    "lever door handle": "레버 손잡이",
    "fingerprint door lock": "도어락",
    "smart door lock with keypad": "도어락",
    "round doorknob on a door": "원형 문고리",
    "lever door handle attached to a door": "레버 손잡이",
    "push bar on a commercial door": "푸시바",
    "push bar": "푸시바",
    "door push button": "문 버튼",
    "recessed door handle": "매립 손잡이",
    "door lock": "도어락",
    "door lock with handle": "도어락",
    "keypad door lock": "도어락",
    "electronic keypad door lock": "도어락",
    "deadbolt lock on a door": "도어락",
    "door lever lock": "도어락",
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
    "hinged glass door with a frame": "여닫이 유리문",
    "glass hinged entrance door": "여닫이 유리문",
    "glass door with a vertical pull handle": "손잡이형 유리문",
    **{label: "유리문" for label in GLASS_DOOR_LABELS},
    "round spherical metal doorknob mounted on a door": "원형 문고리",
    "horizontal lever door handle attached to a door": "레버 손잡이",
    "recessed handle flush with a sliding door surface": "매립 손잡이",
    "long vertical pull handle bar on a commercial glass door": "문손잡이",
    "horizontal panic push bar across a exit door": "푸시바",
    "digital keypad smart door lock with numbers": "도어락",
    "electronic door lock mounted on a door panel": "도어락",
    "door push button switch": "문 버튼",
}

class AsyncOCRReader:
    def __init__(self, speech_instance):
        self.speech = speech_instance
        self.job_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._worker, name="ocr-worker", daemon=True)
        self.thread.start()

    def request_ocr(self, roi_img):
        if not HAS_PYTESSERACT or self.job_queue.full():
            return
        try:
            self.job_queue.put_nowait(roi_img)
        except queue.Full:
            pass

    def _worker(self):
        while self.running:
            try:
                roi_img = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # 1. 흑백 변환 및 3배 확대
                gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

                # 2. 대비 극대화 (선명한 흑백 전처리)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 3. PSM 6 적용 (단일 텍스트 블록/스티커 전용)
                text = pytesseract.image_to_string(thresh, lang='kor+eng', config='--psm 6').strip()
                upper_text = text.upper()

                # 4. 키워드 조건 완화
                if any(k in text for k in ["당기", "당", "PULL"]) or "PULL" in upper_text:
                    self.speech.say("ocr_pull", "문 당기세요")
                elif any(k in text for k in ["미", "밀", "PUSH"]) or "PUSH" in upper_text:
                    self.speech.say("ocr_push", "문 미세요")
                else:
                    digits = "".join([c for c in text if c.isdigit()])
                    if len(digits) >= 2:
                        self.speech.say(f"ocr_num_{digits}", f"숫자 {digits}")

            except Exception:
                pass

    def close(self):
        self.running = False


class UltrasonicSensor:
    TRIG_PIN = 23
    ECHO_PIN = 24
    MAX_DISTANCE_CM = 400.0
    WARNING_DISTANCE_CM = 50.0

    def __init__(self):
        self.gpio = None
        self.ready = False
        self._beep_lock = threading.Lock()
        self._last_beep = 0.0
        self._beep_interval = 0.65
        self._aplay = shutil.which("aplay")
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
                [self._aplay, "-q", "-D", self._alsa_device, "-t", "raw",
                 "-f", "S16_LE", "-r", "24000", "-c", "1"],
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
    shutdown_cmd = shutil.which("shutdown") or "/sbin/shutdown"
    command = ([shutdown_cmd, "-h", "now"] if os.geteuid() == 0
               else ["sudo", "-n", shutdown_cmd, "-h", "now"])
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=5.0, check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "권한이 거부되었습니다.").strip()
            print(f"라즈베리파이 종료 명령 실패: {detail}", flush=True)
    except Exception as exc:
        print(f"라즈베리파이 종료 명령 실행 실패: {exc}", flush=True)


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
        
        self.aplay = shutil.which("aplay")
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
                    [self.aplay, "-q", "-D", device, str(path)],
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
            if self.current_text == text:
                return

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
        if self.ultrasonic_suppressed and not key.startswith(("crosswalk_", "traffic_", "stairs:")):
            return
        
        now = time.time()
        if not force and key == self.last_key and now - self.last_submit_time < 2.5:
            return
            
        if self.current_text is not None:
            return

        self.last_key = key
        self.last_submit_time = now
        print(f"[결과] {text}", flush=True)
        self._queue_audio(text, urgent=False)

    def say_urgent(self, key, text):
        if not isinstance(text, str) or not text.strip():
            return
        now = time.monotonic()
        repeat_window = 1.5 if key == "traffic_stop" else 1.2
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

    if label in DOOR_COMPONENT_LABELS or label in DOOR_LOCK_LABELS or label in LOCK_TYPE_LABELS:
        return bw >= 2 and bh >= 2 and area_ratio <= 0.15

    if label in {SCOOTER_PROMPT, PLAIN_SCOOTER_PROMPT, "scooter", "kick scooter", "electric scooter"}:
        return (
            area_ratio >= 0.002 
            and bh >= height * 0.08
            and 0.15 <= aspect <= 1.1
        )
    if label in VEHICLE_LABELS:
        return (bw >= 20 and bh >= 15 and area_ratio >= 0.0008
                and area_ratio <= 0.8 and 0.45 <= aspect <= 5.0)
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
        return (
            area_ratio >= 0.005
            and bw >= width * 0.10
            and bh >= height * 0.05
            and aspect >= 0.8
            and y2_ratio >= 0.50
        )
    if label in GLASS_DOOR_LABELS or label in GLASS_DOOR_TYPE_LABELS:
        return area_ratio >= 0.04 and bh >= height * 0.22 and 0.20 <= aspect <= 3.5
    if label in DOOR_LABELS:
        return area_ratio >= 0.00005 and bh >= height * 0.010 and aspect <= 14.0
    if label == "elevator":
        return area_ratio >= ELEVATOR_MIN_AREA_RATIO and bw >= width * ELEVATOR_MIN_WIDTH_RATIO and bh >= height * 0.20
    if label in ELEVATOR_SUPPORT_LABELS:
        return area_ratio >= 0.00015 and area_ratio <= 0.08
    if label in DOOR_COMPONENT_LABELS:
        return bw >= 4 and bh >= 4 and area_ratio <= 0.08
    if label in STAIR_LABELS:
        return (area_ratio >= 0.003 or bw >= width * 0.05
                or bh >= height * 0.04)
    return area_ratio >= 0.002


def prepare_inference_frame(frame):
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


def guidance_for_detection(det, width, all_dets=None):
    label, _, x1, y1, x2, y2 = det
    pos = zone(x1, x2, width)

    # 1. 독립 부속품 단독 감지 시
    if label in {"doorknob", "door knob", "door knob mounted on a door", "round doorknob on a door", "round ball-shaped spherical metal doorknob on a door"}:
        return f"{pos} 원형 문고리"
    if label in {"push bar", "push bar on a commercial door", "horizontal panic push bar across a exit door"}:
        return f"{pos} 푸시바, 밀으세요"
    if label in {"lever handle", "lever door handle", "lever door handle attached to a door", "horizontal lever door handle attached to a door", "straight horizontal lever door handle attached to a door"}:
        return f"{pos} 레버 손잡이, 내리세요"
    if label in {"door handle", "door handle mounted on a door", "metal handle attached to a door", "long vertical pull handle bar on a commercial glass door"}:
        return f"{pos} 문손잡이"
    if label in {"door push button", "door push button switch"}:
        return f"{pos} 문 버튼, 누르세요"
    if label in DOOR_LOCK_LABELS or label in LOCK_TYPE_LABELS or "keypad" in label or "lock" in label:
        return f"{pos} 도어락"

    # 2. 문(Door) 구조체 감지 시
    if label in DOOR_LABELS or label in STEEL_DOOR_LABELS or label in GLASS_DOOR_LABELS or label in GLASS_DOOR_TYPE_LABELS:
        
        # 재질 구분
        if "glass" in label.lower() or label in GLASS_DOOR_LABELS or label in GLASS_DOOR_TYPE_LABELS:
            door_name = "유리문"
        elif label == "wooden door":
            door_name = "나무문"
        elif label in STEEL_DOOR_LABELS:
            door_name = "철문"
        else:
            door_name = "문"

        # 문 주변(마진 60px로 확대하여 먼 거리 탐색) 부속품 탐색
        nearby_comp = None
        if all_dets:
            for other in all_dets:
                olabel = other[0]
                if olabel in DOOR_COMPONENT_LABELS or olabel in DOOR_LOCK_LABELS or olabel in LOCK_TYPE_LABELS:
                    ox1, oy1, ox2, oy2 = other[2:6]
                    ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
                    if (x1 - 60) <= ocx <= (x2 + 60) and (y1 - 60) <= ocy <= (y2 + 60):
                        nearby_comp = olabel
                        break

        # A. 버튼/자동문
        if nearby_comp in {"door push button", "door push button switch"} or label in {"automatic door", "automatic glass door", "automatic glass sliding door"}:
            return f"{pos} 자동문, 버튼을 누르세요"

        # B. 미닫이문
        if nearby_comp in {"recessed handle flush with a sliding door surface", "recessed door handle"} or label in {"glass sliding door", "sliding glass door", "manual sliding glass door", "sliding door"}:
            return f"{pos} 미닫이문, 옆으로 미세요"

        # C. 확정 손잡이 단서가 조합된 경우
        if nearby_comp:
            if nearby_comp in {"lever door handle", "horizontal lever door handle attached to a door", "lever handle", "straight horizontal lever door handle attached to a door"}:
                return f"{pos} {door_name}, 레버 손잡이를 내리세요"
            elif nearby_comp in {"doorknob", "door knob", "round spherical metal doorknob mounted on a door", "round ball-shaped spherical metal doorknob on a door"}:
                return f"{pos} {door_name}, 문고리를 돌리세요"
            elif nearby_comp in {"push bar", "push bar on a commercial door"}:
                return f"{pos} {door_name}, 푸시바를 밀으세요"
            elif nearby_comp in {"door handle", "door handle mounted on a door", "long vertical pull handle bar on a commercial glass door"}:
                return f"{pos} {door_name}, 손잡이 잡고 당기세요"
            elif nearby_comp in DOOR_LOCK_LABELS or nearby_comp in LOCK_TYPE_LABELS:
                return f"{pos} {door_name}, 도어락"

        # D. 단서 없을 때
        return f"{pos} {door_name}"

    # 3. 기타 사물
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

    def reset(self):
        self.detections = []
        self.last_seen = None

    def update(self, detections, now=None):
        now = time.monotonic() if now is None else now
        if detections:
            self.detections = list(detections)
            self.last_seen = now
            return list(self.detections)
        if self.last_seen is not None and now - self.last_seen <= CROSSWALK_EXIT_HOLD_SECONDS:
            return list(self.detections)
        self.reset()
        return []


class SituationMachine:
    def __init__(self):
        self.state = "WALK"
        self.forced = False
        self.last_crosswalk_time = None

    def update(self, detections, crosswalk_held_dets):
        now = time.monotonic()
        
        has_crosswalk = (len(crosswalk_held_dets) > 0) or any(d[0] in CROSSWALK_LABELS for d in detections)

        if has_crosswalk:
            self.last_crosswalk_time = now
            self.state = "CROSSWALK"
            return self.state

        if self.state == "CROSSWALK":
            if (self.last_crosswalk_time is not None 
                    and now - self.last_crosswalk_time < CROSSWALK_EXIT_HOLD_SECONDS):
                return "CROSSWALK"
            
            self.state = "WALK"
            self.last_crosswalk_time = None

        return self.state


class CrosswalkSafety:
    def __init__(self):
        self.status = "CLEAR"
        self.stopped_vehicle_tracks = []
        # 차량 감지 안 될 때 기존 정지 메모리 유지 시간 (1.8초)
        self.HOLD_SECONDS = 1.8
        # 차가 완전히 멈췄다고 확정짓기 위해 필요한 최소 체류 시간 (0.7초)
        self.STATIONARY_CONFIRM_SECONDS = 0.8

    def reset(self):
        self.status = "CLEAR"
        self.stopped_vehicle_tracks.clear()

    def update(self, vehicles, width, height, now=None):
        now = time.monotonic() if now is None else now

        # 1. 1.8초 동안 재감지되지 않은 오래된 정지 메모리 정리
        self.stopped_vehicle_tracks = [
            tr for tr in self.stopped_vehicle_tracks
            if now - tr["last_seen"] <= self.HOLD_SECONDS
        ]

        # 2. 감지된 차가 없으면 기존 메모리 기반으로 CLEAR 반환
        if not vehicles:
            self.status = "CLEAR"
            return "CLEAR", False

        moving_vehicle_count = 0
        MAX_DIST_SQ = 0.0064  # 중심점 변동 허용 범위 제곱값
        inv_w = 1.0 / max(1, width)
        inv_h = 1.0 / max(1, height)

        for det in vehicles:
            x1, y1, x2, y2 = det[2:6]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

            matched_track = None
            for track in self.stopped_vehicle_tracks:
                scx, scy = track["center"]
                dx = (cx - scx) * inv_w
                dy = (cy - scy) * inv_h
                dist_sq = dx * dx + dy * dy

                if dist_sq <= MAX_DIST_SQ:
                    matched_track = track
                    break

            if matched_track is not None:
                # 기존 트랙에 매칭됨: 중심점 및 시간 갱신
                matched_track["center"] = (cx, cy)
                matched_track["last_seen"] = now
                
                # 정지 체류 시간이 0.7초 미만이면 아직 멈춘 차로 인정 안함 (움직이는 차로 간주)
                if now - matched_track["first_seen"] < self.STATIONARY_CONFIRM_SECONDS:
                    moving_vehicle_count += 1
            else:
                # 완벽히 새로 나타난 차: 새로 관찰 시작 등록만 하고 '이동 차'로 분류
                self.stopped_vehicle_tracks.append({
                    "center": (cx, cy),
                    "first_seen": now,
                    "last_seen": now
                })
                moving_vehicle_count += 1

        # 3. 0.7초 이상 멈춰서지 않은 '이동 중인 차'가 1대라도 있으면 즉시 WAIT 유지
        if moving_vehicle_count > 0:
            self.status = "WAIT"
            return "WAIT", True

        # 4. 화면의 모든 차가 0.7초 이상 제자리에 멈춰있음이 확정된 경우에만 안정적으로 CLEAR 출력
        self.status = "CLEAR"
        return "CLEAR", False


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
    def is_inside_or_near_door(component, doors, max_pixel_margin=30):
        # 문 감지 결과가 하나도 없는 상황이면 무분별한 오탐지 방지를 위해 손잡이 거부
        if not doors:
            return False
            
        cx, cy = (component[2] + component[4]) / 2.0, (component[3] + component[5]) / 2.0
        
        for door in doors:
            dx1, dy1, dx2, dy2 = door[2:6]
            # 손잡이 중심점이 문 영역(여유 마진 포함) 내에 포함되어 있는지 확인
            if (dx1 - max_pixel_margin) <= cx <= (dx2 + max_pixel_margin) and \
               (dy1 - max_pixel_margin) <= cy <= (dy2 + max_pixel_margin):
                return True
        return False

    def update(self, detections, confirmed_doors):
        candidates = []
        raw_components = [
            d for d in detections
            if d[0] in DOOR_COMPONENT_LABELS
            and d[1] >= CLASS_MIN_CONF.get(d[0], 0.45)
        ]
        
        # 전체 감지된 문 후보(미확정 포함) 추출
        all_doors = [d for d in detections if d[0] in DOOR_LABELS] + confirmed_doors

        for comp in raw_components:
            # 문 영역 내부/근처에 위치한 손잡이만 최종 승인
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
        torch.set_num_threads(max(1, int(os.getenv("WALKING_ASSISTANT_CPU_THREADS", "2"))))
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
            sync_cuda_for_profile(model)

            all_dets = []
            crosswalk_vehicles = []
            crosswalk_dets = []

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = label_of(result, cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                aspect = bw / bh

                # [손잡이 상호 교정 정교화]
                # 1. 원형 문고리로 잡혔으나 가로로 길쭉(aspect >= 1.2)한 경우 -> 레버 손잡이
                if label in {"doorknob", "door knob", "round spherical metal doorknob mounted on a door", "round ball-shaped spherical metal doorknob on a door"}:
                    if aspect >= 1.15:
                        label = "straight horizontal lever door handle attached to a door"
                        conf = max(conf, 0.35)
                        
                # 레버로 잡혔지만 정사각형/둥근 형태(aspect <= 1.05) ➔ 원형 손잡이
                elif label in {"lever handle", "lever door handle", "horizontal lever door handle attached to a door", "straight horizontal lever door handle attached to a door"}:
                    if aspect <= 1.05:
                        label = "round ball-shaped spherical metal doorknob on a door"
                        conf = max(conf, 0.35)

                # [냉장고 vs 철문 교정] 냉장고로 잡혔으나 높이가 방 전체 문(bh > h * 0.7) 수준이면 철문/문으로 변경
                if label in {"refrigerator", "refrigerator in a kitchen", "large household refrigerator"}:
                    if bh >= h * 0.70:
                        label = "steel door"

                if label in {"scooter", "kick scooter", SCOOTER_PROMPT, PLAIN_SCOOTER_PROMPT}:
                    if aspect >= 1.0 or bh < h * 0.08:
                        label = TOY_CAR_PROMPT
                        conf = max(conf, 0.20)

                det = (label, conf, x1, y1, x2, y2)

                # 문 스티커/숫자 표지판 영역이 감지되었을 때 백그라운드 OCR 요청 (지연 0초)
                if label in TEXT_SIGN_PROMPTS and conf >= CLASS_MIN_CONF.get(label, 0.18):
                    rx1, ry1 = max(0, x1 - 5), max(0, y1 - 5)
                    rx2, ry2 = min(w, x2 + 5), min(h, y2 + 5)
                    crop_img = frame[ry1:ry2, rx1:rx2]
                    if crop_img.shape[0] > 10 and crop_img.shape[1] > 10:
                        ocr_reader.request_ocr(crop_img)
                
                if conf >= class_min_conf(label) and valid_geometry(det, w, h):
                    all_dets.append(det)
                    if label in VEHICLE_LABELS and conf >= CROSSWALK_VEHICLE_MIN_CONF:
                        crosswalk_vehicles.append(det)
                    elif label in CROSSWALK_LABELS and conf >= CROSSWALK_MIN_CONF:
                        crosswalk_dets.append(det)

            crosswalk_vehicles = deduplicate_vehicle_detections(crosswalk_vehicles)
            all_dets = deduplicate_detections(all_dets)

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

            crosswalk_dets = crosswalk_frame_hold.update(crosswalk_dets)
            state = machine.update(all_dets, crosswalk_dets)

            general_mode = state != "CROSSWALK"
            ultrasonic_distance_cm = ultrasonic.measure_cm() if general_mode else None
            speech.set_ultrasonic_suppressed(
                general_mode and ultrasonic_distance_cm is not None and ultrasonic_distance_cm >= UltrasonicSensor.MAX_DISTANCE_CM
            )
            ultrasonic.update_warning(ultrasonic_distance_cm, active=general_mode)

            if state == "CROSSWALK":
                # 횡단보도 상태로 처음 변경되었거나 재탐지될 때 기존 정지 차 메모리 리셋
                if last_reported_state != "CROSSWALK":
                    crosswalk_safety.reset()
                    speech.say("crosswalk_front", "정면 횡단보도", force=True)

                traffic_status, urgent_stop = crosswalk_safety.update(crosswalk_vehicles, w, h)
                speech.set_vehicle_priority(traffic_status == "WAIT")

                if last_reported_state != "CROSSWALK":
                    speech.say("crosswalk_front", "정면 횡단보도", force=True)

                if traffic_status == "WAIT":
                    last_crosswalk_wait_time = time.monotonic()

                if urgent_stop or (traffic_status == "WAIT" and last_traffic_status != "WAIT"):
                    speech.say_urgent("traffic_stop", "멈추세요")
                    last_guidance_key = "traffic:WAIT"
                elif traffic_status == "CLEAR":
                    now = time.monotonic()
                    wait_finished = (
                        last_crosswalk_wait_time is None
                        or now - last_crosswalk_wait_time >= CROSSWALK_STOP_TO_CROSS_DELAY_SECONDS
                    )
                    if wait_finished and (
                        last_crosswalk_clear_announce == 0.0
                        or now - last_crosswalk_clear_announce >= CROSSWALK_REPEAT_SECONDS
                    ):
                        speech.say("crosswalk_clear", "건너세요", force=True)
                        last_crosswalk_clear_announce = now
                        last_crosswalk_wait_time = None
                    last_guidance_key = "traffic:CLEAR"
            else:
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
                    if is_walk_obstacle(d) and (d[0] not in DOOR_LABELS or d in confirmed_doors or d[0] in DOOR_COMPONENT_LABELS)
                ]
                primary = target_selector.update(obstacle_candidates, w, h)

                if primary is not None:
                    position = zone(primary[2], primary[4], w)
                    text = guidance_for_detection(primary, w, all_dets=all_dets)  # all_dets 추가 전달
                    guidance_key = f"obstacle:{primary[0]}:{position}"
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
        cap.release()

if __name__ == "__main__":
    main()
