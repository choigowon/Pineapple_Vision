import cv2
import numpy as np
import threading
import time
import subprocess
from ultralytics import YOLO

# --- [스레드 1] 실시간 최신 화면 가로채기 스트림 ---
class RealtimeCameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if self.cap.isOpened(): self.cap.release()


# --- [스레드 2] AI 추론 백그라운드 격리 스레드 (640 해상도) ---
class BackgroundInferenceThread:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='detect')
        self.frame_to_process = None
        self.latest_result = None
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        threading.Thread(target=self.inference_loop, daemon=True).start()
        return self

    def update_frame(self, frame):
        with self.lock:
            self.frame_to_process = frame

    def get_result(self):
        with self.lock:
            res = self.latest_result
            self.latest_result = None  
            return res

    def inference_loop(self):
        while self.started:
            img = None
            with self.lock:
                if self.frame_to_process is not None:
                    img = self.frame_to_process
                    self.frame_to_process = None

            if img is not None:
                results = self.model.predict(img, conf=0.25, iou=0.35, imgsz=640, stream=True, verbose=False)
                objects = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        objects.append({'cls_id': cls_id, 'box': (x1, y1, x2, y2), 'conf': conf_score})
                with self.lock:
                    self.latest_result = objects
            else:
                time.sleep(0.01)


# --- [초고속 오디오 스레드 관리자] 파이썬 메모리 레벨에서 구형 음성을 즉시 폭파 ---
class UltraFastAudioEngine:
    def __init__(self):
        self.current_process = None
        self.lock = threading.Lock()

    def speak_now(self, text):
        with self.lock:
            # 💥 다른 사물이 들어오는 순간 기존 실행 중이던 espeak 프로세스를 OS 레벨에서 즉각 강제 폭파(kill)
            if self.current_process is not None:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=0.05)
                except Exception:
                    pass
            
            # 음성 속도를 185로 올려 더 민첩하게 뱉고, 볼륨을 최대(-a 200)로 상향
            # 쉼표(,)를 붙여서 단어 간의 간격만 또박또박하게 유지
            self.current_process = subprocess.Popen(
                ["espeak", f'"{text}"', "-s", "185", "-a", "200"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

audio_manager = UltraFastAudioEngine()


# --- 메인 제어 커널 ---
def run_system():
    infra_ai = BackgroundInferenceThread('my_fixed_yolov8s.onnx').start()
    vs = RealtimeCameraStream(src=0).start()
    time.sleep(1.0)

    custom_classes = [
        "door", "door", "door", "door", "curb", "stairs down", "stairs up", "ramp", "scooter",
        "person", "bicycle", "motorcycle", "sink", "stop sign", "bench", "chair", "potted plant",
        "tv", "laptop", "cell phone", "microwave", "bollard", "traffic cone", "utility pole",
        "water puddle", "shelf", "stair", "tree", "car", "bus", "truck", "refrigerator", "bed",
        "dining table", "box", "wall", "window", "wardrobe", "hanger"
    ]
    ignored_speech_classes = ["wall", "window", "wardrobe"]

    last_spoken_state = ""
    last_spoken_time = 0
    
    # ⚡ 가로채기 성능을 극한으로 끌어올리기 위해 오디오 발화 최소 방어선 주기를 제거(0.2초)
    AUDIO_MIN_INTERVAL = 0.2  

    print("\n" + "="*50)
    print("🚀 [초고속 가로채기 엔진 + 1단어 거리 브리핑] 구동")
    print("사물이 바뀌면 말하던 도중 즉시 끊고 다음 사물로 넘어갑니다.")
    print("="*50 + "\n")

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h_frame, w_frame = frame.shape[:2]

            infra_ai.update_frame(frame)
            detected_boxes = infra_ai.get_result()

            if detected_boxes is not None:
                detected_objects = []
                for obj in detected_boxes:
                    x1, y1, x2, y2 = obj['box']
                    class_name = custom_classes[obj['cls_id']]
                    
                    if class_name in ignored_speech_classes:
                        continue  

                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / (w_frame * h_frame)
                    
                    # 💡 요구사항 반영: 극단적으로 짧고 즉각적인 단어로 수정
                    if area_ratio > 0.12:
                        distance_status = "Close"
                    elif area_ratio > 0.03:
                        distance_status = "Front"
                    else:
                        distance_status = "Far"

                    detected_objects.append({
                        'name': class_name, 'distance': distance_status, 'y2_coord': y2, 'conf': obj['conf']
                    })

                if detected_objects:
                    # 화면 하단에 가장 가까운 최우선 사물 1개만 스크리닝
                    detected_objects.sort(key=lambda o: o['y2_coord'], reverse=True)
                    
                    closest_target = detected_objects[0]
                    # "Close, person." 또는 "Front, chair." 형태로 문장 조합 최소화
                    speech_text = f"{closest_target['distance']}, {closest_target['name']}."

                    current_time = time.time()
                    
                    # 사물 이름이나 거리가 단 1도라도 바뀌면 즉시 가로채기 지시!
                    if (speech_text != last_spoken_state) or (current_time - last_spoken_time > 1.8):
                        if (current_time - last_spoken_time > AUDIO_MIN_INTERVAL):
                            print(f"⚡ [즉시 가로채기 발화]: {speech_text}")
                            audio_manager.speak_now(speech_text)
                            last_spoken_state = speech_text
                            last_spoken_time = current_time
                else:
                    print(".", end="", flush=True)
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        vs.stop()
        infra_ai.started = False

if __name__ == "__main__":
    run_system()
