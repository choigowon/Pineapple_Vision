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
        # 카메라 기본 해상도 자체도 640으로 짱짱하게 맞춥니다.
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


# --- [스레드 2] AI 추론 백그라운드 격리 스레드 (640 풀 파워 규격 박제) ---
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
                # 🔥 ONNX 모델 구조 규격에 정확하게 맞춰 640으로 세팅!
                # 해상도가 높아졌으니 노이즈 헛소리를 방지하기 위해 conf를 0.25로 조입니다.
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


# --- [비동기 오디오] 말하는 도중 새로운 다중 타겟 포착 시 즉시 가로채기 ---
def speak_interrupt_async(text):
    def run_cmd():
        try:
            subprocess.run(["pkill", "-9", "-f", "espeak"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.01)
            subprocess.run(["espeak", f'"{text}"', "-s", "180"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    threading.Thread(target=run_cmd, daemon=True).start()


# --- 메인 실행 엔진 ---
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
    AUDIO_MIN_INTERVAL = 0.6  

    print("\n" + "="*50)
    print("🚀 [640 풀 해상도 일치 + 다중 인식 전체 출력 모드] 가동")
    print("해상도 오동작 에러를 완벽 청소했습니다. (Ctrl + C 종료)")
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
                    distance_status = "close" if area_ratio > 0.12 else "medium" if area_ratio > 0.03 else "far"

                    detected_objects.append({
                        'name': class_name, 'distance': distance_status, 'y2_coord': y2, 'conf': obj['conf']
                    })

                # 다중 사물 전체 출력 및 음성 조합
                if detected_objects:
                    detected_objects.sort(key=lambda o: o['y2_coord'], reverse=True)
                    
                    print(f"\n📸 [포착된 사물 명단 (총 {len(detected_objects)}개)]")
                    for idx, obj in enumerate(detected_objects):
                        print(f"  └ [{idx+1}] {obj['name']} ({obj['distance']}) | 확신도: {obj['conf']:.2f}")

                    closest_target = detected_objects[0]
                    speech_text = f"{closest_target['name']}, {closest_target['distance']}"
                    
                    if len(detected_objects) > 1:
                        second_target = detected_objects[1]
                        if second_target['name'] != closest_target['name']:
                            speech_text += f" and {second_target['name']} {second_target['distance']}"

                    current_time = time.time()
                    if (speech_text != last_spoken_state) or (current_time - last_spoken_time > 1.2):
                        if (current_time - last_spoken_time > AUDIO_MIN_INTERVAL):
                            speak_interrupt_async(speech_text)
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
