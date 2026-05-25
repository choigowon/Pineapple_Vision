import cv2
import numpy as np
import threading
import time
import subprocess
from ultralytics import YOLOWorld
import torch

# 💡 스레드 효율 분배 고정
torch.set_num_threads(2)

latest_frame = None
frame_lock = threading.Lock()
is_running = True

def speak_pure_audio(text):
    try:
        subprocess.Popen(["espeak", f'"{text}"', "-s", "190"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def inference_thread_func(model, clean_labels):
    global latest_frame, is_running

    last_spoken_state = ""
    last_spoken_time = 0
    AUDIO_COOLDOWN = 1.3

    print("🧠 실시간 추론을 시작합니다.")

    while is_running:
        with frame_lock:
            local_frame = latest_frame
            latest_frame = None  # 프레임 꺼내오고 비우기

        if local_frame is None:
            time.sleep(0.005)
            if not is_running: return
            continue

        try:
            results = model.predict(local_frame, imgsz=320, conf=0.25, iou=0.45, verbose=False)
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    y2 = float(box.xyxy[0][3])

                    raw_name = model.names[cls_id]
                    class_name = clean_labels.get(raw_name, raw_name)

                    detected_objects.append({'name': class_name, 'y2_coord': y2})

            if detected_objects:
                detected_objects.sort(key=lambda o: o['y2_coord'], reverse=True)
                obj_name = detected_objects[0]['name']

                if obj_name not in ["wall", "window", "wardrobe", "shelf", "tree"]:
                    current_time = time.time()

                    if (obj_name != last_spoken_state) or (current_time - last_spoken_time > AUDIO_COOLDOWN):
                        print(f"[즉시 발화] 🎯 {obj_name}")
                        speak_pure_audio(obj_name)

                        last_spoken_state = obj_name
                        last_spoken_time = current_time
            else:
                last_spoken_state = ""

        except Exception as e:
            print(f"추론 오류: {e}")

        time.sleep(0.01)

def run_velog_optimized_camera():
    global latest_frame, is_running

    model = YOLOWorld('fixed_model.pt')

    clean_labels = {
        "a room door with a handle": "door", "a partially visible entrance door": "door",
        "a closed wooden door frame": "door", "a glass door with a metal handle": "door",
        "concrete curb on the street": "curb", "stairs leading downwards": "stairs", "stairs leading upwards": "stairs",
        "wheelchair ramp or slope": "ramp", "electric kick scooter": "scooter",
        "person": "person", "bicycle": "bicycle", "motorcycle": "motorcycle", "sink": "sink",
        "stop sign": "stop sign", "bench": "bench", "chair": "chair", "potted plant": "plant",
        "tv": "tv", "laptop": "laptop", "cell phone": "phone", "microwave": "microwave",
        "bollard": "bollard", "traffic cone": "cone", "utility pole": "pole", "water puddle": "puddle",
        "shelf": "shelf", "stair": "stair", "tree": "tree",
        "a car or a part of a car": "car", "a bus or a part of a bus": "bus", "a truck or a part of a truck": "truck",
        "a large refrigerator": "refrigerator", "a bed in a bedroom": "bed", "a dining table": "table",
        "a cardboard box on the floor": "box", "wall": "wall", "window": "window", "wardrobe": "wardrobe",
        "clothes hanger rack": "hanger"
    }

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ai_thread = threading.Thread(target=inference_thread_func, args=(model, clean_labels), daemon=True)
    ai_thread.start()

    print("🚀 밀림 원천 차단 최적화 모드 시작! (Ctrl+C 종료)")

    try:
        while True:
            # 💡 [밀림 방지 핵심] 하드웨어 버퍼에 쌓인 과거 프레임을 통째로 무시합니다.
            # grab()으로 쌓인 버퍼를 순간적으로 다 털어내야만 '밀려서 출력하는 느낌'이 완벽히 사라집니다.
            for _ in range(7):
                cap.grab()

            ret, frame = cap.retrieve()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            with frame_lock:
                latest_frame = frame

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        is_running = False
        ai_thread.join(timeout=0.2)
        cap.release()

if __name__ == "__main__":
    run_velog_optimized_camera()
