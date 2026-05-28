import cv2
import numpy as np
import threading
import time
from ultralytics import YOLOWorld
import torch

# 💡 스레드 효율 분배 고정
torch.set_num_threads(2)

# 전역 변수 설정
latest_frame = None
processed_frame_to_show = None
frame_lock = threading.Lock()
is_running = True

def speak_pure_audio(text):
    pass

def inference_thread_func(model, clean_labels, crop_config):
    global latest_frame, processed_frame_to_show, is_running

    last_spoken_state = ""
    last_spoken_time = 0
    AUDIO_COOLDOWN = 1.3
    
    start_x, end_x = crop_config

    # 💡 문 수치 상향(0.23), 냉장고 수치 상향(0.30) 밸런스 유지
    class_conf_thresholds = {
        'door': 0.23,           # 너무 쉽게 문으로 착각하지 않도록 문턱 유지
        'refrigerator': 0.30,   # 옷장/선반이 냉장고로 튀는 현상 방지
        'wardrobe': 0.45,       
        'shelf': 0.42       
    }

    print("🧠 [추론 스레드] 밸런싱 패치 버전 사물 탐지를 시작합니다.")

    while is_running:
        with frame_lock:
            local_frame = latest_frame
            latest_frame = None

        if local_frame is None:
            time.sleep(0.005)
            continue

        # 정면 영역 크롭 (480x400)
        cropped_frame = local_frame[:, start_x:end_x]
        display_frame = cropped_frame.copy()

        try:
            results = model.predict(display_frame, imgsz=320, conf=0.15, iou=0.40, verbose=False)
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    raw_name = model.names[cls_id]
                    class_name = clean_labels.get(raw_name, raw_name)

                    # 💡 [일반 물체 필터링] 커스텀 조절 수치 미만은 가차없이 컷
                    if class_name in class_conf_thresholds and conf < class_conf_thresholds[class_name]:
                        continue

                    # 필터를 통과한 정예 사물들만 화면에 박스 그리기
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    label_str = f"{class_name} {conf:.2f}"
                    (t_w, t_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (int(x1), int(y1) - 20), (int(x1) + t_w, int(y1)), (255, 0, 0), -1)
                    cv2.putText(display_frame, label_str, (int(x1), int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    detected_objects.append({'name': class_name, 'y2_coord': y2, 'conf': conf})

            if detected_objects:
                # 확신도(conf) 기준 내림차순 정렬
                detected_objects.sort(key=lambda o: o['conf'], reverse=True)
                best_match_obj = detected_objects[0]
                obj_name = best_match_obj['name']

                if obj_name not in ["wall", "window", "tree"]:
                    current_time = time.time()
                    if (obj_name != last_spoken_state) or (current_time - last_spoken_time > AUDIO_COOLDOWN):
                        print(f"[가장 확실함] 🎯 {obj_name} (확신도: {best_match_obj['conf']:.2f})")
                        last_spoken_state = obj_name
                        last_spoken_time = current_time
            else:
                last_spoken_state = ""

        except Exception as e:
            print(f"추론 오류: {e}")

        with frame_lock:
            processed_frame_to_show = display_frame

        time.sleep(0.01)

def run_debug_viewer():
    global latest_frame, processed_frame_to_show, is_running

    model = YOLOWorld('fixed_model.pt')

    # 💡 실내외 라벨 및 ramp 딕셔너리에서 제거
    clean_labels = {
        "a room door with a handle": "door", "an entrance door": "door", "a glass door": "door",
        "concrete curb on the street": "curb", "stairs leading downwards": "stairs", "stairs leading upwards": "stairs",
        "electric kick scooter": "scooter",
        "person": "person", "bicycle": "bicycle", "motorcycle": "motorcycle", "sink": "sink",
        "stop sign": "stop sign", "bench": "bench", "chair": "chair", "potted plant": "plant",
        "tv": "tv", "laptop": "laptop", "cell phone": "phone", "microwave": "microwave",
        "bollard": "bollard", "traffic cone": "cone", "utility pole": "pole", "water puddle": "puddle",
        "a storage shelf with racks": "shelf", "stair": "stair", "tree": "tree",
        "a passenger car": "car", "a large passenger bus": "bus", "a cargo truck": "truck",
        "a kitchen refrigerator": "refrigerator", "a bed": "bed", "a dining table": "table",
        "a cardboard box": "box", "wall": "wall", "window": "window", "a wooden wardrobe": "wardrobe",
        "clothes hanger rack": "hanger"
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ 카메라는 열 수 없습니다.")
            return

    W, H = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    crop_w = 400
    start_x = (W - crop_w) // 2
    end_x = start_x + crop_w

    ai_thread = threading.Thread(
        target=inference_thread_func, 
        args=(model, clean_labels, (start_x, end_x)), 
        daemon=True
    )
    ai_thread.start()

    print("🖥️ 시각화 뷰어를 재시작합니다. ('q' 입력 시 종료)")

    try:
        while True:
            for _ in range(3):
                cap.grab()
                
            ret, frame = cap.retrieve()
            if not ret:
                break

            with frame_lock:
                latest_frame = frame
                frame_to_display = processed_frame_to_show
                
            if frame_to_display is not None:
                cv2.imshow("YOLO-World Live Debug Window", frame_to_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        is_running = False
        ai_thread.join(timeout=0.2)
        cap.release()
        cv2.destroyAllWindows()
        print("👋 안전하게 종료되었습니다.")

if __name__ == "__main__":
    run_debug_viewer()
