import cv2
import numpy as np
import threading
import time
from ultralytics import YOLOWorld
import torch

# 💡 라즈베리파이4 CPU 코어 분배 최적화
torch.set_num_threads(2)

# 전역 변수 설정
latest_frame = None
frame_lock = threading.Lock()
is_running = True

def speak_pure_audio(text):
    pass

def inference_thread_func(model, clean_labels, crop_config):
    global latest_frame, is_running

    last_spoken_state = ""
    last_spoken_time = 0
    AUDIO_COOLDOWN = 1.3
    
    start_x, end_x = crop_config

    # 검증된 황금 밸런스 임계값
    class_conf_thresholds = {
        'door': 0.20,           
        'window': 0.20,         
        'refrigerator': 0.35,   
        'wardrobe': 0.40,       
        'shelf': 0.40,          
        'bench': 0.32,          
        'table': 0.32           
    }

    print("🧠 [라즈베리파이 엔진] 조기 종료(Early Exit) 초경량화 알고리즘 가동.")

    while is_running:
        with frame_lock:
            local_frame = latest_frame
            latest_frame = None

        if local_frame is None:
            time.sleep(0.01)
            continue

        # 정면 영역 크롭
        cropped_frame = local_frame[:, start_x:end_x]

        try:
            results = model.predict(cropped_frame, imgsz=320, conf=0.15, iou=0.40, verbose=False)
            
            # 💡 [최적화 핵심 1] 모델이 찾은 수많은 박스들을 '처음부터' 신뢰도 높은 순으로 정렬합니다.
            boxes = []
            for result in results:
                if result.boxes:
                    boxes.extend(result.boxes)
            
            # 신뢰도 내림차순 정렬
            boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)

            final_selected_obj = None

            # 💡 [최적화 핵심 2: 조기 종료 루프]
            # 모든 사물을 다 도는 것이 아니라, 가장 점수 높은 애들부터 순서대로 검사하다가
            # 조건에 맞는 '가장 확실한 놈 하나'가 걸리면 그 즉시 루프를 탈출(break)합니다.
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                raw_name = model.names[cls_id]
                class_name = clean_labels.get(raw_name, raw_name)

                # 보정치 계산
                if class_name in ['door', 'window']:
                    conf = min(conf + 0.20, 1.0)
                elif class_name in ['wardrobe', 'shelf']:
                    conf = conf - 0.15

                # 커트라인 통과 검사
                if class_name in class_conf_thresholds and conf < class_conf_thresholds[class_name]:
                    continue  # 통과 못 하면 다음 사물 검사

                # 🎯 필터를 통과한 가장 점수 높은 첫 번째 사물을 찾았으므로 확정!
                final_selected_obj = {'name': class_name, 'conf': conf}
                break  # ✨ 뒤에 남은 수십 개의 사물 연산은 통째로 생략하고 탈출 (CPU 대폭 절약)

            # 최종 확정된 단 하나의 사물만 처리
            if final_selected_obj:
                obj_name = final_selected_obj['name']
                if obj_name not in ["wall", "window", "tree"]:
                    current_time = time.time()
                    if (obj_name != last_spoken_state) or (current_time - last_spoken_time > AUDIO_COOLDOWN):
                        print(f"[라즈베리파이 확정] 🎯 {obj_name} (보정 신뢰도: {final_selected_obj['conf']:.2f})")
                        last_spoken_state = obj_name
                        last_spoken_time = current_time
            else:
                last_spoken_state = ""

        except Exception as e:
            print(f"추론 오류: {e}")

        time.sleep(0.01)

def run_pi_system():
    global latest_frame, is_running

    model = YOLOWorld('fixed_model.pt')

    clean_labels = {
        "a room door with a handle": "door", 
        "an entrance door": "door", 
        "a glass door": "door",
        "concrete curb on the street": "curb", "stairs leading downwards": "stairs", "stairs leading upwards": "stairs",
        "wheelchair ramp or slope": "ramp", "electric kick scooter": "scooter",
        "person": "person", "bicycle": "bicycle", "motorcycle": "motorcycle", "sink": "sink",
        "stop sign": "stop sign", "bench": "bench", "chair": "chair", "potted plant": "plant",
        "tv": "tv", "laptop": "laptop", "cell phone": "phone", "microwave": "microwave",
        "bollard": "bollard", "traffic cone": "cone", "utility pole": "pole", "water puddle": "puddle",
        
        "a storage shelf with racks": "shelf", 
        "stair": "stair", "tree": "tree", "car": "car", "bus": "bus", "truck": "truck",
        "a kitchen refrigerator": "refrigerator", 
        "a bed": "bed", "a dining table": "table", "a cardboard box": "box", "wall": "wall", "window": "window", 
        
        "a wooden wardrobe": "wardrobe",
        "clothes hanger rack": "hanger"
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 라즈베리파이 카메라는 열 수 없습니다.")
        return

    W, H = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    crop_w = 200
    start_x = (W - crop_w) // 2
    end_x = start_x + crop_w

    ai_thread = threading.Thread(
        target=inference_thread_func, 
        args=(model, clean_labels, (start_x, end_x)), 
        daemon=True
    )
    ai_thread.start()

    print("🚀 가벼워진 백그라운드 인지 시스템 구동 중... (Ctrl+C 종료)")

    try:
        while True:
            # 버퍼 비우기 (카메라 프레임 레이트 싱크 고정)
            for _ in range(2):
                cap.grab()
                
            ret, frame = cap.retrieve()
            if not ret:
                break

            with frame_lock:
                latest_frame = frame
            
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\n[시스템 중지] 종료 요청")
    finally:
        is_running = False
        ai_thread.join(timeout=0.5)
        cap.release()
        print("👋 안전하게 종료되었습니다.")

if __name__ == "__main__":
    run_pi_system()
