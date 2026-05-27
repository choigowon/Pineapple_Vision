import os
os.environ["PYTHONUNBUFFERED"] = "1"  # 실시간 로그 방출 활성화

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLOWorld
import torch
import RPi.GPIO as GPIO

# 라즈베리파이 CPU 초기 구동 최적화 및 스레드 고정
torch.set_num_threads(2)

# GPIO 및 버튼 하드웨어 설정
BUTTON_PIN = 3
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 전역 공유 자원 및 플래그
latest_frame = None
frame_lock = threading.Lock()
is_running = True
is_detecting = False  # 💡 버튼으로 켜고 끌 탐지 제어 스위치 플래그

current_zone = "UNKNOWN"  
zone_lock = threading.Lock()

def speak_pure_audio(text):
    def _say():
        try:
            os.system(f"espeak -v ko+f3 -s 160 -a 200 \"{text}\" > /dev/null 2>&1")
        except Exception as e:
            pass
    threading.Thread(target=_say, daemon=True).start()

# 💡 형의 메인 인퍼런스 엔진 원본 로직 100% 복구
def inference_thread_func(model, clean_labels, korean_names, crop_config):
    global latest_frame, is_running, current_zone, is_detecting

    last_spoken_state = ""
    last_spoken_time = 0
    AUDIO_COOLDOWN = 1.3  
    
    start_x, end_x = crop_config
    crop_width = end_x - start_x

    class_conf_thresholds = {
        'stairs': 0.27, 'handrail': 0.18, 'door': 0.18, 'handle': 0.18, 
        'refrigerator': 0.55, 'scooter': 0.20, 'bicycle': 0.22, 'motorcycle': 0.25,     
        'bench': 0.50, 'table': 0.35, 'box': 0.35, 'window': 0.30,
        'wardrobe': 0.40, 'shelf': 0.45, 'chair': 0.35, 'person': 0.30, 
        'plant': 0.30, 'tv': 0.35, 'laptop': 0.35, 'phone': 0.30, 
        'microwave': 0.40, 'bollard': 0.35, 'cone': 0.35, 'pole': 0.35, 
        'puddle': 0.30, 'tree': 0.30, 'car': 0.35, 'bus': 0.35, 
        'truck': 0.35, 'bed': 0.40, 'hanger': 0.40, 'stair': 0.30
    }

    indoor_hints = ['refrigerator', 'bed', 'wardrobe', 'hanger', 'shelf', 'tv', 'laptop', 'microwave']
    outdoor_hints = ['scooter', 'bollard', 'cone', 'pole', 'puddle', 'tree', 'car', 'bus', 'truck']

    print(f"🧠 [라즈베리파이 AI 엔진] 가중치 및 한글 변환 매트릭스 구동 완료")

    while is_running:
        # 💡 탐지 스위치가 꺼져있으면(False) 연산을 수행하지 않고 대기하여 배터리 절약
        if not is_detecting:
            last_spoken_state = ""
            time.sleep(0.1)
            continue

        with frame_lock:
            local_frame = latest_frame
            latest_frame = None

        if local_frame is None:
            time.sleep(0.01)
            continue

        cropped_frame = local_frame[:, start_x:end_x]

        try:
            results = model.predict(cropped_frame, imgsz=320, conf=0.15, iou=0.40, verbose=False)
            
            boxes = []
            for result in results:
                if result.boxes:
                    boxes.extend(result.boxes)
            
            boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
            final_selected_obj = None

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                raw_name = model.names[cls_id]
                class_name = clean_labels.get(raw_name, raw_name)

                # 공간 인지 필터 로직
                with zone_lock:
                    if class_name in indoor_hints and conf > 0.45:
                        if current_zone != "INDOOR": current_zone = "INDOOR"
                    elif class_name in outdoor_hints and conf > 0.45:
                        if current_zone != "OUTDOOR": current_zone = "OUTDOOR"
                    if current_zone == "INDOOR" and class_name in outdoor_hints: conf -= 0.40  
                    elif current_zone == "OUTDOOR" and class_name in indoor_hints: conf -= 0.40  

                # 가중치 튜닝
                if class_name in ['door', 'handle', 'handrail']: 
                    conf = min(conf + 0.35, 1.0)  
                elif class_name in ['wardrobe', 'shelf']:
                    conf = conf - 0.20

                if class_name in class_conf_thresholds and conf < class_conf_thresholds[class_name]:
                    continue  

                obj_center_x = (x1 + x2) / 2.0
                if obj_center_x < (crop_width * 0.33): direction = "좌측"
                elif obj_center_x > (crop_width * 0.66): direction = "우측"
                else: direction = "정면"

                ko_name = korean_names.get(class_name, class_name)
                final_selected_obj = f"{direction} {ko_name}"
                break  

            if final_selected_obj:
                if "wall" not in final_selected_obj and "window" not in final_selected_obj:
                    current_time = time.time()
                    if (final_selected_obj != last_spoken_state) or (current_time - last_spoken_time > AUDIO_COOLDOWN):
                        print(f"[순정 즉시출력] 🎯 {final_selected_obj}")
                        speak_pure_audio(final_selected_obj)
                        last_spoken_state = final_selected_obj
                        last_spoken_time = current_time
            else:
                last_spoken_state = ""

        except Exception as e:
            pass

        time.sleep(0.01)

# 💡 하드웨어 버튼 감지 스레드 (백그라운드 상시 가동)
def button_monitor_thread():
    global is_detecting, is_running
    while is_running:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            start_time = time.time()
            
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.05)
                if time.time() - start_time > 3:
                    break
            
            pressed_time = time.time() - start_time
            
            # [종료] 3초 이상 꾹 누르면 시스템 파워오프
            if pressed_time >= 3:
                os.system('espeak -v ko+f3 -s 160 -a 200 "기기를 종료합니다."')
                is_running = False
                os.system('sudo poweroff')
                break
            # [토글] 짧게 누르면 탐지 Start / Stop 스위치 전환
            else:
                is_detecting = not is_detecting
                if is_detecting:
                    os.system('espeak -v ko+f3 -s 160 -a 200 "탐지 시작." &')
                    print("▶️ 탐지 활성화 (0초 가동 시작)")
                else:
                    os.system('espeak -v ko+f3 -s 160 -a 200 "탐지 일시 정지." &')
                    print("⏸️ 탐지 비활성화 (대기 상태)")
        time.sleep(0.1)

def run_pi_system():
    global latest_frame, is_running

    # 볼륨 강제 확보 및 초기화 사운드 송출
    os.system('amixer set Master 100% > /dev/null 2>&1')
    os.system('espeak -v ko+f3 -s 160 -a 200 "시스템 초기화 중입니다." &')

    print("⏳ [초기화] AI 모델 로딩 중...")
    model = YOLOWorld('fixed_model.pt')

    print("⚡ [초기화] 하드웨어 가속 및 웜업 연산 가동...")
    dummy_img = np.zeros((240, 160, 3), dtype=np.uint8)
    model.predict(dummy_img, imgsz=320, verbose=False)

    # 형이 세팅한 클린 라벨과 한글 매핑 100% 원본 복구
    clean_labels = {
        "automatic glass sliding door with silver metal frame": "door",
        "framed glass door panel for entrance": "door", 
        "room door with a handle": "door", 
        "door handle or door knob": "handle", 
        
        "pedestrian stairs with multiple continuous vertical steps, not a single road curb or flat ramp": "stairs", 
        "pedestrian stairs leading upwards with sequential levels": "stairs", 
        
        "electric kick scooter with a vertical handlebar and a flat board to stand on": "scooter", 
        "person": "person", 
        "riding bicycle": "bicycle", 
        "motorcycle with heavy engine": "motorcycle", 
        
        "bench": "bench", 
        "furniture chair with backrest": "chair", 
        "potted plant": "plant", "tv": "tv", "laptop": "laptop", "cell phone": "phone", "microwave": "microwave", "bollard": "bollard", 
        "traffic cone": "cone", "utility pole": "pole", "water puddle": "puddle", 
        
        "vertical storage shelf with multiple grid racks for holding items, not a flat table": "shelf", 
        "stair": "stair", "tree": "tree", 
        "passenger car on the road": "car", 
        "large passenger bus": "bus", 
        "cargo truck": "truck", 
        
        "kitchen refrigerator appliance": "refrigerator", 
        "bed": "bed", 
        "flat dining table or desk supported by legs with a single flat surface for working, NO multiple shelves": "table", 
        "cardboard box": "box", 
        "wall": "wall", 
        "window fixed in a wall": "window", 
        "wooden wardrobe": "wardrobe", 
        "clothes hanger rack": "hanger", 
        
        "safety handrail or metallic grab bar mounted along stairs": "handrail"
    }

    korean_names = {
        'door': '문', 'handle': '문 손잡이', 'shelf': '선반', 'wardrobe': '옷장', 'hanger': '행거',
        'table': '테이블', 'bench': '벤치', 'refrigerator': '냉장고', 'chair': '의자',
        'person': '사람', 'stairs': '계단', 'stair': '계단', 'tv': '티비', 'laptop': '노트북',
        'scooter': '킥보드', 'bicycle': '자전거', 'motorcycle': '오토바이', 
        'plant': '화분', 'phone': '핸드폰', 'microwave': '전자레인지', 'bollard': '볼라드', 
        'cone': '라바콘', 'pole': '전신주', 'puddle': '물웅덩이', 'tree': '나무', 
        'car': '자동차', 'bus': '버스', 'truck': '트럭', 'bed': '침대', 'box': '상자', 
        'handrail': '난간', 'window': '창문'
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    W, H = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    crop_w = 160
    start_x = (W - crop_w) // 2
    end_x = start_x + crop_w

    # 1. AI 인퍼런스 스레드 기동
    ai_thread = threading.Thread(
        target=inference_thread_func, 
        args=(model, clean_labels, korean_names, (start_x, end_x)), 
        daemon=True
    )
    ai_thread.start()

    # 2. 버튼 감지 스레드 기동
    btn_thread = threading.Thread(target=button_monitor_thread, daemon=True)
    btn_thread.start()

    # 시스템 대기 준비 완료 알림
    os.system('espeak -v ko+f3 -s 160 -a 200 "준비 완료." &')
    print("🚀 [라즈베리파이] 상시 대기 가동 성공. 버튼 입력을 기다립니다.")

    # 3. 메인 카메라 스트리밍 캡처 루프
    while is_running:
        for _ in range(2): 
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret: 
            break
        with frame_lock: 
            latest_frame = frame
        time.sleep(0.04)

    cap.release()
    GPIO.cleanup()

if __name__ == "__main__":
    run_pi_system()
