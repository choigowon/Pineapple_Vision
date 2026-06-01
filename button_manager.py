import os
import sys
import warnings

# ⚡ 자잘한 라이브러리 경고 및 pygame 로고 출력 원천 차단
warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLOWorld
import torch
import RPi.GPIO as GPIO
from gtts import gTTS
import pygame

# 🔥 형의 연산 성능 풀파워 사수 (스레드 2개 고정)
torch.set_num_threads(2)
GPIO.setwarnings(False)

BUTTON_PIN = 3
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

latest_frame = None
frame_lock = threading.Lock()
is_running = True
is_detecting = False  

current_zone = "UNKNOWN"  
zone_lock = threading.Lock()

AUDIO_DIR = "/home/pav/audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

# 💡 [C-Extension 하드웨어 오디오 가속기 초기화]
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
AUDIO_BANK = {} # 메모리(RAM)에 사운드 포인터를 상주시키는 뱅크

def pre_cache_audio(text):
    """프로그램 시작 시 딱 한 번만 gTTS를 생성하고, pygame Sound 객체로 메모리에 상주"""
    global AUDIO_BANK
    safe_name = text.replace(" ", "_")
    fast_path = os.path.join(AUDIO_DIR, f"{safe_name}_fast.mp3")
    
    if not os.path.exists(fast_path):
        try:
            raw_path = os.path.join(AUDIO_DIR, f"{safe_name}_raw.mp3")
            tts = gTTS(text=text, lang='ko', slow=False)
            tts.save(raw_path)
            # 배속 인코딩은 초기화 시점에 딱 한 번만 수행
            os.system(f"sox {raw_path} {fast_path} tempo 1.4 > /dev/null 2>&1")
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            return
            
    # ⚡ 핵심: MP3 파일을 읽어서 C 구조체 포인터로 메모리에 박아버림 (os.system 전면 폐기)
    if os.path.exists(fast_path) and text not in AUDIO_BANK:
        try:
            AUDIO_BANK[text] = pygame.mixer.Sound(fast_path)
        except Exception:
            pass

def play_voice_direct(text, interrupt=False):
    """OS 쉘을 전혀 거치지 않고, 메모리 주소에서 사운드 카드로 0.001초 만에 다이렉트 스트리밍"""
    global AUDIO_BANK
    if not is_detecting and text not in ["탐지 시작", "탐지 일시 정지", "기기를 종료합니다", "시스템 초기화 중입니다", "준비 완료"]:
        return

    if interrupt:
        pygame.mixer.stop() # ⚡ 현재 재생 중인 모든 오디오 하드웨어 채널 즉시 정지
        
    if text in AUDIO_BANK:
        AUDIO_BANK[text].play() # ⚡ 파이썬을 거치지 않고 C 라이브러리가 사운드 카드로 직접 출력
    else:
        # 예외 상황 대비 하드웨어 비프음
        os.system("aplay -q /usr/share/sounds/alsa/Front_Center.wav > /dev/null 2>&1")

# 메인 인퍼런스 엔진
def inference_thread_func(model, clean_labels, korean_names, crop_config):
    global latest_frame, is_running, current_zone, is_detecting

    last_spoken_state = ""
    last_spoken_time = 0
    AUDIO_COOLDOWN = 1.6  
    
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

    while is_running:
        if not is_detecting:
            last_spoken_state = ""
            time.sleep(0.1)
            continue

        with frame_lock:
            local_frame = latest_frame
            latest_frame = None  

        if local_frame is None:
            time.sleep(0.001)  
            continue

        cropped_frame = local_frame[:, start_x:end_x]

        try:
            results = model.predict(cropped_frame, imgsz=320, conf=0.15, iou=0.40, verbose=False)
            
            boxes = []
            for result in results:
                if result.boxes:
                    boxes.extend(result.boxes)
            
            # ⚡ [난간 오탐지 방지] 현재 프레임에 탐지된 모든 클래스명 임시 수집
            present_classes = []
            for b in boxes:
                r_id = int(b.cls[0])
                r_name = model.names[r_id]
                present_classes.append(clean_labels.get(r_name, r_name))
            
            boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
            final_selected_obj = None

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                raw_name = model.names[cls_id]
                class_name = clean_labels.get(raw_name, raw_name)

                with zone_lock:
                    if class_name in indoor_hints and conf > 0.45:
                        if current_zone != "INDOOR": current_zone = "INDOOR"
                    elif class_name in outdoor_hints and conf > 0.45:
                        if current_zone != "OUTDOOR": current_zone = "OUTDOOR"
                    if current_zone == "INDOOR" and class_name in outdoor_hints: conf -= 0.40  
                    elif current_zone == "OUTDOOR" and class_name in indoor_hints: conf -= 0.40  

                if class_name in ['door', 'handle']: 
                    conf = min(conf + 0.35, 1.0)  
                elif class_name == 'handrail':
                    # ⚡ 난간은 주변에 계단(stairs, stair)이 함께 포착될 때만 가중치 부여, 독립적일 땐 패널티
                    if ('stairs' in present_classes) or ('stair' in present_classes):
                        conf = min(conf + 0.35, 1.0)
                    else:
                        conf -= 0.30  # 단독으로 테이블 등을 난간으로 오탐지하는 것 억제
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
                        # ⚡ 텍스트 출력 제거 지침 반영 (print문 완전 삭제)
                        play_voice_direct(final_selected_obj, interrupt=False)
                        
                        last_spoken_state = final_selected_obj
                        last_spoken_time = current_time
            else:
                last_spoken_state = ""

        except Exception as e:
            pass

        time.sleep(0.005)

# 하드웨어 버튼 감지 스레드
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
            
            if pressed_time >= 3:
                play_voice_direct("기기를 종료합니다", interrupt=True)
                time.sleep(1.2)
                is_running = False
                os.system('sudo poweroff')
                break
            else:
                is_detecting = not is_detecting
                if is_detecting:
                    play_voice_direct("탐지 시작", interrupt=True)
                else:
                    play_voice_direct("탐지 일시 정지", interrupt=True)
        time.sleep(0.1)

def run_pi_system():
    global latest_frame, is_running

    clean_labels = {
        "automatic glass sliding door with silver metal frame": "door",
        "framed glass door panel for entrance": "door", 
        "room door with a handle": "door", 
        "door handle or door knob": "handle", 
        "pedestrian stairs with multiple continuous vertical steps, not a single road curb or flat ramp": "stairs", 
        "pedestrian stairs leading upwards with sequential levels": "stairs", 
        "electric kick scooter with a vertical handlebar and a flat board to stand on": "scooter", 
        "person": "person", "riding bicycle": "bicycle", "motorcycle with heavy engine": "motorcycle", 
        "bench": "bench", "furniture chair with backrest": "chair", "potted plant": "plant", "tv": "tv", 
        "laptop": "laptop", "cell phone": "phone", "microwave": "microwave", "bollard": "bollard", 
        "traffic cone": "cone", "utility pole": "pole", "water puddle": "puddle", 
        "vertical storage shelf with multiple grid racks for holding items, not a flat table": "shelf", 
        "stair": "stair", "tree": "tree", "passenger car on the road": "car", "large passenger bus": "bus", 
        "cargo truck": "truck", "kitchen refrigerator appliance": "refrigerator", "bed": "bed", 
        "flat dining table or desk supported by legs with a single flat surface for working, NO multiple shelves": "table", 
        "cardboard box": "box", "wall": "wall", "window fixed in a wall": "window", "wooden wardrobe": "wardrobe", 
        "clothes hanger rack": "hanger", "safety handrail or metallic grab bar mounted along stairs": "handrail"
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

    # ⚡ [하드웨어 뱅크 전수 로드] 부팅 시 메모리에 사운드 싹 다 올려버리기
    system_ment = ["시스템 초기화 중입니다", "준비 완료", "탐지 시작", "탐지 일시 정지", "기기를 종료합니다"]
    for ment in system_ment:
        pre_cache_audio(ment)
        
    for ko_val in korean_names.values():
        for pos in ["정면", "좌측", "우측"]:
            pre_cache_audio(f"{pos} {ko_val}")

    play_voice_direct("시스템 초기화 중입니다", interrupt=True)

    model = YOLOWorld('fixed_model.pt')

    dummy_img = np.zeros((240, 160, 3), dtype=np.uint8)
    model.predict(dummy_img, verbose=False)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    W, H = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    crop_w = 160
    start_x = (W - crop_w) // 2
    end_x = start_x + crop_w

    ai_thread = threading.Thread(target=inference_thread_func, args=(model, clean_labels, korean_names, (start_x, end_x)), daemon=True)
    ai_thread.start()

    btn_thread = threading.Thread(target=button_monitor_thread, daemon=True)
    btn_thread.start()

    play_voice_direct("준비 완료", interrupt=True)

    while is_running:
        ret, frame = cap.read()
        if not ret: break
        with frame_lock: 
            latest_frame = frame
        time.sleep(0.01)

    cap.release()
    GPIO.cleanup()

if __name__ == "__main__":
    run_pi_system()
