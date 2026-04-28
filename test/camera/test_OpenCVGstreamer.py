import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# 1. 캘리브레이션 데이터 로드
try:
    calib_data = np.load('camera_calib_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print("캘리브레이션 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("camera_calib_data.npz 파일이 없습니다. 캘리브레이션을 먼저 진행해주세요.")
    exit()

# 2. YOLOv8-Nano 모델 로드
# 추후 이 부분을 yolov8n_int8.tflite 로드 방식으로 변경하게 됩니다.
model = YOLO("yolov8n.pt") 

# 3. 멀티스레딩을 위한 전역 변수 설정
frame_to_infer = None
lock = threading.Lock()
running = True

# 4. GStreamer 파이프라인 최적화 설정
# USB 카메라(/dev/video0) 기준. 
# max-buffers=1 drop=true 옵션으로 가장 최신 프레임만 유지하여 지연 시간 최소화
def gstreamer_pipeline():
    return (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! appsink max-buffers=1 drop=true"
    )

# 5. 카메라 캡처 및 왜곡 보정 스레드
def capture_thread():
    global frame_to_infer, running
    
    # GStreamer 백엔드 사용
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    # GStreamer 지원이 안 될 경우 일반 웹캠 모드로 대체 (대기 지연 발생 가능)
    if not cap.isOpened():
        print("GStreamer 파이프라인 실패. 일반 V4L2 모드로 전환합니다.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 버퍼 최소화

    # 최적 카메라 매트릭스 계산 (한 번만 수행)
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 광각 렌즈 왜곡 펴기 (Undistort)
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # 스레드 안전하게 최신 프레임 갱신
        with lock:
            frame_to_infer = undistorted_frame.copy()

    cap.release()

# 6. 스레드 시작
cap_thread = threading.Thread(target=capture_thread)
cap_thread.start()

print("실시간 추론을 시작합니다. (종료하려면 'q'를 누르세요)")

# 7. 메인 루프 (AI 추론 및 화면 출력)
prev_time = time.time()

try:
    while True:
        current_frame = None
        
        with lock:
            if frame_to_infer is not None:
                current_frame = frame_to_infer.copy()
        
        if current_frame is not None:
            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # YOLOv8 추론 (화면 출력을 위해 텐서 변환 최적화 등 추가 필요)
            results = model.predict(current_frame, verbose=False, imgsz=320) # 파이4 부하 감소를 위해 해상도 축소
            
            # 결과 그리기
            annotated_frame = results[0].plot()
            
            # FPS 화면 표시
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Pineapple Vision - Real-time Inference", annotated_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
            
except KeyboardInterrupt:
    running = False

finally:
    running = False
    cap_thread.join()
    cv2.destroyAllWindows()