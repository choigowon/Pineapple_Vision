import subprocess
import queue
import threading
import time

def check_under_voltage():
    try:
        # 라즈베리 파이 전용 명령어 실행
        result = subprocess.check_output(['vcgencmd', 'get_throttled']).decode()
        status = int(result.split('=')[1], 16)
        
        # 0x1: 현재 저전압 발생 중, 0x10000: 최근에 저전압 발생했었음
        if status & 0x1:
            return "긴급. 전압이 낮습니다. 보조배터리를 점검하세요."
        return None
    except:
        # PC(Windows) 환경에서는 에러가 나므로 무시
        return None
        
class VoiceManager:
    def __init__(self):
        self.msg_queue = queue.PriorityQueue()
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        
        # 음성 전용 스레드 시작
        self.worker = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker.start()

    def _speech_worker(self):
        while True:
            # 1. 큐에 메시지가 너무 쌓이면 일반 정보(Priority 3) 삭제 (실시간성 유지)
            if self.msg_queue.qsize() > 3:
                self._clear_low_priority()

            # 큐에서 메시지 대기
            priority, text = self.msg_queue.get()
            
            try:
                # subprocess.run을 사용하여 한 문장이 끝날 때까지 대기 (순서 보장)
                # Popen 대신 run을 쓰는 이유는 문장이 겹치지 않게 하기 위함입니다.
                subprocess.run([
                    'espeak', 
                    '-v', 'ko',    # 한국어 음성
                    '-s', '220',   # 속도 (기본 175)
                    '-p', '50',    # 피치
                    text
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            except Exception as e:
                print(f"espeak 에러: {e}")
            
            finally:
                self.msg_queue.task_done()
            
            time.sleep(0.05)

    def _clear_low_priority(self):
        """밀린 일반 정보는 버리고 긴급 정보만 남김"""
        temp = []
        while not self.msg_queue.empty():
            item = self.msg_queue.get()
            if item[0] == 1: # 긴급 메시지만 보관
                temp.append(item)
        for i in temp:
            self.msg_queue.put(i)

    def speak(self, text, priority=3, debounce_time=2):
        """
        priority 1: 긴급 (거리 경고, 전압 저하)
        priority 2: 모드 전환
        priority 3: 일반 정보 (객체 인식, OCR)
        """
        if not text: return
        
        curr_time = time.time()
        
        # 중복 방지 (일반 정보에만 적용)
        if priority == 3 and text == self.last_spoken_text:
            if curr_time - self.last_spoken_time < debounce_time:
                return

        self.msg_queue.put((priority, text))
        
        if priority == 3:
            self.last_spoken_text = text
            self.last_spoken_time = curr_time
    
