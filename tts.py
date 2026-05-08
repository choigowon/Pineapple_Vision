import subprocess
import queue
import threading
import time
        
class VoiceManager:
    def __init__(self):
        self.msg_queue = queue.PriorityQueue()
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        
        # 현재 실행 중인 음성 프로세스를 저장하는 변수
        self.current_process = None
        
        self.worker = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker.start()

    def _speech_worker(self):
        while True:
            if self.msg_queue.qsize() > 3:
                self._clear_low_priority()

            priority, text = self.msg_queue.get()
            
            try:
                # 음성 출력 프로세스 시작 (Popen을 사용하여 비동기 실행)
                self.current_process = subprocess.Popen([
                    'espeak', '-v', 'ko', '-s', '220', '-p', '50', text
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 프로세스가 종료될 때까지 대기
                self.current_process.wait()
                
            except Exception as e:
                print(f"espeak 에러: {e}")
            finally:
                self.current_process = None
                self.msg_queue.task_done()
            
            time.sleep(0.05)

    def _clear_low_priority(self):
        temp = []
        while not self.msg_queue.empty():
            item = self.msg_queue.get()
            if item[0] == 1:
                temp.append(item)
        for i in temp:
            self.msg_queue.put(i)

    def stop_current_voice(self):
        """현재 말하고 있는 음성을 즉시 중단"""
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate() # 프로세스 강제 종료
            print("[시스템] 현재 음성 중단")

    def speak(self, text, priority=3, debounce_time=2):
        """
        priority 1(긴급)이 들어오면 기존 음성을 끊고 즉시 출력 시도
        """
        if not text: return
        
        curr_time = time.time()
        
        # 중복 방지 (일반 정보에만 적용)
        if priority == 3 and text == self.last_spoken_text:
            if curr_time - self.last_spoken_time < debounce_time:
                return

        # 1. 긴급 상황(Priority 1)인 경우 현재 말하는 것을 즉시 중단
        if priority == 1:
            self.stop_current_voice()
            # 큐를 비워버림으로써 긴급 상황이 가장 먼저 나오게 함
            while not self.msg_queue.empty():
                try:
                    self.msg_queue.get_nowait()
                    self.msg_queue.task_done()
                except queue.Empty:
                    break

        # 2. 큐에 메시지 삽입
        self.msg_queue.put((priority, text))
        
        if priority == 3:
            self.last_spoken_text = text
            self.last_spoken_time = curr_time

    def emergency_reset(self):
        """모든 음성 대기열 삭제 및 현재 음성 강제 종료"""
        self.stop_current_voice()
        while not self.msg_queue.empty():
            try:
                self.msg_queue.get_nowait()
                self.msg_queue.task_done()
            except queue.Empty:
                break
        print("시스템 음성 대기열 초기화 완료")
