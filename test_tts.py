import time
from tts import VoiceManager, check_under_voltage

def run_test():
    # 1. 인스턴스 생성
    print("=== TTS 모듈 테스트 시작 ===")
    voice = VoiceManager()

    # 2. 기본 음성 출력 테스트
    print("테스트 1: 기본 출력")
    voice.speak("시스템 테스트를 시작합니다.", priority=2)
    time.sleep(2) # 음성이 나올 시간을 줍니다.

    # 3. 우선순위 큐 테스트 (중요!)
    # 일반 정보들을 먼저 넣고, 바로 뒤에 긴급 메시지를 넣었을 때 
    # 긴급 메시지가 먼저(또는 중간에) 새치기해서 나오는지 확인합니다.
    print("테스트 2: 우선순위 큐 (일반 정보 여러 개 입력 후 긴급 입력)")
    voice.speak("첫 번째 사물입니다.", priority=3)
    voice.speak("두 번째 사물입니다.", priority=3)
    voice.speak("세 번째 사물입니다.", priority=3)
    
    # 순서상 뒤에 넣었지만, priority가 1이므로 큐에서 가장 먼저 뽑힙니다.
    voice.speak("긴급 상황 발생! 전압을 확인하세요.", priority=1)
    
    # 4. 중복 방지(Debounce) 테스트
    print("테스트 3: 중복 방지 (똑같은 말을 짧은 시간에 여러 번 요청)")
    # '사람이 있습니다'를 3번 연속 요청해도 한 번만 나와야 정상입니다.
    for _ in range(3):
        voice.speak("사람이 있습니다.", priority=3, debounce_time=3)
        time.sleep(0.1)

    # 모든 음성이 끝날 때까지 잠시 대기
    print("\n모든 음성 요청 완료. 5초 후 테스트를 종료합니다.")
    time.sleep(5)

if __name__ == "__main__":
    run_test()