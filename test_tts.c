#include <stdio.h>
#include <unistd.h>
#include "tts.h" // 음성 매니저 헤더 불러오기

int main() {
    // 1. 시스템 기동 시 최초 1회 초기화
    init_voice_manager();
    
    printf("🚀 [Pineapple Vision] 시스템 루프 구동\n");
    
    // 2. 일반 안내 방송 요청 (우선순위 3, 디바운스 2초)
    speak("카메라 스트리밍을 시작합니다.", 3, 2.0);
    sleep(2);
    
    speak("전방에 의자가 있습니다.", 3, 2.0);
    
    // 동일 문장 연속 호출 테스트 (디바운스 트리거로 인해 씹혀야 정상)
    speak("전방에 의자가 있습니다.", 3, 2.0); 
    sleep(1);
    
    // 3. 비상 상황 모사 인터랩트 (말하던 의자 안내 끊고 즉시 출력)
    printf("\n⚠️ [위험 상황 발생 이벤트 캐치]\n");
    speak("긴급. 전방에 돌발 장애물이 있습니다.", 1, 0.0);
    
    // 자식 프로세스 처리 대기 후 종료
    sleep(4);
    printf("시스템 종료.\n");
    
    return 0;
}