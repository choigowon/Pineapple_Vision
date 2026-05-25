#ifndef VOICE_MANAGER_H
#define VOICE_MANAGER_H

#include <pthread.h>
#include <sys/types.h>

#define MAX_TEXT_LEN 256
#define MAX_QUEUE_SIZE 50

// 큐 아이템 구조체
typedef struct {
    int priority; 
    char text[MAX_TEXT_LEN];
} MsgItem;

// VoiceManager 전역 구조체 정의
typedef struct {
    MsgItem queue[MAX_QUEUE_SIZE];
    int queue_size;
    
    char last_spoken_text[MAX_TEXT_LEN];
    double last_spoken_time;
    
    pid_t current_pid; 
    
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_t worker_thread;
} VoiceManager;

// 외부 파일(main.c)에서 공유해서 사용할 수 있도록 전역 변수 선언
extern VoiceManager vm;

// 외부에서 호출할 함수 인터페이스 선언
void init_voice_manager();
void speak(const char* text, int priority, double debounce_time);
void emergency_reset();
double get_current_time();

#endif // VOICE_MANAGER_H