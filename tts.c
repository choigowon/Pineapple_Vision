#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>
#include "tts.h"

// 전역 변수 실체화
VoiceManager vm;

double get_current_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (ts.tv_nsec / 1000000000.0);
}

void queue_put(VoiceManager *v, int priority, const char *text) {
    if (v->queue_size >= MAX_QUEUE_SIZE) return; 
    
    MsgItem item;
    item.priority = priority;
    strncpy(item.text, text, MAX_TEXT_LEN - 1);
    item.text[MAX_TEXT_LEN - 1] = '\0';
    
    int i = v->queue_size - 1;
    while (i >= 0 && v->queue[i].priority > priority) {
        v->queue[i + 1] = v->queue[i];
        i--;
    }
    v->queue[i + 1] = item;
    v->queue_size++;
}

MsgItem queue_get(VoiceManager *v) {
    MsgItem item = v->queue[0];
    for (int i = 1; i < v->queue_size; i++) {
        v->queue[i - 1] = v->queue[i];
    }
    v->queue_size--;
    return item;
}

void queue_clear(VoiceManager *v) {
    v->queue_size = 0;
}

void clear_low_priority(VoiceManager *v) {
    int write_idx = 0;
    for (int i = 0; i < v->queue_size; i++) {
        if (v->queue[i].priority == 1) {
            v->queue[write_idx++] = v->queue[i];
        }
    }
    v->queue_size = write_idx;
}

void stop_current_voice_internal() {
    if (vm.current_pid > 0) {
        kill(vm.current_pid, SIGKILL); 
        int status;
        waitpid(vm.current_pid, &status, 0); 
        printf("[시스템] 현재 음성 중단 (PID: %d)\n", vm.current_pid);
        vm.current_pid = -1;
    }
}

void* speech_worker(void* arg) {
    VoiceManager *v = (VoiceManager*)arg;
    
    while (1) {
        pthread_mutex_lock(&v->lock);
        while (v->queue_size == 0) {
            pthread_cond_wait(&v->cond, &v->lock);
        }
        if (v->queue_size > 3) {
            clear_low_priority(v);
        }
        MsgItem item = queue_get(v);
        pthread_mutex_unlock(&v->lock);
        
        pid_t pid = fork();
        if (pid == 0) { // 자식
            freopen("/dev/null", "w", stdout);
            freopen("/dev/null", "w", stderr);
            execlp("espeak", "espeak", "-v", "ko", "-s", "220", "-p", "50", item.text, (char *)NULL);
            exit(1);
        } else if (pid > 0) { // 부모
            pthread_mutex_lock(&v->lock);
            v->current_pid = pid;
            pthread_mutex_unlock(&v->lock);
            
            int status;
            waitpid(pid, &status, 0); 
            
            pthread_mutex_lock(&v->lock);
            v->current_pid = -1;
            pthread_mutex_unlock(&v->lock);
        }
        usleep(50000); 
    }
    return NULL;
}

void speak(const char* text, int priority, double debounce_time) {
    if (text == NULL || strlen(text) == 0) return;
    double curr_time = get_current_time();
    
    pthread_mutex_lock(&vm.lock);
    if (priority == 3 && strcmp(text, vm.last_spoken_text) == 0) {
        if (curr_time - vm.last_spoken_time < debounce_time) {
            pthread_mutex_unlock(&vm.lock);
            return;
        }
    }
    if (priority == 1) {
        stop_current_voice_internal(); 
        queue_clear(&vm);              
    }
    queue_put(&vm, priority, text);
    if (priority == 3) {
        strncpy(vm.last_spoken_text, text, MAX_TEXT_LEN - 1);
        vm.last_spoken_text[MAX_TEXT_LEN - 1] = '\0';
        vm.last_spoken_time = curr_time;
    }
    pthread_cond_signal(&vm.cond);
    pthread_mutex_unlock(&vm.lock);
}

void emergency_reset() {
    pthread_mutex_lock(&vm.lock);
    stop_current_voice_internal();
    queue_clear(&vm);
    printf("시스템 음성 대기열 초기화 완료\n");
    pthread_mutex_unlock(&vm.lock);
}

void init_voice_manager() {
    vm.queue_size = 0;
    vm.current_pid = -1;
    vm.last_spoken_text[0] = '\0';
    vm.last_spoken_time = 0.0;
    pthread_mutex_init(&vm.lock, NULL);
    pthread_cond_init(&vm.cond, NULL);
    pthread_create(&vm.worker_thread, NULL, speech_worker, &vm);
}