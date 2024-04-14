#ifndef LOGGING_H
#define LOGGING_H

#include <string>

typedef void (*LogCallback)(const char* str);

class Log {
    private:
        LogCallback log_callback;

    public:
        Log() {
            this->log_callback = [](const char* str) {};
        }

        Log(LogCallback callback) : log_callback(callback) {}

        void operator()(const std::string& message) {
            log_callback(message.c_str());
        }
};

#endif