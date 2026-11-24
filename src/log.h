// keylime: logging. Every line goes to the console and to logs\keylime.log.
#pragma once
#include <cstdarg>
#include <cstdint>

namespace kl {

enum class Level { Trace, Info, Warn, Error };

void log_open(const wchar_t* path);
void logf(Level level, const char* area, const char* fmt, ...);
bool trace_enabled();

}  // namespace kl

#define KL_INFO(area, ...) ::kl::logf(::kl::Level::Info, area, __VA_ARGS__)
#define KL_WARN(area, ...) ::kl::logf(::kl::Level::Warn, area, __VA_ARGS__)
#define KL_ERROR(area, ...) ::kl::logf(::kl::Level::Error, area, __VA_ARGS__)
#define KL_TRACE(area, ...) \
    do { if (::kl::trace_enabled()) ::kl::logf(::kl::Level::Trace, area, __VA_ARGS__); } while (0)
