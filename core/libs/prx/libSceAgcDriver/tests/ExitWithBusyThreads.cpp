#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {

constexpr unsigned Children = 8;
constexpr unsigned ChurnThreads = 4;
constexpr std::uint64_t ChurnRounds = 20000;
constexpr DWORD ChildExitLimitMs = 10000;

std::atomic<std::uint64_t> rounds{0};

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

[[noreturn]] void Churn() {
    for (;;) {
        std::mutex mutex;
        std::condition_variable changed;
        std::unique_lock lock(mutex);
        changed.wait_until(lock, std::chrono::system_clock::time_point{});
        rounds.fetch_add(1, std::memory_order_relaxed);
    }
}

int Child() {
    Require(LoadLibraryW(L"libSceAgcDriver.prx") != nullptr, "cannot load the driver library");
    for (unsigned thread = 0; thread < ChurnThreads; ++thread) std::thread(Churn).detach();
    while (rounds.load(std::memory_order_relaxed) < ChurnRounds) std::this_thread::yield();
    return 0;
}

bool ChildExits(const std::wstring& executable) {
    auto commandLine = L"\"" + executable + L"\" child";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    Require(CreateProcessW(executable.data(), commandLine.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process) != 0, "cannot create the child");
    const bool exited = WaitForSingleObject(process.hProcess, ChildExitLimitMs) == WAIT_OBJECT_0;
    DWORD exitCode = 1;
    const bool reported = exited && GetExitCodeProcess(process.hProcess, &exitCode) != 0;
    if (!exited) TerminateProcess(process.hProcess, 1);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    Require(!exited || (reported && exitCode == 0), "the child failed");
    return exited;
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "child") == 0) return Child();
        std::wstring executable(MAX_PATH, L'\0');
        const auto length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
        Require(length > 0 && length < executable.size(), "cannot read the test executable's path");
        executable.resize(length);
        for (unsigned child = 0; child < Children; ++child) {
            Require(ChildExits(executable), "a process that loaded the driver library did not finish exiting while other threads were busy with condition variables");
        }
        std::cout << "exit with busy threads tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
