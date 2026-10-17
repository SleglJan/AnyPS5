#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

#ifdef _WIN32
std::int64_t PrivateBytes() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) return static_cast<std::int64_t>(counters.PrivateUsage);
    return 0;
}

std::unique_ptr<AgcDriver::VulkanDevice> OpenRequired() {
    auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "the Vulkan device opened before did not open again");
    return device;
}

void GateAfterDestroy() {
    {
        const auto device = OpenRequired();
        static_cast<void>(AgcDriver::CreationGate::RamSideBytes());
    }
    {
        const AgcDriver::VulkanDevice compiler{AgcDriver::VulkanDevice::CompileOnly{}};
    }
    static_cast<void>(AgcDriver::CreationGate::RamSideBytes());
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    static_cast<void>(AgcDriver::CreationGate::RamSideBytes());
}

void SourcesFollowTheRecordingDevice() {
    {
        const auto first = OpenRequired();
    }
    const auto device = OpenRequired();
    {
        const AgcDriver::VulkanDevice compiler{AgcDriver::VulkanDevice::CompileOnly{}};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    auto* residency = AgcDriver::Graphics::ResidencyDriver::Active();
    Require(residency != nullptr, "the second device has no active residency driver");
    static_cast<void>(residency->Budget());
    constexpr std::int64_t Bytes = std::int64_t{512} << 20u;
    const auto privateBefore = PrivateBytes();
    const auto ramBefore = static_cast<std::int64_t>(AgcDriver::CreationGate::RamSideBytes());
    const auto block = device->ContinuationBlock(static_cast<std::size_t>(Bytes));
    const auto privateAfter = PrivateBytes();
    const auto ramAfter = static_cast<std::int64_t>(AgcDriver::CreationGate::RamSideBytes());
    Require(block != nullptr, "the device-local buffer was not created");
    const auto grewPrivate = privateAfter - privateBefore;
    const auto grewRam = ramAfter - ramBefore;
    Require(grewRam <= grewPrivate - Bytes / 2, "after a device was replaced, a 512 MiB device-local buffer of the second device grew the creation gate's RAM side by " + std::to_string(grewRam >> 20) + " MiB while the private bytes grew by " + std::to_string(grewPrivate >> 20) + " MiB: the gate does not measure the second device");
    static_cast<void>(residency->Budget());
}

int RunChild(const char* scenario) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    try {
        if (std::strcmp(scenario, "gate-after-destroy") == 0) GateAfterDestroy();
        else if (std::strcmp(scenario, "sources-follow") == 0) SourcesFollowTheRecordingDevice();
        else throw std::runtime_error(std::string("unknown scenario ") + scenario);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << scenario << ": " << error.what() << '\n';
        return 1;
    }
}

DWORD RunScenario(const char* scenario) {
    std::wstring executable(MAX_PATH, L'\0');
    executable.resize(GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size())));
    std::wstring commandLine = L"\"" + executable + L"\" --scenario ";
    for (const char* c = scenario; *c != '\0'; ++c) commandLine.push_back(static_cast<wchar_t>(*c));
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    Require(CreateProcessW(executable.c_str(), commandLine.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process) != 0, "cannot start the scenario child");
    const auto waited = WaitForSingleObject(process.hProcess, 90000);
    if (waited != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 0xdead);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return code;
}
#endif

}

int main(int argc, char** argv) {
#ifndef _WIN32
    std::puts("skipped: the scenarios run in child processes started on Windows");
    return VulkanTestSkipped;
#else
    if (argc == 3 && std::strcmp(argv[1], "--scenario") == 0) return RunChild(argv[2]);
    try {
        if (!OpenVulkanTestDevice()) return VulkanTestSkipped;
        for (const char* scenario : {"gate-after-destroy", "sources-follow"}) {
            const auto code = RunScenario(scenario);
            Require(code == 0, std::string("scenario ") + scenario + " ended with exit code 0x" + [&] { char text[16]; std::snprintf(text, sizeof(text), "%lx", static_cast<unsigned long>(code)); return std::string(text); }() + ": a query used the handles of a destroyed device");
        }
        std::puts("device source lifetime tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
