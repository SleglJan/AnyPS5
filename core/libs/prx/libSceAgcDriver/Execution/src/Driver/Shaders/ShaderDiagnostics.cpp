#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "ControlFlow/CalleeLinker.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Optimization/RequestMemoryView.hpp"
#include "ShaderCacheDirectory.hpp"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

void Driver::reportSkip(const char* kind, const std::string& what) {
    static std::mutex reportedMutex;
    static std::set<std::size_t> reported;
    static std::map<std::string, int> requestDumps;

    static const bool traceSkips = std::getenv("APS5_TRACE_SKIPS") != nullptr;
    static const int dumpLimit = [] { const char* text = std::getenv("APS5_SKIP_REQUEST_DUMPS"); return text ? std::atoi(text) : 8; }();
    static constexpr std::string_view marker = "\nRecompileRequest:\n";
    std::lock_guard lock(reportedMutex);
    std::string line;
    static bool announced = false;
    if (!announced) {
        announced = true;
        char text[160];
        std::snprintf(text, sizeof(text), "[gpu] skip reports: one write each, RecompileRequest dumps capped at %d per reason (APS5_SKIP_REQUEST_DUMPS)\n", dumpLimit);
        line = text;
    }
    const std::string prefix = "[gpu] skipped " + std::string(kind);
    if (reported.insert(std::hash<std::string>{}(what)).second) {
        const auto request = what.find(marker);
        if (request == std::string::npos) {
            line += prefix + ": " + what + "\n";
        } else {
            const auto reason = what.substr(0, request);
            const auto suffix = what.find(' ', request + marker.size());
            auto& dumps = requestDumps[reason];
            if (dumpLimit < 0 || dumps < dumpLimit) {
                ++dumps;
                line += prefix + ": " + what + "\n";
            } else {
                if (dumps == dumpLimit) {
                    ++dumps;
                    line += prefix + ": further RecompileRequest dumps for \"" + reason + "\" are left out\n";
                }
                line += prefix + ": " + reason + " (request left out)" + (suffix == std::string::npos ? std::string() : what.substr(suffix)) + "\n";
            }
        }
    } else if (traceSkips) {
        line += prefix + " again: " + what.substr(0, 100) + "\n";
    }
    if (!line.empty()) std::fwrite(line.data(), 1, line.size(), stderr);
}

namespace {

struct DumpedCallees {
    ShaderRecompiler::SrtRuntime runtime;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> code;
};

bool ReadDumpedCalleeCode(void* context, std::uint64_t address, std::span<std::uint32_t> words) {
    const auto bytes = words.size() * sizeof(std::uint32_t);
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address)), bytes)) return false;
    GuestMemory::Read(address, std::as_writable_bytes(words));
    static_cast<DumpedCallees*>(context)->code.emplace_back(address, address + bytes);
    return true;
}

bool ReadDumpedCalleeData(void* context, std::uint64_t address, std::span<std::uint32_t> words) {
    const auto& runtime = static_cast<const DumpedCallees*>(context)->runtime;
    for (std::size_t word = 0; word < words.size(); ++word) {
        if (!runtime.readMemory(runtime.userContext, address + word * sizeof(std::uint32_t), &words[word])) return false;
    }
    return true;
}

std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> CalleeCodeRegions(const ShaderRecompiler::RecompileRequest& request) {
    DumpedCallees callees;
    ShaderRecompiler::RequestMemoryView view(request.context.memory);
    const ShaderRecompiler::CalleeMemory memory{&callees, &ReadDumpedCalleeCode, &ReadDumpedCalleeData, nullptr};
    try {
        callees.runtime = view.MakeRuntime(request.context.userData, request.shader.codeAddress);
        static_cast<void>(ShaderRecompiler::LinkCallees(request, memory));
    } catch (const std::exception&) {
    }
    try {
        if (const auto split = ShaderRecompiler::SplitTableCalls(request, memory)) static_cast<void>(ShaderRecompiler::ReadTableCallTargets(request, *split, memory));
    } catch (const std::exception&) {
    }
    auto ranges = std::move(callees.code);
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    for (const auto& range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second) merged.back().second = std::max(merged.back().second, range.second);
        else merged.push_back(range);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> captured;
    for (const auto& region : request.context.memory) captured.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
    std::sort(captured.begin(), captured.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> missing;
    for (auto [begin, end] : merged) {
        for (const auto& [regionBegin, regionEnd] : captured) {
            if (regionEnd <= begin || regionBegin >= end) continue;
            if (regionBegin > begin) missing.emplace_back(begin, regionBegin);
            begin = std::max(begin, regionEnd);
            if (begin >= end) break;
        }
        if (begin < end) missing.emplace_back(begin, end);
    }
    std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> regions;
    for (const auto& [begin, end] : missing) {
        std::vector<std::byte> bytes(static_cast<std::size_t>(end - begin));
        GuestMemory::Read(begin, bytes);
        regions.emplace_back(begin, std::move(bytes));
    }
    return regions;
}

}

std::string Driver::dumpRequest(std::uint64_t address, const ShaderRecompiler::RecompileRequest& request) {
    static std::mutex dumpMutex;
    static std::set<std::uint64_t> dumped;
    static std::set<std::uint64_t> addresses;
    char name[96];
    std::lock_guard lock(dumpMutex);
    std::snprintf(name, sizeof(name), "shader_%llx.req", static_cast<unsigned long long>(address));
    try {
        const auto code = CalleeCodeRegions(request);
        std::vector<ShaderRecompiler::MemoryRegion> memory(request.context.memory.begin(), request.context.memory.end());
        for (const auto& [base, bytes] : code) memory.push_back({base, bytes});
        auto withCallees = request;
        withCallees.context.memory = memory;
        const auto text = ShaderRecompiler::RequestSerializer{}.Serialize(withCallees);
        const auto hash = ShaderRecompiler::HashBytes(std::as_bytes(std::span(text.data(), text.size())));
        if (!dumped.insert(hash).second) return name;
        if (!addresses.insert(address).second) std::snprintf(name, sizeof(name), "shader_%llx_%016llx.req", static_cast<unsigned long long>(address), static_cast<unsigned long long>(hash));
        if (std::FILE* file = std::fopen(name, "wb")) {
            std::fwrite(text.data(), 1, text.size(), file);
            std::fclose(file);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gpu] could not serialize request for 0x%llx: %s\n", static_cast<unsigned long long>(address), error.what());
    }
    return name;
}

}
