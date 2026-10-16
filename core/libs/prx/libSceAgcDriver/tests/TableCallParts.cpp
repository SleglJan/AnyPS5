#include "Recompiler.hpp"
#include "ControlFlow/CalleeLinker.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint64_t Srt = 0x1100010000ull;
constexpr std::uint64_t Table = 0x1100020000ull;
constexpr std::uint64_t Code = 0x1100030000ull;
constexpr std::uint64_t Output = 0x1100050000ull;

void Require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::uint32_t Low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address);
}

std::uint32_t High(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 32u);
}

struct GuestMemory {
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
};

bool ReadGuestMemory(void* context, std::uint64_t address, std::span<std::uint32_t> words) {
    for (const auto& [base, region] : static_cast<const GuestMemory*>(context)->regions) {
        if (address >= base && (address - base) / 4u + words.size() <= region.size()) {
            std::copy_n(region.begin() + static_cast<std::ptrdiff_t>((address - base) / 4u), words.size(), words.begin());
            return true;
        }
    }
    return false;
}

std::vector<std::uint32_t> Caller(bool wave32) {
    std::vector<std::uint32_t> words{0x36020081u};
    if (wave32) words.insert(words.end(), {0xbe9e037eu, 0x7e280501u, 0xbe9f037eu, 0x7da40214u});
    else words.insert(words.end(), {0xbe9e047eu, 0x7e280501u, 0x7d840214u, 0xbea0246au});
    words.insert(words.end(), {0x8f148414u, 0xf4080400u, 0xfa000000u, 0xbf8cc07fu, 0xf4280208u, 0x28000000u, 0xbf8cc07fu, 0xbe8e0308u, 0xbe8f0309u, 0xbe8e210eu});
    if (wave32) words.insert(words.end(), {0x8a7e7e1fu, 0xbf89fff1u, 0xbefe031eu});
    else words.insert(words.end(), {0x8afe7e20u, 0xbf89fff1u, 0xbefe041eu});
    words.insert(words.end(), {0xf4080600u, 0xfa000010u, 0x34060082u, 0xbf8cc07fu, 0xe0701000u, 0x80060203u, 0xbf810000u});
    return words;
}

struct Module {
    std::vector<std::uint32_t> words;
    std::size_t atomics = 0;
    std::size_t pointers = 0;
    std::size_t loops = 0;
    std::size_t calls = 0;
};

Module Parse(std::vector<std::uint32_t> words) {
    Module module;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = words[cursor] & 0xffffu;
        if (count == 0 || cursor + count > words.size()) throw std::runtime_error("truncated SPIR-V instruction");
        module.atomics += op == spv::OpAtomicIAdd ? 1u : 0u;
        module.pointers += op == spv::OpConvertUToPtr ? 1u : 0u;
        module.loops += op == spv::OpLoopMerge ? 1u : 0u;
        module.calls += op == spv::OpFunctionCall ? 1u : 0u;
        cursor += count;
    }
    module.words = std::move(words);
    return module;
}

class Parts {
public:
    Parts(bool wave32, const GuestMemory& memory) : caller(Caller(wave32)), memory(memory) {
        const auto view = [](const std::vector<std::uint32_t>& words) { return std::as_bytes(std::span(words)); };
        for (const auto& [base, words] : memory.regions) regions.push_back({base, view(words)});
        request.shader = {ShaderStage::Compute, 0x1100040000ull, caller, 0, {}};
        request.context.waveSize = wave32 ? 32u : 64u;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.memory = regions;
        request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {true, false, false}, false, 1u, {}};
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32;
        request.target.bdaAbiVersion = 1;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.layout.pushConstantSizeBytes = 128;
        request.useCache = false;
        const CalleeMemory callees{const_cast<GuestMemory*>(&memory), &ReadGuestMemory, &ReadGuestMemory, nullptr};
        auto found = SplitTableCalls(request, callees);
        Require(found.has_value(), "a waterfall table call was not split");
        split = std::move(*found);
        targets = ReadTableCallTargets(request, split, callees);
        extended.assign(userData.begin(), userData.end());
        extended.resize(extended.size() + ContinuationAbi::UserDataWords, 0u);
    }

    Module Compile(const SplitCode& part, ContinuationAbi::Role role, std::uint32_t entrySite) const {
        auto source = request;
        source.shader.code = part.code;
        source.context.userData = extended;
        if (role == ContinuationAbi::Role::Continuation) {
            auto compute = *request.context.compute;
            compute.numThreads = {request.context.waveSize, 1u, 1u};
            source.context.compute = compute;
        }
        source.context.continuation = TableCallShape(split, part, role, entrySite);
        const auto result = Recompile(source);
        Require(!result.spirv.empty(), "a table-call part did not recompile");
        return Parse(result.spirv.Copy());
    }

    std::vector<std::uint32_t> caller;
    std::array<std::uint32_t, 2> userData{Low(Srt), High(Srt)};
    std::vector<std::uint32_t> extended;
    const GuestMemory& memory;
    std::vector<MemoryRegion> regions;
    RecompileRequest request{};
    TableCallSplit split;
    std::vector<TableCallTarget> targets;

private:
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
};

void Save(const char* directory, const std::string& name, const Module& module) {
    if (directory == nullptr) return;
    std::ofstream file(std::string(directory) + "/" + name + ".spv", std::ios::binary);
    file.write(reinterpret_cast<const char*>(module.words.data()), static_cast<std::streamsize>(module.words.size() * sizeof(std::uint32_t)));
    Require(file.good(), "could not write " + name);
}

void Check(bool wave32, const char* directory) {
    const std::vector<std::uint32_t> first{0x4a040085u, 0xbe80200eu};
    const std::vector<std::uint32_t> second{0x4a0400c0u, 0xbe80200eu};
    std::vector<std::uint32_t> page(1024u, 0u);
    std::copy(first.begin(), first.end(), page.begin());
    std::copy(second.begin(), second.end(), page.begin() + 0x200);
    const std::vector<std::uint32_t> srt{Low(Table), High(Table) | (16u << 16u), 4u, 0x31016facu, Low(Output), High(Output) & 0xffffu, 0x100u, 0x31016facu};
    const std::vector<std::uint32_t> records{Low(Code), High(Code), 0x00c0ffeeu, 0x12u, Low(Code + 0x800u), High(Code + 0x800u), 0x0badf00du, 0x13u, Low(Code), High(Code), 0x0badf00du, 0x14u, 0u, 0u, 0u, 0u};
    const GuestMemory memory{{{Srt, srt}, {Table, records}, {Code, page}}};
    const Parts parts(wave32, memory);
    const auto what = std::string(wave32 ? "wave32" : "wave64");
    Require(parts.split.rounds == parts.request.context.waveSize && parts.split.wavesPerGroup == (wave32 ? 2u : 1u), what + ": the waterfall call's rounds or waves per group are wrong");
    Require(parts.targets.size() == 3u && parts.targets[0].target == Code && parts.targets[2].target == Code + 0x800u, what + ": the call table's records were not read as three bins");
    Require(std::ranges::binary_search(parts.split.userDataScalars, 0u) && std::ranges::binary_search(parts.split.userDataScalars, 1u), what + ": the SRT pointer is not restored from user data");
    const auto head = parts.Compile(parts.split.head, ContinuationAbi::Role::Head, 0u);
    Save(directory, what + "-head", head);
    Require(head.atomics >= 1u && head.pointers >= 1u && head.loops >= 1u, what + ": the head does not bin its waves");
    for (std::size_t bin = 0; bin < parts.targets.size(); bin += 2u) {
        const auto& target = parts.targets[bin];
        const auto continuation = parts.Compile(TableCallContinuation(parts.split, target), ContinuationAbi::Role::Continuation, target.site);
        Save(directory, what + "-continuation-" + std::to_string(bin), continuation);
        Require(continuation.pointers >= 1u && continuation.atomics >= 1u && continuation.calls >= 1u, what + ": a continuation does not restore, call its callee and bin again");
    }
}

}

int main(int argc, char** argv) {
    try {
        const char* directory = argc > 1 ? argv[1] : nullptr;
        Check(true, directory);
        Check(false, directory);
    } catch (const std::exception& error) {
        std::cerr << "TableCallParts: " << error.what() << '\n';
        return 1;
    }
    std::puts("table call part tests passed");
    return 0;
}
