#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Stride = 48;
constexpr std::uint32_t Records = 12;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t MaxGroups = 8;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Format32SInt = 21;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Format11_11_10UInt = 34;
constexpr std::uint32_t Format8_8_8_8UNorm = 56;
constexpr std::uint32_t Type2D = 9;

TestGuest::Array<float, Width * Height> FloatTexels;
TestGuest::Array<float, Width * Height> OtherFloatTexels;
TestGuest::Array<std::uint32_t, Width * Height> UintTexels;
TestGuest::Array<std::int32_t, Width * Height> SintTexels;
TestGuest::Array<std::uint32_t, Width * Height> PackedTexels;
TestGuest::Array<std::uint32_t, Records * Stride / 4> Table;
TestGuest::Array<std::uint32_t, 64> Keys;
TestGuest::Array<std::uint32_t, MaxGroups * 128 * 4> Output;
TestGuest::Array<std::uint32_t, MaxGroups * 128 * 4> OtherOutput;
TestGuest::Array<std::uint32_t, 32> Srt;
TestGuest::Array<std::uint32_t, 5 * Width * Height> RuntimeTexels;
TestGuest::Vector<std::uint32_t> RuntimeBlocks(0x8000u);
constexpr std::uint32_t PointerBytes = MaxGroups * 16u;
constexpr std::uint32_t ArgumentWord = 1024u;

alignas(256) constexpr std::array<std::uint32_t, 30> SampleTableCode{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080700u, 0xfa000020u, 0xf4080800u, 0xfa000030u,
    0x8f108202u, 0xf4200444u, 0x20000000u, 0x9312b011u, 0xf4280902u, 0x24000010u, 0x360200bfu, 0x7e020d01u,
    0x060202f0u, 0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c8f08u, 0x01090401u, 0x7e120202u,
    0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 22> SampleDirectCode{
    0xf4080700u, 0xfa000020u, 0xf4080800u, 0xfa000030u, 0xf4080900u, 0xfa000040u, 0x360200bfu, 0x7e020d01u,
    0x060202f0u, 0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c8f08u, 0x01090401u, 0x7e120202u,
    0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 23> LoadTableCode{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080700u, 0xfa000020u, 0x8f108202u, 0xf4200444u,
    0x20000000u, 0x9312b011u, 0xf42c0802u, 0x24000000u, 0x360200bfu, 0x7e040280u, 0xf0008f08u, 0x00090401u,
    0x7e120202u, 0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 25> LoadTableCutCode{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080700u, 0xfa000020u, 0x8f108202u, 0xf4200444u,
    0x20000000u, 0x9312b011u, 0xf42c0802u, 0x24000000u, 0x360200bfu, 0x7e040280u, 0x7da400ffu, 0x00001000u,
    0xf0008f08u, 0x00090401u, 0x7e120202u, 0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u,
    0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 17> LoadDirectCode{
    0xf4080700u, 0xfa000020u, 0xf4080800u, 0xfa000030u, 0xf4080900u, 0xfa000040u, 0x360200bfu, 0x7e040280u,
    0xf0008f08u, 0x00090401u, 0x7e120202u, 0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u,
    0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 22> GatherDirectCode{
    0xf4080700u, 0xfa000020u, 0xf4080800u, 0xfa000030u, 0xf4080900u, 0xfa000040u, 0x360200bfu, 0x7e020d01u,
    0x060202f0u, 0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf11c8108u, 0x01090401u, 0x7e120202u,
    0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 16> QueryDirectCode{
    0xf4080700u, 0xfa000020u, 0xf4080800u, 0xfa000030u, 0xf4080900u, 0xfa000040u, 0x7e020280u, 0xf0388f08u,
    0x00090401u, 0x7e120202u, 0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 29> RuntimeSampleCode{
    0xf4080100u, 0xfa000000u, 0xf4080700u, 0xfa000020u, 0x8f108402u, 0xf4240282u, 0x20000000u, 0xf4080b05u,
    0xfa000020u, 0xf4080805u, 0xfa000000u, 0xf42c0916u, 0xfa000020u, 0x360200bfu, 0x7e020d01u, 0x060202f0u,
    0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c8f08u, 0x01090401u, 0x7e120202u, 0x34101286u,
    0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 29> RuntimeGatherCode{
    0xf4080100u, 0xfa000000u, 0xf4080700u, 0xfa000020u, 0x8f108402u, 0xf4240282u, 0x20000000u, 0xf4080b05u,
    0xfa000020u, 0xf4080805u, 0xfa000000u, 0xf42c0916u, 0xfa000020u, 0x360200bfu, 0x7e020d01u, 0x060202f0u,
    0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf11c8108u, 0x01090401u, 0x7e120202u, 0x34101286u,
    0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 24> RuntimeLoadCode{
    0xf4080100u, 0xfa000000u, 0xf4080700u, 0xfa000020u, 0x8f108402u, 0xf4240282u, 0x20000000u, 0xf4080b05u,
    0xfa000020u, 0xf4080805u, 0xfa000000u, 0xf42c0916u, 0xfa000020u, 0x360200bfu, 0x7e040280u, 0xf0008f08u,
    0x00090401u, 0x7e120202u, 0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 23> RuntimeQueryCode{
    0xf4080100u, 0xfa000000u, 0xf4080700u, 0xfa000020u, 0x8f108402u, 0xf4240282u, 0x20000000u, 0xf4080b05u,
    0xfa000020u, 0xf4080805u, 0xfa000000u, 0xf42c0916u, 0xfa000020u, 0x7e020280u, 0xf0388f08u, 0x00090401u,
    0x7e120202u, 0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 4> PointClamp{0x92u, (4u * 256u) << 12u, 0u, 0u};
constexpr std::array<std::uint32_t, 4> LinearClamp{0x92u, (4u * 256u) << 12u, (1u << 20u) | (1u << 22u), 0u};

std::array<std::uint32_t, 4> Image(const void* texels, std::uint32_t format, std::uint32_t type = Type2D) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((Width - 1u) & 3u) << 30u), ((Width - 1u) >> 2u) | ((Height - 1u) << 14u) | (1u << 31u), 0xfacu | (type << 28u)};
}

std::array<std::uint32_t, 4> Buffer(const void* base, std::uint32_t bytes) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base));
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void SetRecord(std::uint32_t record, const std::array<std::uint32_t, 4>& image) {
    auto* words = Table.data() + record * (Stride / 4u);
    std::fill(words, words + Stride / 4u, 0u);
    std::copy(PointClamp.begin(), PointClamp.end(), words);
    std::copy(image.begin(), image.end(), words + 4);
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "cannot open a temporary file for stderr");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        _dup2(_fileno(file), _fileno(stderr));
#else
        saved = dup(fileno(stderr));
        dup2(fileno(file), fileno(stderr));
#endif
    }
    std::string Finish() {
        std::fflush(stderr);
#ifdef _WIN32
        _dup2(saved, _fileno(stderr));
        _close(saved);
#else
        dup2(saved, fileno(stderr));
        close(saved);
#endif
        std::rewind(file);
        std::string text;
        char buffer[4096];
        for (std::size_t read = 0; (read = std::fread(buffer, 1, sizeof(buffer), file)) != 0;) text.append(buffer, read);
        std::fclose(file);
        std::fwrite(text.data(), 1, text.size(), stderr);
        return text;
    }

private:
    std::FILE* file = nullptr;
    int saved = -1;
};

struct Outcome {
    TestGuest::Vector<std::uint32_t> words;
    std::string log;
    std::uint64_t variantId = 0;
    std::uint32_t tableElements = 0;
    std::size_t tableRanges = 0;
    AgcDriver::Graphics::DeviceImageTable::Counts before;
    AgcDriver::Graphics::DeviceImageTable::Counts after;
};

ShaderRecompiler::RecompileResult Compile(const AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> span, std::span<const ShaderRecompiler::MemoryRegion> memory) {
    const auto rootAddress = reinterpret_cast<std::uintptr_t>(Srt.data());
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(rootAddress), static_cast<std::uint32_t>(rootAddress >> 32u)};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(span.data()), span, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    return ShaderRecompiler::Recompile(request);
}

Outcome Execute(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> span, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint32_t groups) {
    std::fill(Output.begin(), Output.end(), 0xdeadbeefu);
    const auto result = Compile(device, span, memory);
    Outcome outcome;
    outcome.variantId = result.PipelineVariantId();
    outcome.tableRanges = result.imageTableRanges.size();
    for (const auto& ranges : result.imageTableReadRanges) outcome.tableRanges += ranges.size();
    for (const auto& binding : result.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) outcome.tableElements = binding.count;
    }
    outcome.before = device.DeviceImageCounts();
    StderrCapture capture;
    try {
        device.Dispatch(result, groups, 1, 1, reinterpret_cast<std::uintptr_t>(span.data()));
        device.WaitIdle();
    } catch (...) {
        static_cast<void>(capture.Finish());
        throw;
    }
    outcome.log = capture.Finish();
    outcome.after = device.DeviceImageCounts();
    outcome.words.assign(Output.begin(), Output.begin() + groups * Lanes * 4u);
    return outcome;
}

template <std::size_t CodeWords>
std::vector<ShaderRecompiler::MemoryRegion> TableMemory(const std::array<std::uint32_t, CodeWords>& code, bool tableInMemory, std::span<std::uint32_t> written = Output) {
    const auto table = Buffer(Table.data(), static_cast<std::uint32_t>(decltype(Table)::ByteSize));
    const auto keys = Buffer(Keys.data(), static_cast<std::uint32_t>(decltype(Keys)::ByteSize));
    const auto output = Buffer(written.data(), static_cast<std::uint32_t>(written.size_bytes()));
    std::copy(table.begin(), table.end(), Srt.begin());
    std::copy(keys.begin(), keys.end(), Srt.begin() + 4);
    std::copy(output.begin(), output.end(), Srt.begin() + 8);
    std::copy(PointClamp.begin(), PointClamp.end(), Srt.begin() + 12);
    const std::span<const std::uint32_t> span(code);
    std::vector<ShaderRecompiler::MemoryRegion> memory{
        {reinterpret_cast<std::uintptr_t>(span.data()), std::as_bytes(span)},
        {reinterpret_cast<std::uintptr_t>(Srt.data()), std::as_bytes(std::span(Srt))},
        {reinterpret_cast<std::uintptr_t>(Keys.data()), std::as_bytes(std::span(Keys))},
    };
    if (tableInMemory) memory.push_back({reinterpret_cast<std::uintptr_t>(Table.data()), std::as_bytes(std::span(Table))});
    return memory;
}

template <std::size_t CodeWords>
Outcome Run(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, CodeWords>& code, std::uint32_t groups, bool tableInMemory = true) {
    const auto memory = TableMemory(code, tableInMemory);
    return Execute(device, std::span<const std::uint32_t>(code), memory, groups);
}

template <std::size_t CodeWords>
Outcome RunDirect(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, CodeWords>& code, const std::array<std::uint32_t, 4>& image, const std::array<std::uint32_t, 4>& sampler = PointClamp) {
    const auto output = Buffer(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    std::copy(output.begin(), output.end(), Srt.begin() + 8);
    std::copy(sampler.begin(), sampler.end(), Srt.begin() + 12);
    std::fill(Srt.begin() + 16, Srt.begin() + 24, 0u);
    std::copy(image.begin(), image.end(), Srt.begin() + 16);
    const std::span<const std::uint32_t> span(code);
    const std::array<ShaderRecompiler::MemoryRegion, 2> memory{{
        {reinterpret_cast<std::uintptr_t>(span.data()), std::as_bytes(span)},
        {reinterpret_cast<std::uintptr_t>(Srt.data()), std::as_bytes(std::span(Srt))},
    }};
    return Execute(device, span, memory, 1);
}

template <std::size_t CodeWords>
Outcome RunRuntime(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, CodeWords>& code, std::uint32_t groups) {
    const auto pointers = Buffer(RuntimeBlocks.data(), PointerBytes);
    const auto output = Buffer(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    std::copy(pointers.begin(), pointers.end(), Srt.begin());
    std::copy(output.begin(), output.end(), Srt.begin() + 8);
    const std::span<const std::uint32_t> span(code);
    const std::array<ShaderRecompiler::MemoryRegion, 2> memory{{
        {reinterpret_cast<std::uintptr_t>(span.data()), std::as_bytes(span)},
        {reinterpret_cast<std::uintptr_t>(Srt.data()), std::as_bytes(std::span(Srt))},
    }};
    return Execute(device, span, memory, groups);
}

void SetArgument(std::uint32_t record, const std::array<std::uint32_t, 4>& sampler, const std::array<std::uint32_t, 4>& image) {
    auto* words = RuntimeBlocks.data() + ArgumentWord + record * 64u;
    std::fill(words, words + 64, 0u);
    std::copy(sampler.begin(), sampler.end(), words);
    const auto heap = Buffer(words + 16, 64u);
    std::copy(heap.begin(), heap.end(), words + 8);
    std::copy(image.begin(), image.end(), words + 24);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(words));
    RuntimeBlocks[record * 4u] = static_cast<std::uint32_t>(address);
    RuntimeBlocks[record * 4u + 1u] = static_cast<std::uint32_t>(address >> 32u);
}

TestGuest::Vector<std::uint32_t> Group(const Outcome& outcome, std::uint32_t group) {
    return TestGuest::Vector<std::uint32_t>(outcome.words.begin() + group * Lanes * 4u, outcome.words.begin() + (group + 1u) * Lanes * 4u);
}

bool Faulted(const Outcome& outcome) {
    return outcome.log.find("image table:") != std::string::npos;
}

bool ZeroGroup(const Outcome& outcome, std::uint32_t group) {
    const auto words = Group(outcome, group);
    return std::all_of(words.begin(), words.end(), [](std::uint32_t word) { return word == 0u; });
}

bool ZeroOrUntouched(const Outcome& outcome) {
    return std::all_of(outcome.words.begin(), outcome.words.end(), [](std::uint32_t word) { return word == 0u || word == 0xdeadbeefu; });
}

void RequireRed(const Outcome& outcome, std::uint32_t group, const std::function<std::uint32_t(std::uint32_t)>& expected, const std::string& what) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto actual = outcome.words[(group * Lanes + lane) * 4u];
        Require(actual == expected(lane % Width), what + ": lane " + std::to_string(lane) + " is " + Hex(actual) + ", expected " + Hex(expected(lane % Width)));
    }
}

std::uint64_t NewMisses(const Outcome& outcome) {
    return outcome.after.misses - outcome.before.misses;
}

void Fill() {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto index = y * Width + x;
            FloatTexels[index] = static_cast<float>(x) + 100.0f * static_cast<float>(y) + 0.25f;
            OtherFloatTexels[index] = -static_cast<float>(x) - 0.5f;
            UintTexels[index] = 0x1000u + x + 0x100u * y;
            SintTexels[index] = -3 * static_cast<std::int32_t>(x + 1u) - 1000 * static_cast<std::int32_t>(y);
            PackedTexels[index] = (x << 22u) | ((y + 1u) << 11u) | (x + 7u);
        }
    }
}

void RunFirstUse(AgcDriver::VulkanDevice& device) {
    Fill();
    Table = {};
    const std::array<std::array<std::uint32_t, 4>, 5> images{
        Image(FloatTexels.data(), Format32Float),
        Image(UintTexels.data(), Format32UInt),
        Image(SintTexels.data(), Format32SInt),
        Image(UintTexels.data(), Format8_8_8_8UNorm),
        Image(PackedTexels.data(), Format11_11_10UInt),
    };
    for (std::uint32_t record = 0; record < images.size(); ++record) SetRecord(record, images[record]);
    Keys = {};
    for (std::uint32_t group = 0; group < 6u; ++group) Keys[group] = group;
    const auto first = Run(device, LoadTableCode, 6);
    Require(!Faulted(first), "device image table: the first use of valid entries faulted:\n" + first.log);
    for (std::uint32_t group = 0; group < 6u; ++group) Require(ZeroGroup(first, group), "device image table: the first use of record " + std::to_string(group) + " did not read zeros");
    Require(NewMisses(first) == 5u * Lanes && first.after.dropped == first.before.dropped, "device image table: the first use did not record one miss per lane of the five non-null records (" + std::to_string(NewMisses(first)) + ")");
    Require(first.tableElements == 0u && first.tableRanges == 0u, "device image table: the build bound table elements or captured table ranges");
    const auto second = Run(device, LoadTableCode, 6);
    Require(!Faulted(second), "device image table: valid entries faulted after the drain:\n" + second.log);
    Require(NewMisses(second) == 0u && second.after.distinct - second.before.distinct == 5u && second.after.inserted - second.before.inserted == 5u, "device image table: the drain did not insert the five distinct records, or the second use missed again");
    for (std::uint32_t group = 0; group < 5u; ++group) {
        const auto control = RunDirect(device, LoadDirectCode, images[group]);
        Require(Group(second, group) == Group(control, 0), "device image table: record " + std::to_string(group) + " differs from the direct binding of the same words");
    }
    RequireRed(second, 0, [](std::uint32_t x) { return std::bit_cast<std::uint32_t>(FloatTexels[x]); }, "device image table: float texels");
    RequireRed(second, 1, [](std::uint32_t x) { return UintTexels[x]; }, "device image table: uint texels");
    RequireRed(second, 2, [](std::uint32_t x) { return static_cast<std::uint32_t>(SintTexels[x]); }, "device image table: sint texels");
    RequireRed(second, 4, [](std::uint32_t x) { return x + 7u; }, "device image table: the red channel of an 11_11_10 texture");
    Require(ZeroGroup(second, 5), "device image table: a null record did not read zeros");
    const auto third = Run(device, LoadTableCode, 6);
    Require(NewMisses(third) == 0u && Group(third, 0) == Group(second, 0) && third.variantId == first.variantId, "device image table: a steady-state use missed, changed its texels or its pipeline");

    const auto sampledFirst = Run(device, SampleTableCode, 3);
    for (std::uint32_t group = 0; group < 3u; ++group) Require(ZeroGroup(sampledFirst, group), "device image table: the first sample of record " + std::to_string(group) + " did not return zeros");
    Require(NewMisses(sampledFirst) == 3u * Lanes, "device image table: sampling through a direct sampler did not miss once per lane");
    const auto sampled = Run(device, SampleTableCode, 3);
    Require(!Faulted(sampled) && NewMisses(sampled) == 0u, "device image table: sampling after the drain faulted or missed:\n" + sampled.log);
    for (std::uint32_t group = 0; group < 3u; ++group) {
        const auto control = RunDirect(device, SampleDirectCode, images[group]);
        Require(Group(sampled, group) == Group(control, 0), "device image table: sampled record " + std::to_string(group) + " differs from the direct binding of the same words");
    }
}

void RunChanges(AgcDriver::VulkanDevice& device) {
    Fill();
    Table = {};
    SetRecord(0, Image(FloatTexels.data(), Format32Float));
    SetRecord(1, Image(UintTexels.data(), Format32UInt));
    Keys = {};
    Keys[1] = 1u;
    static_cast<void>(Run(device, LoadTableCode, 2));
    const auto warm = Run(device, LoadTableCode, 2);
    RequireRed(warm, 0, [](std::uint32_t x) { return std::bit_cast<std::uint32_t>(FloatTexels[x]); }, "device image table changes: float texels before the change");

    SetRecord(0, Image(OtherFloatTexels.data(), Format32Float));
    const auto changed = Run(device, LoadTableCode, 2);
    Require(changed.variantId == warm.variantId, "device image table changes: other table contents built another pipeline");
    Require(ZeroGroup(changed, 0) && NewMisses(changed) == Lanes, "device image table changes: a record that changed its T# did not miss once per lane");
    RequireRed(changed, 1, [](std::uint32_t x) { return UintTexels[x]; }, "device image table changes: an unchanged record beside a changed one");
    const auto settled = Run(device, LoadTableCode, 2);
    RequireRed(settled, 0, [](std::uint32_t x) { return std::bit_cast<std::uint32_t>(OtherFloatTexels[x]); }, "device image table changes: the new T#'s texels");

    OtherFloatTexels[5] = 1234.5f;
    const auto written = Run(device, LoadTableCode, 2);
    Require(NewMisses(written) == 0u && written.after.refreshed > written.before.refreshed, "device image table changes: texels the CPU wrote were not refreshed in place");
    RequireRed(written, 0, [](std::uint32_t x) { return std::bit_cast<std::uint32_t>(OtherFloatTexels[x]); }, "device image table changes: texels after a CPU write");

    const auto unread = Run(device, LoadTableCode, 2, false);
    Require(!Faulted(unread) && unread.variantId == warm.variantId, "device image table changes: a request without the table's memory faulted or built another pipeline:\n" + unread.log);
    RequireRed(unread, 0, [](std::uint32_t x) { return std::bit_cast<std::uint32_t>(OtherFloatTexels[x]); }, "device image table changes: texels read through a table the capture never read");
}

void RunRuntimeDescriptors(AgcDriver::VulkanDevice& device) {
    Fill();
    constexpr auto texels = Width * Height;
    std::copy_n(reinterpret_cast<const std::uint32_t*>(FloatTexels.data()), texels, RuntimeTexels.begin());
    std::copy_n(UintTexels.begin(), texels, RuntimeTexels.begin() + texels);
    std::copy_n(reinterpret_cast<const std::uint32_t*>(SintTexels.data()), texels, RuntimeTexels.begin() + 2 * texels);
    std::copy_n(UintTexels.begin(), texels, RuntimeTexels.begin() + 3 * texels);
    std::copy_n(PackedTexels.begin(), texels, RuntimeTexels.begin() + 4 * texels);
    const std::array<std::array<std::uint32_t, 4>, 5> images{
        Image(RuntimeTexels.data(), Format32Float),
        Image(RuntimeTexels.data() + texels, Format32UInt),
        Image(RuntimeTexels.data() + 2 * texels, Format32SInt),
        Image(RuntimeTexels.data() + 3 * texels, Format8_8_8_8UNorm),
        Image(RuntimeTexels.data() + 4 * texels, Format11_11_10UInt),
    };
    std::fill(RuntimeBlocks.begin(), RuntimeBlocks.end(), 0u);
    for (std::uint32_t record = 0; record < images.size(); ++record) SetArgument(record, PointClamp, images[record]);
    const auto first = RunRuntime(device, RuntimeLoadCode, 5);
    Require(!Faulted(first), "run-time descriptors: the first load of valid T#s faulted:\n" + first.log);
    for (std::uint32_t group = 0; group < 5u; ++group) Require(ZeroGroup(first, group), "run-time descriptors: the first load through record " + std::to_string(group) + " did not read zeros");
    Require(NewMisses(first) == 5u * Lanes, "run-time descriptors: the first load did not miss once per lane (" + std::to_string(NewMisses(first)) + ")");
    Require(first.tableElements == 0u && first.tableRanges == 0u, "run-time descriptors: the build bound table elements or captured table ranges");
    const auto loaded = RunRuntime(device, RuntimeLoadCode, 5);
    Require(!Faulted(loaded) && NewMisses(loaded) == 0u && loaded.after.inserted - loaded.before.inserted == 5u, "run-time descriptors: the drain did not insert the five T#s, or the second load missed:\n" + loaded.log);
    for (std::uint32_t group = 0; group < 5u; ++group) Require(Group(loaded, group) == Group(RunDirect(device, LoadDirectCode, images[group]), 0), "run-time descriptors: the load through record " + std::to_string(group) + " differs from the direct binding of its T#");
    RequireRed(loaded, 2, [](std::uint32_t x) { return static_cast<std::uint32_t>(SintTexels[x]); }, "run-time descriptors: sint texels");

    const auto queriedFirst = RunRuntime(device, RuntimeQueryCode, 5);
    Require(!Faulted(queriedFirst) && NewMisses(queriedFirst) == 5u * Lanes, "run-time descriptors: the first query did not miss once per lane");
    const auto queried = RunRuntime(device, RuntimeQueryCode, 5);
    Require(!Faulted(queried) && NewMisses(queried) == 0u, "run-time descriptors: the second query faulted or missed:\n" + queried.log);
    for (std::uint32_t group = 0; group < 5u; ++group) Require(Group(queried, group) == Group(RunDirect(device, QueryDirectCode, images[group]), 0), "run-time descriptors: the query through record " + std::to_string(group) + " differs from the direct binding of its T#");
    Require(queried.words[0] == Width && queried.words[1] == Height, "run-time descriptors: the query did not return the T#'s size");

    const auto sampled = [&](const std::array<std::uint32_t, 29>& runtimeCode, const std::array<std::uint32_t, 22>& directCode, const std::string& what) {
        const auto missed = RunRuntime(device, runtimeCode, 3);
        Require(!Faulted(missed) && NewMisses(missed) == 3u * Lanes, "run-time descriptors: the first " + what + " through a run-time S# did not miss once per lane:\n" + missed.log);
        for (std::uint32_t group = 0; group < 3u; ++group) Require(ZeroGroup(missed, group), "run-time descriptors: the first " + what + " through record " + std::to_string(group) + " did not return zeros");
        const auto hit = RunRuntime(device, runtimeCode, 3);
        Require(!Faulted(hit) && NewMisses(hit) == 0u, "run-time descriptors: the second " + what + " faulted or missed:\n" + hit.log);
        for (std::uint32_t group = 0; group < 3u; ++group) Require(Group(hit, group) == Group(RunDirect(device, directCode, images[group]), 0), "run-time descriptors: the " + what + " through record " + std::to_string(group) + " differs from the direct binding of its T# and S#");
    };
    sampled(RuntimeSampleCode, SampleDirectCode, "sample");
    sampled(RuntimeGatherCode, GatherDirectCode, "gather");

    SetArgument(3, LinearClamp, images[0]);
    const auto other = RunRuntime(device, RuntimeSampleCode, 4);
    Require(!Faulted(other) && NewMisses(other) == Lanes && ZeroGroup(other, 3) && !ZeroGroup(other, 0), "run-time descriptors: another S# beside a known T# did not miss once per lane and read zeros:\n" + other.log);
    const auto linear = RunRuntime(device, RuntimeSampleCode, 4);
    Require(!Faulted(linear) && NewMisses(linear) == 0u && linear.after.inserted - linear.before.inserted == 1u, "run-time descriptors: the T# with another S# was not inserted as its own entry:\n" + linear.log);
    Require(Group(linear, 3) == Group(RunDirect(device, SampleDirectCode, images[0], LinearClamp), 0), "run-time descriptors: the sample through the second S# differs from the direct binding of that S#");

    auto unnormalized = PointClamp;
    unnormalized[0] |= 1u << 15u;
    SetArgument(4, unnormalized, images[0]);
    const auto unread = RunRuntime(device, RuntimeSampleCode, 5);
    Require(!Faulted(unread) && ZeroGroup(unread, 4) && NewMisses(unread) == Lanes, "run-time descriptors: the first use of an unnormalized run-time S# faulted, sampled or did not miss:\n" + unread.log);
    const auto accepted = RunRuntime(device, RuntimeSampleCode, 5);
    Require(!Faulted(accepted) && NewMisses(accepted) == 0u && accepted.after.rejected == accepted.before.rejected, "run-time descriptors: an unnormalized run-time S# on a single-level 2D T# faulted, missed again or was rejected:\n" + accepted.log);
    RequireRed(accepted, 4, [](std::uint32_t) { return std::bit_cast<std::uint32_t>(FloatTexels[0]); }, "run-time descriptors: a sample through an unnormalized run-time S# at coordinates within the first texel");
    Require(Group(accepted, 4) == Group(RunDirect(device, SampleDirectCode, images[0], unnormalized), 0), "run-time descriptors: the sample through an unnormalized run-time S# differs from the direct binding of that S#");

    auto wrapped = unnormalized;
    wrapped[0] &= ~7u;
    SetArgument(4, wrapped, images[0]);
    const auto wrapUnread = RunRuntime(device, RuntimeSampleCode, 5);
    Require(!Faulted(wrapUnread) && ZeroGroup(wrapUnread, 4) && NewMisses(wrapUnread) == Lanes, "run-time descriptors: the first use of an unnormalized run-time S# that wraps faulted, sampled or did not miss:\n" + wrapUnread.log);
    const auto wrapAccepted = RunRuntime(device, RuntimeSampleCode, 5);
    Require(!Faulted(wrapAccepted) && NewMisses(wrapAccepted) == 0u && wrapAccepted.after.rejected == wrapAccepted.before.rejected, "run-time descriptors: an unnormalized run-time S# that wraps (sampled as clamp-to-edge) faulted, missed again or was rejected:\n" + wrapAccepted.log);
    RequireRed(wrapAccepted, 4, [](std::uint32_t) { return std::bit_cast<std::uint32_t>(FloatTexels[0]); }, "run-time descriptors: a sample through an unnormalized run-time S# that wraps at coordinates within the first texel");
    Require(Group(wrapAccepted, 4) == Group(RunDirect(device, SampleDirectCode, images[0], wrapped), 0), "run-time descriptors: the sample through an unnormalized run-time S# that wraps differs from the direct binding of that S#");
}

void RunRejections(AgcDriver::VulkanDevice& device) {
    Fill();
    Table = {};
    SetRecord(0, Image(FloatTexels.data(), Format32Float));
    SetRecord(7, {0x1234u, 0x5678u, 0x9abcu, 0x0000ffacu});
    Keys = {};
    Keys[0] = 7u;
    const auto cut = Run(device, LoadTableCutCode, 1);
    Require(!Faulted(cut) && NewMisses(cut) == 0u, "device image table rejections: an invalid record read with EXEC clear faulted or missed:\n" + cut.log);
    const auto first = Run(device, LoadTableCode, 1);
    Require(!Faulted(first) && ZeroGroup(first, 0) && NewMisses(first) == Lanes, "device image table rejections: the first use of an invalid record faulted, read texels or did not miss:\n" + first.log);
    const auto second = Run(device, LoadTableCode, 1);
    Require(second.after.rejected == second.before.rejected + 1u && NewMisses(second) == 0u, "device image table rejections: the invalid record was not rejected once by the drain");
    Require(second.log.find("[device-images]") != std::string::npos && second.log.find("not an image") != std::string::npos, "device image table rejections: the drain did not report the rejection:\n" + second.log);
    Require(Faulted(second) && second.log.find("not an image") != std::string::npos && second.log.find("00001234") != std::string::npos && ZeroOrUntouched(second), "device image table rejections: a rejected record selected with EXEC set did not fault with its words and reason, or read texels:\n" + second.log);
    const auto again = Run(device, LoadTableCutCode, 1);
    Require(!Faulted(again) && NewMisses(again) == 0u, "device image table rejections: a rejected record read with EXEC clear faulted or missed:\n" + again.log);
}

}

void RunTwoQueues(AgcDriver::VulkanDevice& device) {
    Fill();
    Table = {};
    const std::array<std::array<std::uint32_t, 4>, 5> images{
        Image(PackedTexels.data(), Format32UInt),
        Image(SintTexels.data(), Format32UInt),
        Image(FloatTexels.data(), Format32UInt),
        Image(OtherFloatTexels.data(), Format32UInt),
        Image(UintTexels.data(), Format32SInt),
    };
    for (std::uint32_t record = 0; record < images.size(); ++record) SetRecord(record, images[record]);
    Keys = {};
    for (std::uint32_t group = 0; group < images.size(); ++group) Keys[group] = group;
    std::fill(Output.begin(), Output.end(), 0xdeadbeefu);
    std::fill(OtherOutput.begin(), OtherOutput.end(), 0xdeadbeefu);
    const std::span<const std::uint32_t> span(LoadTableCode);
    const auto program = reinterpret_cast<std::uintptr_t>(span.data());
    const auto groups = static_cast<std::uint32_t>(images.size());
    const auto first = Compile(device, span, TableMemory(LoadTableCode, true));
    const auto before = device.DeviceImageCounts();
    std::exception_ptr failure;
    std::thread compute([&] {
        try {
            AgcDriver::GuestMemory::TagGpuLockThread(0x20);
            device.Dispatch(first, groups, 1, 1, program);
        } catch (...) {
            failure = std::current_exception();
        }
    });
    compute.join();
    if (failure) std::rethrow_exception(failure);
    const auto second = Compile(device, span, TableMemory(LoadTableCode, true, OtherOutput));
    device.Dispatch(second, groups, 1, 1, program);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.ReapRecorded();
    }
    device.WaitIdle();
    const auto after = device.DeviceImageCounts();
    Require(after.misses - before.misses == groups * Lanes, "device image table: of two queues' dispatches whose CPU pre-pass named none of their T#s, the second missed keys the first's drained misses had inserted before it was recorded (" + std::to_string(after.misses - before.misses) + " miss records, not " + std::to_string(groups * Lanes) + ")");
}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        RunFirstUse(*device);
        RunChanges(*device);
        RunRejections(*device);
        RunRuntimeDescriptors(*device);
        RunTwoQueues(*device);
        std::puts("device image table tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
