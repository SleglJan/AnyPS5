#include "DrawSubmitFixture.hpp"

#ifdef _WIN32

namespace {

using namespace DrawSubmit;

constexpr std::uint32_t Extent = 64;
constexpr std::uint32_t KeyBytes = Extent * Extent * 4u * 4u / 256u;
constexpr std::uint32_t KeyLanes = KeyBytes / 4u;
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t DccEnable = 0x10000000u;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t Uncompressed = 0xffffffffu;
constexpr std::array<std::array<float, 4>, 3> Corners{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

Header VertexHeader;
Header PixelHeader;

const Program& CornerVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Indexed(BufferLoadDwordX4, 10, 5, 8)
              << WaitAll
              << Export(PositionTarget, 10, false)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& FlatPixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Vop1(VMovB32, 0, InlineOne)
              << Vop1(VMovB32, 1, InlineInteger(0))
              << Vop1(VMovB32, 2, InlineInteger(0))
              << Vop1(VMovB32, 3, InlineOne)
              << Export(ColorTarget, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& CopyKeysProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(2), 0)
              << Mubuf(BufferLoadDword, 2, 1, 0)
              << WaitAll
              << Mubuf(BufferStoreDword, 2, 1, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

bool Resident(std::uint64_t address) {
    bool resident = false;
    NewPath::Locked([&] { resident = NewPath::Residency().Manager().Slot(NewPath::PageOf(address)).has_value(); });
    return resident;
}

void KeysCopiedAfterTheirStore() {
    NewPath::PrivatePages target(1);
    NewPath::SharedPages keys(1);
    NewPath::PrivatePages vertices(1);
    NewPath::PrivatePages output(1);
    std::memset(target.Data(), 0x40, target.Bytes());
    for (std::uint32_t word = 0; word < NewPath::WordsPerPage; ++word) keys.Words()[word] = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(keys.Address(), keys.Bytes());
    for (std::uint32_t word = 0; word < NewPath::WordsPerPage; ++word) output.Words()[word] = DeviceTable::Untouched;
    std::memcpy(vertices.Data(), Corners.data(), sizeof(Corners));
    Register(VertexHeader, CornerVertex(), VertexShaderType);
    Register(PixelHeader, FlatPixel(), PixelShaderType);
    {
        Commands warm;
        const auto from = NewPath::BufferResource(vertices.Address(), KeyBytes);
        const auto to = NewPath::BufferResource(output.Address(KeyBytes), KeyBytes);
        const std::array<std::uint32_t, 8> warmUserData{from[0], from[1], from[2], from[3], to[0], to[1], to[2], to[3]};
        warm.Dispatch(CopyKeysProgram(), warmUserData, KeyLanes);
        warm.Submit(0);
        NewPath::WaitIdle();
    }
    NewPath::Locked([] { NewPath::Residency().Manager().SetBudget(std::uint64_t{1} << 30); });
    AgcDriver::Graphics::ResidencyCounts before;
    NewPath::Locked([&] { before = NewPath::Residency().Manager().Counts(); });

    Commands commands;
    Baseline(commands, {target.Data(), Extent, Extent}, CenterAndPosition);
    Context(commands, 0x31c, 0x28028u | DccEnable);
    Context(commands, 0x3b8, RenderTargetAttrib3);
    Context(commands, 0x31b, 0u);
    Context(commands, 0x325, Low(keys.Address() >> 8u));
    Context(commands, 0x3a8, Low(keys.Address() >> 40u));
    Programs(commands, CornerVertex(), VertexUserWords, FlatPixel(), 0u);
    Instances(commands, 1u);
    const auto corners = vertices.Address();
    const std::array<std::uint32_t, VertexUserWords> vertexUserData{Low(corners), (High(corners) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Corners.size()), 0x01016facu};
    VertexUserData(commands, vertexUserData);
    DrawAuto(commands, static_cast<std::uint32_t>(Corners.size()));
    const auto source = NewPath::BufferResource(keys.Address(), KeyBytes);
    const auto destination = NewPath::BufferResource(output.Address(), KeyBytes);
    const std::array<std::uint32_t, 8> copyUserData{source[0], source[1], source[2], source[3], destination[0], destination[1], destination[2], destination[3]};
    commands.Dispatch(CopyKeysProgram(), copyUserData, KeyLanes);
    std::string failure;
    try {
        commands.Submit(0);
        NewPath::WaitIdle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Expect(failure.empty(), "DCC key store before a mirror copy: the submission failed: " + failure);
    AgcDriver::Graphics::ResidencyCounts after;
    NewPath::Locked([&] { after = NewPath::Residency().Manager().Counts(); });
    Expect(Resident(keys.Address()) && (after.admitted > before.admitted || after.refreshed > before.refreshed), "DCC key store before a mirror copy: neither the dispatch's build nor the key store copied the key page into a residency mirror, so the test does not cover the copy-in (admitted " + std::to_string(after.admitted - before.admitted) + ", refreshed " + std::to_string(after.refreshed - before.refreshed) + ", inadmissible " + std::to_string(after.inadmissible - before.inadmissible) + ", deferred " + std::to_string(after.deferred - before.deferred) + ")");
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), output.Bytes());
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t lane = 0; lane < KeyLanes; ++lane) {
        const auto copied = output.Words()[lane];
        if (copied == Uncompressed) continue;
        if (wrong++ == 0u) first = "key word " + std::to_string(lane) + " is " + Hex(copied);
    }
    Expect(wrong == 0u, "DCC key store before a mirror copy: a dispatch in the draw's batch copied keys other than the uncompressed ones the draw stored (a mirror filled before the key store holds the old ones): " + std::to_string(wrong) + " of " + std::to_string(KeyLanes) + " words are not 0xffffffff, first " + first);
    std::vector<std::uint32_t> stored(KeyLanes);
    AgcDriver::GuestMemory::Read(keys.Address(), std::as_writable_bytes(std::span(stored)));
    Expect(std::ranges::all_of(stored, [](std::uint32_t word) { return word == Uncompressed; }), "DCC key store before a mirror copy: the keys in guest memory are not uncompressed after the draw");
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("DCC key store before a mirror copy", KeysCopiedAfterTheirStore);
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
