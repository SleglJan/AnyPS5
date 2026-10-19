#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <string>
#include <span>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t Tag = 0x5au;
constexpr std::uint32_t Pages = 3;

AgcDriver::Graphics::ResidencyDriverCounts DriverCounts() {
    return Residency().Counts();
}

constexpr std::uint32_t StrideShift = 11;
constexpr std::uint32_t VXorB32 = 0x1du;

const Program& SegmentReadProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(StrideShift), 0)
              << Mubuf(BufferLoadDword, 20, 1, 0)
              << WaitAll
              << Vop1(VMovB32, 30, Vgpr(0))
              << Vop1(VMovB32, 31, InlineInteger(0))
              << ImageStoreX(20, 30, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

constexpr std::uint32_t OversizedPadding = 60000;
constexpr std::uint32_t PadRegisters = 8;

class OversizedProgram {
public:
    OversizedProgram() {
        using namespace Isa;
        *this << Vop2(VLshlRevB32, 1, InlineInteger(StrideShift), 0)
              << Mubuf(BufferLoadDword, 20, 1, 0)
              << WaitAll
              << Vop1(VMovB32, 30, Vgpr(0))
              << Vop1(VMovB32, 31, InlineInteger(0))
              << Vop1(VMovB32, 40, Vgpr(20));
        for (std::uint32_t pad = 0; pad < PadRegisters; ++pad) *this << Vop2(VAddNcU32, 41 + pad, InlineInteger(pad + 1u), 20);
        for (std::uint32_t index = 0; index < OversizedPadding; ++index) *this << Vop2(VXorB32, 40, Vgpr(40), 41 + index % PadRegisters);
        *this << ImageStoreX(40, 30, 4) << EndProgram;
        storage.assign(code.size() + 64u, 0u);
        std::memcpy(reinterpret_cast<void*>(Address()), code.data(), code.size() * 4u);
    }

    std::uint64_t Address() const { return (reinterpret_cast<std::uintptr_t>(storage.data()) + 255u) & ~std::uintptr_t{255u}; }

private:
    OversizedProgram& operator<<(std::uint32_t word) {
        code.push_back(word);
        return *this;
    }
    OversizedProgram& operator<<(const std::array<std::uint32_t, 2>& pair) { return *this << pair[0] << pair[1]; }

    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> storage;
};

const OversizedProgram& OversizedReadProgram() {
    static const OversizedProgram program;
    return program;
}

void DispatchAt(Commands& commands, std::uint64_t program, std::span<const std::uint32_t> userData, std::uint32_t threads) {
    commands.Registers(0x207, {threads, 1u, 1u});
    commands.Registers(0x20c, {Low(program >> 8u), Low(program >> 40u)});
    commands.Registers(0x213, {(static_cast<std::uint32_t>(userData.size()) << 1u) | (1u << 7u)});
    commands.Registers(0x240, userData);
    commands.Append(0x15u, {1u, 1u, 1u, 0x41u});
}

class SegmentReader {
public:
    std::vector<std::uint32_t> Read(std::uint64_t address, std::uint64_t program = SegmentReadProgram().Address()) {
        for (std::uint32_t lane = 0; lane < MirrorReader::Lanes; ++lane) texels.Words()[lane] = MirrorReader::Sentinel;
        std::array<std::uint32_t, 12> user{};
        const auto buffer = BufferResource(address, static_cast<std::uint32_t>(std::uint64_t{Pages} * PageBytes));
        const auto image = ImageResource(texels.Address(), Format32Sint, MirrorReader::Lanes, 1u);
        std::copy(buffer.begin(), buffer.end(), user.begin());
        std::copy(image.begin(), image.end(), user.begin() + 4);
        Commands commands;
        DispatchAt(commands, program, user, MirrorReader::Lanes);
        commands.Submit(0);
        WaitIdle();
        AgcDriver::Graphics::StorageTexture::FlushPending(texels.Address(), MirrorReader::Lanes * 4u, nullptr, "segment program selection reader");
        WaitIdle();
        AgcDriver::GuestMemory::FlushGpuWrites(texels.Address(), MirrorReader::Lanes * 4u);
        const auto layout = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32Sint, MirrorReader::Lanes, 1u, 1u).at(0);
        std::vector<std::uint32_t> lanes(MirrorReader::Lanes);
        for (std::uint32_t lane = 0; lane < MirrorReader::Lanes; ++lane) std::memcpy(&lanes[lane], texels.Data() + layout.tiledOffset + lane * 4u, 4u);
        return lanes;
    }

private:
    PrivatePages texels{1};
};

void ExpectPattern(const std::vector<std::uint32_t>& lanes, const std::string& what) {
    for (std::uint32_t lane = 0; lane < MirrorReader::Lanes; ++lane) {
        const std::uint64_t byte = std::uint64_t{lane} << StrideShift;
        const auto page = byte / PageBytes;
        const auto wanted = page < Pages ? Pattern(Tag, page, static_cast<std::uint32_t>((byte % PageBytes) / 4u)) : 0u;
        if (lanes[lane] == wanted) continue;
        Expect(false, what + ": lane " + std::to_string(lane) + " at byte " + Hex(byte) + " read " + Hex(lanes[lane]) + ", expected " + Hex(wanted));
        return;
    }
}


void SettledDispatchesBindTheSegmentProgram() {
    SegmentReader reader;
    SharedPages pages(Pages);
    Fill(pages.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(pages.Address(page));
    Settle();
    for (std::uint32_t page = 0; page < Pages; ++page) Expect(Resident(pages.Address(page)), "settled: page " + std::to_string(page) + " has no mirror under full residency");

    const auto first = DriverCounts();
    ExpectPattern(reader.Read(pages.Address()), "pending: the first settled dispatch's read across both page edges");
    const auto pending = DriverCounts();
    Expect(pending.settledDispatches == first.settledDispatches + 1u && pending.unsettledDispatches == first.unsettledDispatches, "pending: the first dispatch over three touching mirror pages was not judged settled");
    Expect(pending.segmentPending == first.segmentPending + 1u && pending.segmentDispatches == first.segmentDispatches, "pending: the first settled dispatch of a program before its segment program exists did not run the walk program counted as segment pending (pending +" + std::to_string(pending.segmentPending - first.segmentPending) + ", segment +" + std::to_string(pending.segmentDispatches - first.segmentDispatches) + ")");

    auto before = pending;
    auto after = pending;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    do {
        before = DriverCounts();
        const auto lanes = reader.Read(pages.Address());
        after = DriverCounts();
        if (after.segmentDispatches != before.segmentDispatches) {
            ExpectPattern(lanes, "segment: the segment program's read across both page edges of one mirrored entry");
            break;
        }
        ExpectPattern(lanes, "segment: a settled dispatch's read while its segment program is pending");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (std::chrono::steady_clock::now() < deadline);
    Expect(after.segmentDispatches == before.segmentDispatches + 1u && after.segmentPending == before.segmentPending, "segment: a settled dispatch never bound the segment program of its program within 60 s");
    Expect(after.segmentFaults == first.segmentFaults, "segment: the segment program raised an out-of-segment error on reads inside its settled range");
    Settle();
}

void UnsettledDispatchesKeepTheWalk() {
    SharedPages writable(Pages);
    PrivatePages addresses(1);
    Fill(writable.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(writable.Address(), writable.Bytes());
    const auto target = writable.Address(1, 0, 2);
    addresses.Words()[0] = Low(target);
    addresses.Words()[1] = High(target);
    AgcDriver::GuestMemory::CollectWritesUncached(addresses.Address(), addresses.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(writable.Address(page));
    Settle();
    for (std::uint32_t attempt = 0; attempt < 2u; ++attempt) {
        const auto before = DriverCounts();
        Commands commands;
        commands.Dispatch(UnboundedProgram(), BufferResource(addresses.Address(), 16u), 1);
        commands.Submit(0);
        WaitIdle();
        const auto after = DriverCounts();
        Expect(after.unsettledDispatches == before.unsettledDispatches + 1u && after.settledDispatches == before.settledDispatches, "unsettled: a dispatch that accesses memory through an address it loads was not counted unsettled (attempt " + std::to_string(attempt) + ")");
        Expect(after.segmentDispatches == before.segmentDispatches && after.segmentPending == before.segmentPending, "unsettled: a dispatch that accesses memory through an address it loads left the walk program (attempt " + std::to_string(attempt) + ")");
        AgcDriver::GuestMemory::FlushGpuWrites(writable.Address(1), PageBytes);
        Expect(writable.Words(1)[2] == UnboundedValue, "unsettled: the walk program's store through the loaded address did not reach guest memory: " + Hex(writable.Words(1)[2]));
    }
    Settle();
}

void OversizedProgramsKeepTheWalk() {
    SegmentReader reader;
    SharedPages pages(Pages);
    Fill(pages.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(pages.Address(page));
    Settle();
    for (std::uint32_t attempt = 0; attempt < 3u; ++attempt) {
        const auto before = DriverCounts();
        ExpectPattern(reader.Read(pages.Address(), OversizedReadProgram().Address()), "oversized: the walk program's read across both page edges (attempt " + std::to_string(attempt) + ")");
        const auto after = DriverCounts();
        Expect(after.settledDispatches == before.settledDispatches + 1u, "oversized: the dispatch over three touching mirror pages was not judged settled (attempt " + std::to_string(attempt) + ")");
        Expect(after.segmentRefused == before.segmentRefused + 1u && after.segmentPending == before.segmentPending && after.segmentDispatches == before.segmentDispatches, "oversized: a settled dispatch of a program over " + std::to_string(AgcDriver::VulkanDevice::SegmentProgramWords) + " SPIR-V words was not kept on the walk program counted as segment refused: size (refused +" + std::to_string(after.segmentRefused - before.segmentRefused) + ", pending +" + std::to_string(after.segmentPending - before.segmentPending) + ", segment +" + std::to_string(after.segmentDispatches - before.segmentDispatches) + ", attempt " + std::to_string(attempt) + ")");
        if (attempt == 0u) std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    Settle();
}

void Run(MirrorReader&) {
    SettledDispatchesBindTheSegmentProgram();
    OversizedProgramsKeepTheWalk();
    UnsettledDispatchesKeepTheWalk();
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("segment program selection", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
