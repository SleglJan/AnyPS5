#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t Rounds = 32;
constexpr std::uint32_t RoundPages = 2;
constexpr std::uint32_t CpuBase = 0x6c000000u;
constexpr std::uint32_t CpuSubpage = 3;
constexpr std::uint32_t DirtyCpuSubpage = 5;
constexpr std::uint32_t UnsettledCpuSubpage = 7;

const Program& AddressReadProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(3), 0)
              << Mubuf(BufferLoadDwordX2, 2, 1, 0)
              << WaitAll
              << Global(GlobalLoadDword, 20, 2, 0, NullSgpr)
              << WaitAll
              << Vop1(VMovB32, 30, Vgpr(0))
              << Vop1(VMovB32, 31, InlineInteger(0))
              << ImageStoreX(20, 30, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

class AddressReader {
public:
    static constexpr std::uint32_t Lanes = SubpagesPerPage;

    AddressReader() : addresses(1), texels(1) {}

    std::vector<std::uint32_t> Read(std::uint64_t page) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto address = page + std::uint64_t{lane} * SubpageBytes;
            addresses.Words()[lane * 2u] = Low(address);
            addresses.Words()[lane * 2u + 1u] = High(address);
            texels.Words()[lane] = MirrorReader::Sentinel;
        }
        std::array<std::uint32_t, 12> user{};
        const auto buffer = BufferResource(addresses.Address(), Lanes * 8u);
        const auto image = ImageResource(texels.Address(), Format32Sint, Lanes, 1u);
        std::copy(buffer.begin(), buffer.end(), user.begin());
        std::copy(image.begin(), image.end(), user.begin() + 4);
        Commands commands;
        commands.Dispatch(AddressReadProgram(), user, Lanes);
        commands.Submit(0);
        WaitIdle();
        AgcDriver::Graphics::StorageTexture::FlushPending(texels.Address(), Lanes * 4u, nullptr, "address reader");
        WaitIdle();
        AgcDriver::GuestMemory::FlushGpuWrites(texels.Address(), Lanes * 4u);
        const auto layout = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32Sint, Lanes, 1u, 1u).at(0);
        std::vector<std::uint32_t> lanes(Lanes);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) std::memcpy(&lanes[lane], texels.Data() + layout.tiledOffset + lane * 4u, 4u);
        return lanes;
    }

private:
    PrivatePages addresses;
    PrivatePages texels;
};

AddressReader* addressReader = nullptr;

bool ImportSettled(std::uint64_t address) {
    bool settled = false;
    Locked([&] { settled = Residency().Manager().ImportSettled(PageOf(address)); });
    return settled;
}

std::uint32_t DirtyMask(std::uint64_t address) {
    std::uint32_t mask = 0;
    Locked([&] { mask = Residency().Manager().Dirty(PageOf(address)); });
    return mask;
}

void Round(MirrorReader& reader, std::uint32_t round) {
    const std::string what = "round " + std::to_string(round);
    const auto tag = 1u + round % 14u;
    PrivatePages pages(RoundPages);
    Fill(pages.Words(), RoundPages, tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    const auto initial = [&](std::uint64_t page, std::uint32_t subpage) { return Pattern(tag, page, subpage * WordsPerSubpage); };
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, initial, what + ": the read that admits the fresh import");
    Expect(Resident(pages.Address()) && Resident(pages.Address(PageBytes)), what + ": the read did not admit both pages");
    Expect(!AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address())), what + ": page 0 is guarded before a submission issued after its import completed");
    Expect(!ImportSettled(pages.Address()) && !ImportSettled(pages.Address(PageBytes)), what + ": the imports settled before an ordering point after the admitting submission completed");
    pages.Words()[UnsettledCpuSubpage * WordsPerSubpage] = CpuBase + 0x20000u + round;
    const auto admitted = [&](std::uint64_t page, std::uint32_t subpage) { return page == 0 && subpage == UnsettledCpuSubpage ? CpuBase + 0x20000u + round : initial(page, subpage); };
    Settle();
    Expect(ImportSettled(pages.Address()) && ImportSettled(pages.Address(PageBytes)), what + ": the imports did not settle after a completed submission issued after them");
    Expect(AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address())), what + ": page 0 has no write guard after its import settled");
    const auto direct = addressReader->Read(pages.Address());
    for (std::uint32_t subpage = 0; subpage < AddressReader::Lanes; ++subpage) {
        if (direct[subpage] == admitted(0, subpage)) continue;
        Expect(false, what + ": a read through addresses loaded from another page, after the settling ordering point, read " + Hex(direct[subpage]) + " in subpage " + std::to_string(subpage) + ", expected " + Hex(admitted(0, subpage)) + " (a CPU store between the admission and the settle)");
        break;
    }
    Expect(Resident(pages.Address()), what + ": page 0 left residency before the read through loaded addresses");
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, admitted, what + ": the read after the settling ordering point missed a CPU store made between the admission and the settle");
    Expect(AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address())) && AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address(PageBytes))), what + ": a settled CPU-writable private page has no write guard");
    Store(pages.Address(PageBytes));
    Expect(Resident(pages.Address(PageBytes)) && DirtyMask(pages.Address(PageBytes)) == 0xffffu, what + ": page 1 is not resident with 16 dirty subpages after the GPU store (mask " + Hex(DirtyMask(pages.Address(PageBytes))) + ")");
    const auto stored = [&](std::uint64_t page, std::uint32_t subpage) { return page == 1 ? WrittenBase + subpage : admitted(page, subpage); };
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, stored, what + ": the GPU read of the dirty mirror");
    pages.Words()[CpuSubpage * WordsPerSubpage] = CpuBase + round;
    pages.Words()[WordsPerPage + DirtyCpuSubpage * WordsPerSubpage] = CpuBase + 0x10000u + round;
    const auto written = [&](std::uint64_t page, std::uint32_t subpage) {
        if (page == 0 && subpage == CpuSubpage) return CpuBase + round;
        if (page == 1 && subpage == DirtyCpuSubpage) return CpuBase + 0x10000u + round;
        return stored(page, subpage);
    };
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, written, what + ": the GPU read after CPU stores into a guarded clean subpage and a dirty one");
    std::uint32_t wrong = 0;
    for (std::uint32_t page = 0; page < RoundPages; ++page) {
        for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) wrong += pages.Words()[page * WordsPerPage + subpage * WordsPerSubpage] != written(page, subpage) ? 1u : 0u;
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " wrong subpage words after the GPU store and the CPU stores");
    Settle();
}

void Run(MirrorReader& reader) {
    AddressReader addresses;
    addressReader = &addresses;
    {
        PrivatePages warm(1);
        addresses.Read(warm.Address());
    }
    const auto before = Counts();
    for (std::uint32_t round = 0; round < Rounds && failures == 0; ++round) Round(reader, round);
    const auto after = Counts();
    Expect(after.importsSettled >= before.importsSettled + std::uint64_t{Rounds} * RoundPages, std::to_string(after.importsSettled - before.importsSettled) + " imports settled in " + std::to_string(Rounds) + " rounds of " + std::to_string(RoundPages) + " fresh pages");
    Expect(after.cpuWriteBacks > before.cpuWriteBacks, "no CPU read wrote a dirty private page back");
    addressReader = nullptr;
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency private guards", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
