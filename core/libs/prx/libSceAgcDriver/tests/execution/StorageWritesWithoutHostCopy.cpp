#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t ImageLanes = 256;
constexpr std::uint32_t ImageBase = 0x51000000u;
constexpr std::uint32_t Sentinel = 0x0badf00du;

class HostViewPages {
public:
    explicit HostViewPages(std::uint32_t pages) : bytes(std::uint64_t{pages} * PageBytes) {
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, High(bytes), Low(bytes), nullptr);
        Require(section != nullptr, "host view: cannot create a section");
        base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
        Require(base != nullptr, "host view: cannot reserve guest pages");
        GuestArena::GuestArenaMap_nid_postfix(base, bytes, section, 0, PAGE_READWRITE);
        GuestAllocations::Mutation().Add(base, bytes, true, true, true);
        host = static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_READ, 0, 0, static_cast<SIZE_T>(bytes)));
        Require(host != nullptr, "host view: cannot map the section outside the guest arena");
    }

    ~HostViewPages() {
        UnmapViewOfFile(host);
        GuestAllocations::Mutation().Remove(base);
        GuestArena::GuestArenaReset_nid_postfix(base, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(base, bytes);
        CloseHandle(section);
    }

    HostViewPages(const HostViewPages&) = delete;
    HostViewPages& operator=(const HostViewPages&) = delete;

    std::uint64_t Address(std::uint64_t offset = 0) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base)) + offset; }
    volatile std::uint32_t* Guest(std::uint64_t offset = 0) const { return reinterpret_cast<volatile std::uint32_t*>(base + offset); }
    const volatile std::uint32_t* Host(std::uint64_t offset = 0) const { return reinterpret_cast<const volatile std::uint32_t*>(host + offset); }
    std::uint64_t Bytes() const { return bytes; }

private:
    HANDLE section = nullptr;
    std::uint8_t* base = nullptr;
    std::uint8_t* host = nullptr;
    std::uint64_t bytes = 0;
};

const Program& ImageStoreProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VAddNcU32, 20, Literal, 0) << ImageBase
              << Vop1(VMovB32, 30, Vgpr(0))
              << Vop1(VMovB32, 31, InlineInteger(0))
              << ImageStoreX(20, 30, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

void AdmitAll(const HostViewPages& pages) {
    for (std::uint64_t offset = 0; offset < pages.Bytes(); offset += PageBytes) Use(pages.Address(offset));
}

void BufferStore(MirrorReader& reader) {
    const std::string what = "a storage buffer write";
    HostViewPages pages(2);
    Fill(pages.Guest(), 2, 8);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    AdmitAll(pages);
    Expect(Resident(pages.Address()), what + ": the page was not admitted");
    Store(pages.Address());
    ExpectRead(reader.Read(pages.Address(), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return WrittenBase + subpage; }, what + ": the GPU read after the store");
    std::uint32_t copied = 0;
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) copied += pages.Host()[subpage * WordsPerSubpage] != Pattern(8, 0, subpage * WordsPerSubpage) ? 1u : 0u;
    Expect(copied == 0u, what + ": " + std::to_string(copied) + " of the 16 stored words reached host memory before any CPU read; the data must stay in the mirror until a CPU read or an eviction");
    std::uint32_t wrong = 0;
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) wrong += pages.Guest()[subpage * WordsPerSubpage] != WrittenBase + subpage ? 1u : 0u;
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " of the 16 stored words wrong");
    std::uint32_t written = 0;
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) written += pages.Host()[subpage * WordsPerSubpage] == WrittenBase + subpage ? 1u : 0u;
    Expect(written == SubpagesPerPage, what + ": after the CPU read, host memory holds " + std::to_string(written) + " of the 16 stored words");
    Settle();
}

void ImageStore(MirrorReader& reader) {
    const std::string what = "a storage image write";
    static_cast<void>(reader);
    HostViewPages pages(1);
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) pages.Guest()[word] = Sentinel;
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    AdmitAll(pages);
    const auto layout = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32Sint, ImageLanes, 1u, 1u).at(0);
    Require(layout.tiledOffset + ImageLanes * 4u <= pages.Bytes(), "host view: the image does not fit its page");
    Commands commands;
    commands.Dispatch(ImageStoreProgram(), ImageResource(pages.Address(), Format32Sint, ImageLanes, 1u), ImageLanes);
    commands.Submit(0);
    WaitIdle();
    const auto texel = [&](std::uint32_t lane) { return static_cast<std::uint32_t>((layout.tiledOffset + lane * 4u) / 4u); };
    std::uint32_t copied = 0;
    for (std::uint32_t lane = 0; lane < ImageLanes; ++lane) copied += pages.Host()[texel(lane)] != Sentinel ? 1u : 0u;
    Expect(copied == 0u, what + ": " + std::to_string(copied) + " of the " + std::to_string(ImageLanes) + " stored texels reached host memory before any CPU read; the data must stay in the mirror until a CPU read or an eviction");
    std::uint32_t wrong = 0;
    std::string first;
    for (std::uint32_t lane = 0; lane < ImageLanes; ++lane) {
        const auto read = pages.Guest()[texel(lane)];
        if (read == ImageBase + lane) continue;
        if (wrong++ == 0u) first = "texel " + std::to_string(lane) + " is " + Hex(read);
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " stored texels wrong, first " + first);
    std::uint32_t written = 0;
    for (std::uint32_t lane = 0; lane < ImageLanes; ++lane) written += pages.Host()[texel(lane)] == ImageBase + lane ? 1u : 0u;
    Expect(written == ImageLanes, what + ": after the CPU read, host memory holds " + std::to_string(written) + " of the " + std::to_string(ImageLanes) + " stored texels");
    Settle();
}

void Run(MirrorReader& reader) {
    BufferStore(reader);
    ImageStore(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("storage writes without a host copy", Run);
#else
    std::printf("skipped, mirrors are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
