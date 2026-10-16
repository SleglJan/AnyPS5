#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_RESIDENCYRULESFIXTURE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_RESIDENCYRULESFIXTURE_HPP

#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <functional>

#ifdef _WIN32

namespace ResidencyRules {

using namespace NewPath;
using AgcDriver::Graphics::ResidencyAccess;
using AgcDriver::Graphics::ResidencyCounts;

constexpr std::uint32_t WriteLanes = SubpagesPerPage;
constexpr std::uint32_t WrittenBase = 0x7d000000u;
constexpr std::uint32_t BufferLoadDwordX2 = 0x0du;
constexpr std::uint32_t UnboundedValue = 0x5b000005u;
constexpr std::uint64_t TestBudget = std::uint64_t{1} << 30;

inline int failures = 0;

inline void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

inline const Program& StoreProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(12), 0)
              << Vop2(VAddNcU32, 2, Literal, 0) << WrittenBase
              << Mubuf(BufferStoreDword, 2, 1, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

inline const Program& UnboundedProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(0))
              << Mubuf(BufferLoadDwordX2, 2, 1, 0)
              << Vop1(VMovB32, 4, Literal) << UnboundedValue
              << WaitAll
              << Global(GlobalStoreDword, 0, 2, 4, NullSgpr)
              << EndProgram;
        return built;
    }();
    return program;
}

inline std::uint32_t Pattern(std::uint32_t tag, std::uint64_t page, std::uint32_t word) {
    return (tag << 28u) | (static_cast<std::uint32_t>(page) << 14u) | word;
}

inline void Fill(volatile std::uint32_t* words, std::uint64_t pages, std::uint32_t tag) {
    for (std::uint64_t page = 0; page < pages; ++page) {
        for (std::uint32_t word = 0; word < WordsPerPage; ++word) words[page * WordsPerPage + word] = Pattern(tag, page, word);
    }
}

inline ResidencyCounts Counts() {
    ResidencyCounts counts;
    Locked([&] { counts = Residency().Manager().Counts(); });
    return counts;
}

inline bool Resident(std::uint64_t address) {
    bool resident = false;
    Locked([&] { resident = Residency().Manager().Slot(PageOf(address)).has_value(); });
    return resident;
}

inline void Use(std::uint64_t address) {
    Locked([&] { Residency().Manager().Use(address, PageBytes, ResidencyAccess::Read); });
}

using Expected = std::function<std::uint32_t(std::uint64_t page, std::uint32_t subpage)>;

inline void ExpectRead(const std::vector<std::uint32_t>& lanes, std::uint32_t pages, const Expected& expected, const std::string& what) {
    for (std::uint32_t lane = 0; lane < MirrorReader::Lanes; ++lane) {
        const auto page = lane / SubpagesPerPage;
        const auto subpage = lane % SubpagesPerPage;
        const auto wanted = page < pages ? expected(page, subpage) : 0u;
        if (lanes[lane] == wanted) continue;
        Expect(false, what + ": page " + std::to_string(page) + " subpage " + std::to_string(subpage) + " read " + Hex(lanes[lane]) + ", expected " + Hex(wanted));
        return;
    }
}

inline void Store(std::uint64_t address) {
    Commands commands;
    commands.Dispatch(StoreProgram(), BufferResource(address, static_cast<std::uint32_t>(PageBytes)), WriteLanes);
    commands.Submit(0);
    WaitIdle();
}

class Aliases {
public:
    explicit Aliases(std::uint32_t pages) : bytes(std::uint64_t{pages} * PageBytes) {
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
        Require(section != nullptr, "aliases: cannot create a section");
        for (auto*& view : views) {
            view = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
            Require(view != nullptr, "aliases: cannot reserve a view");
            GuestArena::GuestArenaMap_nid_postfix(view, bytes, section, 0, PAGE_READWRITE);
            GuestAllocations::Mutation().Add(view, bytes, true, true, true);
        }
    }

    ~Aliases() {
        for (auto* view : views) {
            GuestAllocations::Mutation().Remove(view);
            GuestArena::GuestArenaReset_nid_postfix(view, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(view, bytes);
        }
        CloseHandle(section);
    }

    Aliases(const Aliases&) = delete;
    Aliases& operator=(const Aliases&) = delete;

    std::uint64_t Address(std::uint32_t view, std::uint64_t page = 0, std::uint32_t subpage = 0) const {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(views[view])) + page * PageBytes + std::uint64_t{subpage} * SubpageBytes;
    }

    volatile std::uint32_t* Words(std::uint32_t view) const { return reinterpret_cast<volatile std::uint32_t*>(views[view]); }

    void Collect() const {
        for (std::uint32_t view = 0; view < 2u; ++view) AgcDriver::GuestMemory::CollectWritesUncached(Address(view), bytes);
    }

private:
    HANDLE section = nullptr;
    std::array<std::uint8_t*, 2> views{};
    std::uint64_t bytes = 0;
};

inline int Main(const char* name, void (*run)(MirrorReader&)) {
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        MirrorReader reader;
        {
            PrivatePages warm(1);
            reader.Read(warm.Address(), 1);
            reader.Read(warm.Address(), 1);
        }
        Locked([] { Residency().Manager().SetBudget(TestBudget); });
        run(reader);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d %s checks failed\n", failures, name);
        return 1;
    }
    std::printf("%s tests passed\n", name);
    return 0;
}

}

#endif

#endif
