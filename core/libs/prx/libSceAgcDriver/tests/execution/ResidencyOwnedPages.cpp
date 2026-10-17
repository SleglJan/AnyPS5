#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <memory>
#include <span>
#include <vector>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;
using AgcDriver::Graphics::Buffer;
using AgcDriver::Graphics::IResidencyOwner;
using AgcDriver::Graphics::MirrorPiece;
using AgcDriver::Graphics::MirrorSpan;
using AgcDriver::Graphics::Recorder;
using AgcDriver::Graphics::ResidencyRange;

constexpr std::uint32_t FirstRendered = 0x6a000001u;
constexpr std::uint32_t SecondRendered = 0x6a000002u;
constexpr std::uint32_t ThirdRendered = 0x6a000003u;
constexpr std::uint32_t FourthRendered = 0x6a000004u;

class RenderTarget final : public IResidencyOwner {
public:
    RenderTarget(const AgcDriver::Graphics::Context& context, std::uint64_t base, std::uint64_t bytes) : context(context), base(base), bytes(bytes) {
        image = std::make_shared<Buffer>(context, static_cast<std::size_t>(bytes), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Locked([&] { Residency().Manager().Own(base, bytes, this); });
    }

    ~RenderTarget() override {
        Locked([&] { Residency().Manager().Disown(this); });
    }

    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;

    void Render(std::uint32_t value) {
        Locked([&] {
            auto* recorder = Recorder::Active();
            const auto commands = recorder->Commands();
            const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
            const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
            const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
            barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, image->Handle(), 0, bytes, value);
            barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
            recorder->MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
            recorder->Keep(image);
            Residency().Manager().MarkStale(base, bytes, this);
        });
    }

    void Retile(std::span<const ResidencyRange> ranges) override {
        auto* recorder = Recorder::Active();
        for (const auto& range : ranges) {
            ++retiles;
            const MirrorSpan span(&Residency(), range.address, range.address + range.bytes, AgcDriver::Graphics::ResidencyAccess::Write);
            const auto pieces = span.Pieces(range.address, range.address + range.bytes);
            const auto objects = Residency().ChunkObjects(true);
            recorder->DeclareSection(objects);
            const auto commands = recorder->Commands();
            const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
            const auto copy = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
            const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
            const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
            barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            for (const auto& piece : pieces) {
                const auto length = piece.end - piece.begin;
                if (piece.buffer != VK_NULL_HANDLE) {
                    const VkBufferCopy region{piece.begin - base, piece.offset, length};
                    copy(commands, image->Handle(), piece.buffer, 1, &region);
                    continue;
                }
                const auto* import = AgcDriver::Graphics::HostImportFor(context, piece.begin, static_cast<std::size_t>(length));
                Require(import != nullptr, "owned pages: a retile target has no host import");
                const VkBufferCopy region{piece.begin - base, piece.begin - import->base, length};
                copy(commands, image->Handle(), import->buffer, 1, &region);
            }
            barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
            recorder->MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
            recorder->Keep(image);
            span.NoteWritten(pieces);
        }
    }

    std::uint32_t retiles = 0;

private:
    AgcDriver::Graphics::Context context;
    std::uint64_t base;
    std::uint64_t bytes;
    std::shared_ptr<Buffer> image;
};

AgcDriver::Graphics::Context ContextOf(std::uint64_t residentAddress) {
    std::optional<AgcDriver::Graphics::Context> context;
    Locked([&] {
        const auto slot = Residency().Manager().Slot(PageOf(residentAddress));
        Require(slot.has_value(), "owned pages: the page to take a context from is not resident");
        context = Residency().Mirrors().Chunk(slot->chunk)->context;
    });
    return *context;
}

std::uint32_t WrongWords(const volatile std::uint32_t* words, std::uint32_t count, std::uint32_t value) {
    std::uint32_t wrong = 0;
    for (std::uint32_t word = 0; word < count; ++word) wrong += words[word] != value ? 1u : 0u;
    return wrong;
}

void ReleaseAccessGuards(std::uint64_t address, std::uint64_t bytes) {
    for (auto page = address; page < address + bytes; page += SubpageBytes) {
        if (GuestArena::GuestArenaAccessGuarded_nid_postfix(static_cast<std::uintptr_t>(page))) static_cast<void>(GuestArena::GuestArenaHandleAccess_nid_postfix(static_cast<std::uintptr_t>(page), false));
    }
}

void CpuReadGetsRenderedBytes() {
    const std::string what = "a CPU read of a page a render target owns";
    SharedPages pages(1);
    Fill(pages.Words(), 1, 7);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address());
    RenderTarget target(ContextOf(pages.Address()), pages.Address(), PageBytes);
    target.Render(FirstRendered);
    std::uint16_t stale = 0;
    Locked([&] { stale = Residency().Manager().Stale(PageOf(pages.Address())); });
    Expect(stale == 0xffffu && target.retiles == 0u, what + ": the rendered page is not stale in its owner before any reader (stale " + Hex(stale) + ", " + std::to_string(target.retiles) + " retiles)");
    const auto wrong = WrongWords(pages.Words(), WordsPerPage, FirstRendered);
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words miss the rendered value");
    Expect(target.retiles == 1u, what + ": the read retiled " + std::to_string(target.retiles) + " times");
    Settle();
}

void ReadThroughTheTable(MirrorReader& reader) {
    const std::string what = "a read through the page table of a page a render target owns";
    SharedPages pages(2);
    Fill(pages.Words(), 2, 8);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address());
    RenderTarget target(ContextOf(pages.Address()), pages.Address(), PageBytes);
    target.Render(SecondRendered);
    const auto before = Counts();
    ExpectRead(reader.Read(pages.Address(), 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return page == 0 ? SecondRendered : Pattern(8, page, subpage * WordsPerSubpage); }, what);
    const auto after = Counts();
    Expect(target.retiles == 1u && after.writtenBackBytes == before.writtenBackBytes, what + ": the dispatch's retile ran " + std::to_string(target.retiles) + " times and wrote back " + std::to_string(after.writtenBackBytes - before.writtenBackBytes) + " bytes before any CPU read");
    std::uint16_t dirty = 0;
    Locked([&] { dirty = Residency().Manager().Dirty(PageOf(pages.Address())); });
    Expect(dirty == 0xffffu, what + ": the retiled page is not dirty in its mirror (" + Hex(dirty) + ")");
    target.Render(ThirdRendered);
    ExpectRead(reader.Read(pages.Address(), 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return page == 0 ? ThirdRendered : Pattern(8, page, subpage * WordsPerSubpage); }, what + ", after a second render");
    Expect(WrongWords(pages.Words(), WordsPerPage, ThirdRendered) == 0u, what + ": the CPU read after the second render misses its bytes");
    Settle();
}

void UnmapWritesBackFirst(MirrorReader& reader) {
    const std::string what = "an unmap of a page a render target owns";
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(PageBytes), nullptr);
    Require(section != nullptr, "owned pages: cannot create a section");
    auto* first = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, PageBytes));
    auto* second = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, PageBytes));
    Require(first != nullptr && second != nullptr, "owned pages: cannot reserve two pages");
    GuestArena::GuestArenaMap_nid_postfix(first, PageBytes, section, 0, PAGE_READWRITE);
    GuestAllocations::Mutation().Add(first, PageBytes, true, true, true);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(first));
    Fill(reinterpret_cast<volatile std::uint32_t*>(first), 1, 9);
    AgcDriver::GuestMemory::CollectWritesUncached(address, PageBytes);
    Use(address);
    {
        RenderTarget target(ContextOf(address), address, PageBytes);
        target.Render(FourthRendered);
        ReleaseAccessGuards(address, PageBytes);
        Expect(target.retiles == 1u, what + ": releasing the access guards before the unmap's mutation retiled " + std::to_string(target.retiles) + " times");
    }
    GuestAllocations::Mutation().Remove(first);
    GuestArena::GuestArenaReset_nid_postfix(first, PageBytes);
    GuestArena::GuestArenaMap_nid_postfix(second, PageBytes, section, 0, PAGE_READWRITE);
    GuestAllocations::Mutation().Add(second, PageBytes, true, true, true);
    const auto wrong = WrongWords(reinterpret_cast<volatile std::uint32_t*>(second), WordsPerPage, FourthRendered);
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words of another view of the section, mapped after the unmap, miss the rendered bytes");
    ExpectRead(reader.Read(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(second)), 1), 1, [](std::uint64_t, std::uint32_t) { return FourthRendered; }, what + ": a dispatch through the other view");
    GuestAllocations::Mutation().Remove(second);
    GuestArena::GuestArenaReset_nid_postfix(second, PageBytes);
    GuestArena::GuestArenaRelease_nid_postfix(first, PageBytes);
    GuestArena::GuestArenaRelease_nid_postfix(second, PageBytes);
    CloseHandle(section);
}

void PrivatePageHeldForReaders(MirrorReader& reader) {
    const std::string what = "a page of private memory a render target owns";
    SharedPages shared(1);
    Use(shared.Address());
    PrivatePages pages(1);
    std::memset(pages.Data(), 0, PageBytes);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    RenderTarget target(ContextOf(shared.Address()), pages.Address(), PageBytes);
    target.Render(FirstRendered);
    std::uint16_t stale = 0;
    Locked([&] { stale = Residency().Manager().Stale(PageOf(pages.Address())); });
    Expect(target.retiles == 0u && stale == 0xffffu, what + ": the render was retiled " + std::to_string(target.retiles) + " times before any reader (stale " + Hex(stale) + ")");
    ExpectRead(reader.Read(pages.Address(), 1), 1, [](std::uint64_t, std::uint32_t) { return FirstRendered; }, what + ": a dispatch through the page table");
    Expect(target.retiles == 1u, what + ": the dispatch retiled " + std::to_string(target.retiles) + " times");
    target.Render(SecondRendered);
    std::vector<std::uint32_t> read(WordsPerPage);
    AgcDriver::GuestMemory::Read(pages.Address(), std::as_writable_bytes(std::span(read)));
    const auto wrong = static_cast<std::uint32_t>(std::count_if(read.begin(), read.end(), [](std::uint32_t word) { return word != SecondRendered; }));
    Expect(wrong == 0u && target.retiles == 2u, what + ": a driver read after the second render misses " + std::to_string(wrong) + " words (" + std::to_string(target.retiles) + " retiles)");
    Settle();
}

void Run(MirrorReader& reader) {
    CpuReadGetsRenderedBytes();
    ReadThroughTheTable(reader);
    UnmapWritesBackFirst(reader);
    PrivatePageHeldForReaders(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency owned pages", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
