#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYDRIVER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYDRIVER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <vulkan/vulkan.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

static_assert(ResidencyStreams == MaxRecorderStreams);

struct ResidencyGuestImport {
    ResidencyImport attributes;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress address = 0;
};

class IResidencyGuestMemory {
public:
    virtual ~IResidencyGuestMemory() = default;
    virtual std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) = 0;
    virtual std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) = 0;
    virtual std::uint64_t Generation() = 0;
    virtual void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) = 0;
    virtual bool Guard(std::uint64_t page) = 0;
    virtual void Unguard(std::span<const std::uint64_t> pages) = 0;
    virtual bool GuardAccess(std::uint64_t page, ResidencyMask mask) = 0;
    virtual void UnguardAccess(std::uint64_t page) = 0;
    virtual std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) = 0;
    virtual void SettleWrites(std::uint64_t page) {
        static_cast<void>(page);
    }
    virtual bool ImportSettled(std::uint64_t page) {
        static_cast<void>(page);
        return true;
    }
    virtual std::uint64_t ImportStamp() { return 0; }
    virtual void SettleImport(std::uint64_t page, std::uint64_t stamp) {
        static_cast<void>(page);
        static_cast<void>(stamp);
    }
    virtual std::uint64_t GuestBytes() = 0;
    virtual void CollectWrites(std::uint64_t address, std::uint64_t bytes) {
        static_cast<void>(address);
        static_cast<void>(bytes);
    }
    virtual ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) {
        static_cast<void>(page);
        static_cast<void>(generation);
        return {};
    }
};

struct ResidencyPublishedPage {
    std::uint64_t page = 0;
    std::uint64_t mirror = 0;
};

struct ResidencyPublication {
    std::uint64_t serial = 0;
    std::vector<ResidencyPublishedPage> changed;
};

struct ResidencyDriverCounts {
    std::uint64_t orderingPoints = 0;
    std::uint64_t appliedAtOrderingPoints = 0;
    std::uint64_t vanishedCopies = 0;
    std::uint64_t frames = 0;
    std::uint64_t touchHarvests = 0;
    std::uint64_t touchedPages = 0;
    std::uint64_t touchOverflows = 0;
    std::uint64_t budgetBytes = 0;
};

class ResidencyDriver final : public IResidencyDevice {
public:
    ResidencyDriver(const Context& context, Recorder& recorder, IResidencyGuestMemory& guest, const ResidencyPolicy& policy);
    ResidencyDriver(const ResidencyDriver&) = delete;
    ResidencyDriver& operator=(const ResidencyDriver&) = delete;
    ~ResidencyDriver() override;

    static ResidencyDriver* Active();
    VkDevice Device() const { return context.device; }

    ResidencyManager& Manager() { return manager; }
    const ResidencyMirrors& Mirrors() const { return mirrors; }
    void OrderingPoint();
    void BeforeDispatch();
    void BeforeCompletionSignal();
    void Flip();
    void UseForRead(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    void BeforeWritingDispatch(std::span<const ResidencyRange> grants, bool writableImports);
    void NoteRecordedWrites(std::uint64_t address, std::uint64_t bytes);
    void PrepareImportWrite(std::uint64_t address, std::uint64_t bytes);
    void RefreshImportWritten(std::uint64_t address, std::uint64_t bytes);
    void StoreThroughMirrors(std::uint64_t address, std::span<const std::byte> bytes);
    void AccessFault(std::uint64_t page);
    void WriteBackAll();
    struct MirrorTarget {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        std::uint64_t end = 0;
    };
    std::optional<MirrorTarget> MirrorOf(std::uint64_t address);
    void NoteDriverWrite(std::uint64_t address, std::uint64_t bytes);
    void DropMirrors(std::span<const ResidencyRange> ranges);
    ResidencyPublication Publish();
    std::vector<Recorder::SectionObject> ChunkObjects(bool written) const;
    void KeepChunks();
    VkDeviceAddress TouchTableAddress() const;
    ResidencyDriverCounts Counts() const;

    bool Reached(const ResidencyToken& token) override;
    void Wait(const ResidencyToken& token) override;
    ResidencyToken Newest() override;
    ResidencyToken Current() override;
    ResidencyImport Import(std::uint64_t page) override;
    std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) override;
    std::uint64_t Generation() override;
    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override;
    std::uint64_t AllocateChunk(std::uint32_t chunk, std::uint32_t slots) override;
    void ReleaseChunk(std::uint32_t chunk) override;
    ResidencyToken CopyIn(std::span<const ResidencyCopy> copies) override;
    ResidencyToken WriteBack(std::span<const ResidencyCopy> copies) override;
    bool Guard(std::uint64_t page) override;
    void Unguard(std::span<const std::uint64_t> pages) override;
    bool GuardAccess(std::uint64_t page, ResidencyMask mask) override;
    void UnguardAccess(std::uint64_t page) override;
    void SettleWrites(std::uint64_t page) override;
    std::uint64_t ImportStamp() override;
    void SettleImport(std::uint64_t page, std::uint64_t stamp) override;
    bool ImportSettled(std::uint64_t page) override;
    std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) override;
    std::uint64_t Budget() override;
    std::uint64_t OtherDeviceBytes() override;
    std::uint64_t GuestBytes() override;
    void CollectWrites(std::uint64_t firstPage, std::uint64_t pages) override;
    ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) override;

private:
    struct TouchReadback {
        std::shared_ptr<Buffer> buffer;
        ResidencyToken token{};
        bool pending = false;
    };

    ResidencyToken record(std::span<const ResidencyCopy> copies, bool intoMirror);
    void frameBoundary();
    void harvestTouches();

    Context context;
    Recorder& recorder;
    IResidencyGuestMemory& guest;
    std::uint32_t deviceHeap = 0;
    ResidencyMirrors mirrors;
    ResidencyManager manager;
    std::shared_ptr<Buffer> touchTable;
    std::atomic<VkDeviceAddress> touchAddress{0};
    std::array<TouchReadback, 2> touchReadbacks;
    std::atomic<std::uint64_t> flips{0};
    std::uint64_t flipsSeen = 0;
    std::uint64_t frames = 0;
    std::uint64_t touchHarvests = 0;
    std::uint64_t touchedPages = 0;
    std::uint64_t touchOverflows = 0;
    std::uint64_t budgetBytes = 0;
    std::atomic<std::uint64_t> orderingPoints{0};
    std::atomic<std::uint64_t> appliedAtOrderingPoints{0};
    std::uint64_t vanishedCopies = 0;
};

void PublishForCompletionSignal();
void PublishBeforeCompletionSignal(SignalKind kind);

}

#endif
