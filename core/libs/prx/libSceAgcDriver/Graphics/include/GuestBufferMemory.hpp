#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTBUFFERMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTBUFFERMEMORY_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "BdaAbi.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class BdaPageTable;

// A guest allocation imported with VK_EXT_external_memory_host, usable by the GPU in place.
struct HostImport {
    std::uint64_t base;
    std::uint64_t bytes;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
    void* alias = nullptr;
    std::shared_ptr<const GuestAllocations::Range> range {};
    std::vector<std::shared_ptr<const GuestAllocations::Range>> ranges {};
    // Identity for the life of this import (see HostImportSerial); 0 until first asked for.
    std::uint64_t serial = 0;
    bool unwatched = false;
    bool dmaBuf = false;
    bool joined = false;
    std::shared_ptr<void> chunk;
    mutable std::uint64_t lastUse = 0;
};

enum class ImportWatch : std::uint8_t { Watch, Unwatch };

void RetainRetiredImport(std::shared_ptr<void> holder, const HostImport* entry);

struct ImportProbe {
    const char* failure = nullptr;
    VkResult result = VK_SUCCESS;
    std::uint32_t pages = 0;
    std::uint32_t writtenAtImport = 0;
    std::uint32_t writtenAfterSubmit = 0;
    std::uint32_t writtenByCpu = 0;
    std::uint32_t writtenAfterFill = 0;
};

ImportProbe ProbeImportWriteProtection(const Context& context);
ImportProbe ProbeDmaBufImportWriteProtection(const Context& context);
ImportWatch PrepareImportWatch(const Context& context);
void SetImportWatch(const Context& context, ImportWatch watch);

// The host import of the registered allocation containing [address, address + bytes), made on demand
// (alignment and budget permitting), or null. Bytes at `address` are at `address - import->base` in
// the import's buffer.
const HostImport* HostImportFor(const Context& context, std::uint64_t address, std::size_t bytes);
void SetHostImportWindow(std::uint64_t bytes);
void SetHostImportRangeWindow(std::uint64_t bytes);
void SetHostImportBudget(std::uint64_t bytes);
std::uint64_t HostImportLiveBytes();
bool ImportMappedRanges(const Context& context, const GuestAllocations::Mapped& ranges, std::uint64_t generation, bool adoptDevice);
// Whether an existing import covers [address, address + bytes), without reconciling the imports
// with the registry or making one (HostImportFor may take a registry lease): a hint for choices
// made outside the device lock (a sampled texture's path, a dispatch's pre-sync); the path taken
// re-checks with HostImportFor when it binds the import.
bool HostImportCovers(const Context& context, std::uint64_t address, std::size_t bytes);
struct RegisteredPiece {
    std::uint64_t address;
    std::uint64_t bytes;
    bool writable;
    bool gpu;
};
// Whether a readable registered allocation contains [address, address + bytes) right now (one
// registry lease): a storage image whose memory was freed or re-registered has nothing to store to.
bool RegisteredReadableCovers(std::uint64_t address, std::size_t bytes, std::vector<RegisteredPiece>* pieces = nullptr);
std::uint64_t RegisteredReadableEnd(std::uint64_t address);

// A persistent device copy of one registered range of the main guest image (which cannot be host
// imported); see GuestBufferMemory.cpp.
struct ImageMirror;
class Recorder;

// Identity of the import serving [address, address + bytes): a serial unique for the life of one
// import, 0 when no import serves the range. An equal serial later means the same VkBuffer still
// backs the range (a re-import gets a new serial), which is what a descriptor set written against it
// needs to stay valid (see the ShaderResources cache). With `reconcile`, imports whose registered
// range changed are dropped first, as an upload does; without it (right after an upload, before the
// work using the import is recorded) the imports are left as they are.
std::uint64_t HostImportSerial(const Context& context, std::uint64_t address, std::size_t bytes, bool reconcile);

// The import table's identity (the fast Revalidate): serials proved under one identity stand
// while the table still has it, i.e. the same device, the same epoch (bumped by every retire)
// and the registry generation it was reconciled with, which must still be the live one.
struct HostImportsProof {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t epoch = 0;
    std::uint64_t refreshedGeneration = 0;
};
// Whether the table still matches `proof` (one lock); false for an empty or stale proof.
bool HostImportsUnchanged(const Context& context, const HostImportsProof& proof);
// The table's identity when it is reconciled with the live registry (one lock), else empty.
HostImportsProof HostImportsIdentity(const Context& context);

// Identity of the read-only image mirror serving [address, address + bytes) (the counterpart of
// HostImportSerial for main-image ranges, which no import serves): a serial unique for the life of
// one mirror with its top bit set, 0 when no read-only mirror whose registered range still exists
// covers the range. A read-only mirror's bytes and VkBuffer never change, so a descriptor set written
// against it stays valid while the serial is unchanged.
std::uint64_t ImageMirrorSerial(const Context& context, std::uint64_t address, std::size_t bytes);

// Deferred lease release (see GuestBufferMemory.cpp). An address-based build leases every readable
// registered allocation until its write-back, which runs when its batch completed; the guest's
// unmap/mprotect/free of a leased allocation waits for that through the registry's pin waiter, which
// submits the recorder and finishes its batches up to the newest one recorded with a lease. SyncLeaseWork
// says whether the build's work must instead be synced as soon as it is recorded
// (APS5_SYNC_LEASE_DISPATCH=1, the behaviour before deferral). The driver and the draw path count
// their choice with CountLeaseOutcome right after the lease-holding resources were kept by the open
// prints the [address-sync] leases line every 10 s from LeaseCounters (cumulative).
bool SyncLeaseWork();
void CountLeaseOutcome(bool synced, const std::array<std::uint64_t, MaxRecorderStreams>& batch);
struct LeaseStats {
    std::uint64_t deferred = 0;
    std::uint64_t synced = 0;
    // Pin-contention waits made by guest threads (the waiter calls), how many of them finished the
    // newest lease batch, how many had to drain the whole recorder instead (no lease batch noted),
    // how many only dropped the cached address space (no GPU wait, see AddressSpaceStats), and
    // their total time.
    std::uint64_t contentionWaits = 0;
    std::uint64_t contentionSyncs = 0;
    std::uint64_t contentionDrains = 0;
    std::uint64_t cacheDrops = 0;
    double contentionMs = 0;
};
LeaseStats LeaseCounters();

// The cached address space of address-based builds (see GuestBufferMemory.cpp): builds served by
// the cached space, rebuilds by reason, builds whose space could not be published (an import
// retired while it was built) and the pin waiter's drops (cumulative). `enabled` is false under
// APS5_NO_ADDRESS_SPACE_CACHE=1, when every build takes the per-build path.
struct AddressSpaceStats {
    bool enabled = true;
    std::uint64_t hits = 0;
    std::uint64_t rebuiltFirst = 0;
    std::uint64_t rebuiltGeneration = 0;
    std::uint64_t rebuiltEpoch = 0;
    std::uint64_t rebuiltDevice = 0;
    std::uint64_t rebuiltWaiterDrop = 0;
    std::uint64_t unpublished = 0;
    // Builds that left the space for the per-build path: a region partially overlapping one of
    // its regions, or an import retired under it before the upload.
    std::uint64_t dissolvedOverlap = 0;
    std::uint64_t dissolvedImports = 0;
    std::uint64_t waiterDrops = 0;
};
AddressSpaceStats AddressSpaceCounters();

struct MirrorStats {
    std::uint64_t mirrors = 0;
    std::uint64_t bytes = 0;
    std::uint64_t rebuilds = 0;
    std::uint64_t sweeps = 0;
    std::uint64_t deviceBytes = 0;
};
MirrorStats MirrorCounters();
void ClearImageMirrors(VkDevice device);
void ClearHostImports(VkDevice device);
void BumpHostImportsEpoch();

class GuestBufferMemory {
public:
    explicit GuestBufferMemory(const Context& context);
    void AcquireRegistered();
    static void Prewarm(const Context& context);
    static void CountAddressBuild();
    void AddInPlace(std::uint64_t address, std::size_t bytes, bool written);
    void Upload();
    void UploadPrepare();
    void UploadFinish();
    VkDescriptorBufferInfo Descriptor(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const;
    static constexpr std::uint64_t AddressAlignment = 16;
    std::pair<VkDeviceAddress, std::uint64_t> Address(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const;
    static std::uint64_t ViewBytes(std::uint64_t bytes, std::uint32_t adjustment);
    std::vector<ShaderRecompiler::BdaAbi::Range> AddressRanges() const;
    std::vector<ShaderRecompiler::BdaAbi::Grant> WriteGrants() const;
    std::vector<ShaderRecompiler::BdaAbi::Range> OwnRanges() const;
    std::shared_ptr<const BdaPageTable> PageTable(const Context& context) const;
    struct ImportSegment {
        std::uint64_t begin;
        std::uint64_t end;
        VkBuffer buffer;
        VkDeviceSize offset;
    };
    std::vector<ImportSegment> WritableImportSegments(std::uint64_t begin, std::uint64_t end) const;
    void SetAccessRanges(std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges, bool unbounded) {
        accessRanges = std::move(ranges);
        accessUnbounded = unbounded;
    }
    std::shared_ptr<const BdaPageTable> PageTableWith(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> overrides) const;
    struct AddressSpace;
    void Release();
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const;
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& Writes() const { return writes; }
    void MarkDirectWrites() const;
    bool HoldsLease() const { return !lease.empty() || space != nullptr; }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> RegionReads() const;

private:
    struct Region {
        std::uint64_t begin;
        std::uint64_t end;
        bool writable;
        const HostImport* direct = nullptr;
        std::shared_ptr<ImageMirror> mirror {};
        bool pending = false;
    };

    enum class BaseOverlap { None, Inside, Partial };
    BaseOverlap baseOverlap(std::uint64_t begin, std::uint64_t end, const Region** owner) const;
    void dissolveSpace(bool resolve);
    const Region* owner(std::uint64_t address) const;
    static ShaderRecompiler::BdaAbi::Range addressRange(const Region& region);
    void validate(std::uint64_t address, std::size_t bytes) const;
    Context context;
    GuestAllocations::Lease lease;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> accessRanges;
    bool accessUnbounded = true;
    std::shared_ptr<const BdaPageTable> buildWith(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> overrides) const;
    std::shared_ptr<const AddressSpace> space;
    std::shared_ptr<const void> retiredSince;
    std::uint64_t importsEpoch = 0;
    std::uint64_t importFloor = 0;
    std::vector<Region> regions;
    bool regionsSorted = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    bool prepared = false;
    bool uploaded = false;
    bool committed = false;
};

}

#endif
