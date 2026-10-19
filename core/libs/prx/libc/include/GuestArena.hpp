#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP

#include <cstddef>
#include <cstdint>

// Guest virtual memory is placed inside one reserved arena below the PS5 application map limit
// absolute address and breaks on host addresses outside that range.
namespace GuestArena {

enum class MappingChange : std::uint32_t { Committed, Mapped, Released };
enum class GuardRefusal : std::uint8_t { None, UnsettledImport, PrivateHostWrite, PrivatePinned, ResidentView, SharedPinned, AliasHostWrite, Unavailable };
constexpr std::uint32_t GuardRefusals = 8;
constexpr std::uint32_t RefusalBit(GuardRefusal refusal) { return 1u << static_cast<std::uint32_t>(refusal); }

#ifdef _WIN32
using MappingHandle = void*;
#else
using MappingHandle = int;
#endif

#ifndef _WIN32
using SharedBackingResolver = bool (*)(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
using SharedBackingWriter = bool (*)(std::uintptr_t address, const void* source, std::size_t bytes);
#endif

extern "C" {

bool GuestArenaAvailable_nid_postfix();
bool GuestArenaContains_nid_postfix(const void* pointer, std::size_t bytes);
void* GuestArenaAllocate_nid_postfix(std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateAtOrAbove_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateFromHint_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
std::uintptr_t GuestArenaHighestFree_nid_no_patch(std::size_t bytes, std::size_t alignment);
void GuestArenaMarkUsed_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaRelease_nid_postfix(const void* pointer, std::size_t bytes);
// The reserved range, and whether it was reserved with page write watching (Windows MEM_WRITE_WATCH).
void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes);
bool GuestArenaWriteWatched_nid_postfix();
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection);
void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule);
void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, MappingHandle section, std::uint64_t offset, std::uint32_t protection);
std::uint64_t GuestArenaCommitGeneration_nid_postfix();
void GuestArenaSetMappingObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes, std::uint64_t generation, MappingChange change, std::uint32_t protection));
void GuestArenaForEachMapping_nid_postfix(void (*callback)(void* context, std::uintptr_t address, std::size_t bytes, MappingChange change, std::uint32_t protection), void* context);
#ifdef _WIN32
bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address);
bool GuestArenaHandleRead_nid_postfix(std::uintptr_t address);
void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection, std::uintptr_t* end = nullptr);
bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectArmedWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectFreshWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
void GuestArenaCollectWork_nid_postfix(std::uint64_t* systemCalls, std::uint64_t* pieces);
bool GuestArenaHostRegionOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes);
GuardRefusal GuestArenaGuardWrites_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaUnguardWrites_nid_postfix(std::uintptr_t address, std::size_t bytes);
GuardRefusal GuestArenaGuardAccess_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaNoteImport_nid_postfix(std::uintptr_t address, std::size_t bytes);
std::uint64_t GuestArenaImportStamp_nid_postfix();
void GuestArenaSettleImports_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t stamp);
bool GuestArenaImportSettled_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaUnguardAccess_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestArenaAccessGuarded_nid_postfix(std::uintptr_t page);
bool GuestArenaHandleAccess_nid_postfix(std::uintptr_t address, bool write);
void GuestArenaSetAccessGuardObserver_nid_postfix(void (*callback)(std::uintptr_t page));
bool GuestArenaWriteGuarded_nid_postfix(std::uintptr_t page);
void GuestArenaSetWriteGuardObserver_nid_postfix(void (*callback)(const std::uintptr_t* pages, std::size_t count));
void GuestArenaSetGuardRefusalObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes, std::uint32_t cleared));
void GuestArenaAliases_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*callback)(void* context, std::uintptr_t address, std::size_t bytes), void* context);
bool GuestArenaBacking_nid_postfix(std::uintptr_t page, std::uint64_t* section, std::uint64_t* offset, bool* uniform, bool* aliased);
void GuestArenaMapDirect_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection);
void* GuestArenaMapAlias_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaUnmapAlias_nid_postfix(void* alias);
bool GuestArenaResidentSpan_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end);
void GuestArenaForgetSection_nid_postfix(void* section);
void GuestArenaSetSharedImports_nid_postfix(bool imported);
#else
void GuestArenaSetSharedBacking_nid_postfix(SharedBackingResolver resolver);
void GuestArenaSetSharedBackingWriter_nid_no_patch(SharedBackingWriter writer);
bool GuestArenaWriteSharedBacking_nid_no_patch(std::uintptr_t address, const void* source, std::size_t bytes);
bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
void GuestArenaDetach_nid_postfix(void* pointer, std::size_t bytes);
#endif
bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes);

}

class HostWrite {
public:
    HostWrite(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), open(GuestArenaBeginHostWrite_nid_postfix(pointer, bytes)) {}
    ~HostWrite() {
        if (open) GuestArenaEndHostWrite_nid_postfix(pointer, bytes);
    }
    HostWrite(const HostWrite&) = delete;
    HostWrite& operator=(const HostWrite&) = delete;
    bool Open() const { return open; }

private:
    void* pointer;
    std::size_t bytes;
    bool open;
};

}

#endif
