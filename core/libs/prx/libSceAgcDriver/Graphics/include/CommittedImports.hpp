#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_COMMITTEDIMPORTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_COMMITTEDIMPORTS_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace AgcDriver::Graphics {

struct ImportedPage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t serial = 0;
    bool cpuWritable = false;
};

struct CommittedImportStats {
    std::uint64_t views = 0;
    std::uint64_t viewBytes = 0;
    std::uint64_t pinnedBytes = 0;
    std::uint64_t retiredViews = 0;
    std::uint64_t budget = 0;
    std::uint64_t deferredImports = 0;
    std::uint64_t pendingWaits = 0;
    std::uint64_t pendingWaitNanoseconds = 0;
    std::uint64_t cancelledImports = 0;
    std::uint64_t pendingBytes = 0;
};

std::uint64_t GuestImportBudget(const Context& context);
std::uint64_t NewHostImportSerial();
void AttachCommittedImports(const Context& context, bool watched);
void DetachCommittedImports(VkDevice device);
void NoteGuestMapping(std::uint64_t address, std::size_t bytes, GuestArena::MappingChange change, std::uint32_t protection);
const HostImport* CommittedImportCovering(VkDevice device, std::uint64_t begin, std::uint64_t end);
bool CommittedImportsOverlap(VkDevice device, std::uint64_t begin, std::uint64_t end);
std::optional<ImportedPage> ImportOf(VkDevice device, std::uint64_t page);
std::vector<std::pair<std::uint64_t, std::uint64_t>> CommittedImportRanges(VkDevice device, std::uint64_t& generation);
std::uint64_t CommittedImportGeneration(VkDevice device);
std::uint64_t CommittedImportPinnedBytes();
CommittedImportStats CommittedImportCounters();
void SetCommittedImportBudget(std::uint64_t bytes);
void ReleaseRetiredCommittedImports();

}

#endif
