#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYGUEST_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYGUEST_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "BdaAbi.hpp"
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

class ResidencyGuest final : public IResidencyGuestMemory {
public:
    explicit ResidencyGuest(VkDevice device) : device(device) {}
    std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) override;
    std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) override;
    std::uint64_t Generation() override;
    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override;
    ResidencyRefusal Guard(std::uint64_t page) override;
    void Unguard(std::span<const std::uint64_t> pages) override;
    ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) override;
    void UnguardAccess(std::uint64_t page) override;
    void SettleWrites(std::uint64_t page) override;
    std::uint64_t ImportStamp() override;
    bool ImportSettled(std::uint64_t page) override;
    void SettleImport(std::uint64_t page, std::uint64_t stamp) override;
    std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) override;
    std::uint64_t GuestBytes() override;
    std::optional<ResidencyReuseStamp> ReuseStamp() override;
    void CollectWrites(std::uint64_t address, std::uint64_t bytes) override;
    ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) override;

private:
    VkDevice device;
};

ResidencyPolicy DeviceResidencyPolicy(const Context& context);
void InstallResidencyHooks();
void RemoveResidencyHooks();
void NoteImportWritten(std::uint64_t address, std::uint64_t bytes);
void RefreshImportWritten(std::uint64_t address, std::uint64_t bytes);
void StoreThroughMirrors(std::uint64_t address, std::span<const std::byte> bytes);
void PrepareImportWrite(std::uint64_t address, std::uint64_t bytes);
std::vector<ResidencyRange> MirrorStoreSpans(std::uint64_t address, std::uint64_t bytes);
bool AnyMirrored(std::span<const ResidencyRange> spans);
std::vector<ShaderRecompiler::BdaAbi::Range> WithResidentPages(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);

}

#endif
