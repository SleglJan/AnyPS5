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
    bool Guard(std::uint64_t page) override;
    void Unguard(std::span<const std::uint64_t> pages) override;
    bool GuardAccess(std::uint64_t page, ResidencyMask mask) override;
    void UnguardAccess(std::uint64_t page) override;
    std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) override;
    std::uint64_t GuestBytes() override;
    ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) override;

private:
    VkDevice device;
};

ResidencyPolicy DeviceResidencyPolicy(const Context& context);
void InstallResidencyHooks();
void RemoveResidencyHooks();
void NoteImportWritten(std::uint64_t address, std::uint64_t bytes);
void RefreshImportWritten(std::uint64_t address, std::uint64_t bytes);
void PrepareImportWrite(std::uint64_t address, std::uint64_t bytes);
std::vector<ShaderRecompiler::BdaAbi::Range> WithResidentPages(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);

}

#endif
