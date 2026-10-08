#include "SceTypes.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <link.h>

extern "C" {
int APS5_VABI sceKernelGetModuleInfoForUnwind(uint64_t addr, int flags, ModuleInfoForUnwind* info);
void* APS5_VABI dlopen_nid_postfix(const char* path, int flags);
void* APS5_VABI dlsym_nid_postfix(void* handle, const char* name);
int APS5_VABI dlclose_nid_postfix(void* handle);
}

static void Require(bool value) { if (!value) std::abort(); }

struct TableSearch {
    std::uintptr_t address;
    std::uintptr_t header;
    bool found;
};

static int FindTables(dl_phdr_info* image, std::size_t, void* data) {
    auto& search = *static_cast<TableSearch*>(data);
    bool contains = false;
    std::uintptr_t header = 0;
    for (std::uint16_t i = 0; i < image->dlpi_phnum; ++i) {
        const auto& segment = image->dlpi_phdr[i];
        const std::uintptr_t start = image->dlpi_addr + segment.p_vaddr;
        if (segment.p_type == PT_LOAD && search.address >= start && search.address - start < segment.p_memsz) contains = true;
        if (segment.p_type == PT_GNU_EH_FRAME) header = start;
    }
    if (!contains) return 0;
    search.header = header;
    search.found = true;
    return 1;
}

static ModuleInfoForUnwind Query(const void* address) {
    ModuleInfoForUnwind info{};
    Require(sceKernelGetModuleInfoForUnwind(reinterpret_cast<std::uint64_t>(address), 1, &info) == 0);
    Require(info.st_size == sizeof(ModuleInfoForUnwind));
    Require(reinterpret_cast<std::uint64_t>(address) - info.seg0_addr < info.seg0_size);
    return info;
}

static void RequireTables(const void* address) {
    TableSearch search{reinterpret_cast<std::uintptr_t>(address), 0, false};
    dl_iterate_phdr(FindTables, &search);
    Require(search.found && search.header != 0);
    const auto* header = reinterpret_cast<const std::uint8_t*>(search.header);
    Require(header[0] == 1 && header[1] == 0x1b);
    std::int32_t offset = 0;
    std::memcpy(&offset, header + 4, sizeof(offset));
    const auto info = Query(address);
    Require(info.eh_frame_hdr_addr == search.header);
    Require(info.eh_frame_addr == search.header + 4 + static_cast<std::intptr_t>(offset));
    Require(info.eh_frame_size != 0);
}

static void RequireNoTables(const void* address) {
    const auto info = Query(address);
    Require(info.eh_frame_hdr_addr == 0 && info.eh_frame_addr == 0 && info.eh_frame_size == 0);
}

int main(int argc, char** argv) {
    Require(argc == 2);
    RequireTables(reinterpret_cast<const void*>(&Query));
    void* module = dlopen_nid_postfix(argv[1], 2);
    Require(module != nullptr);
    const void* add = dlsym_nid_postfix(module, "GuestModuleAdd_nid_postfix");
    Require(add != nullptr);
    RequireTables(add);
    RequireNoTables(reinterpret_cast<const void*>(&dl_iterate_phdr));
    RequireNoTables(reinterpret_cast<const void*>(&sceKernelGetModuleInfoForUnwind));
    Require(dlclose_nid_postfix(module) == 0);
}
