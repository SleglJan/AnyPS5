#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "ResidencyRead_comp_spv.h"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#endif
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t GuestPages = 16;
constexpr std::uint32_t ChunkSlots = 4;
constexpr std::uint32_t OtherQueue = 0x20;
constexpr std::uint32_t WordsPerSubpage = ResidencySubpageBytes / 4;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

struct FaultLog {
    std::uintptr_t base = 0;
    std::uintptr_t bytes = 0;
    std::array<std::atomic<bool>, GuestPages> guarded{};
    std::array<std::atomic<std::uint64_t>, 1024> pages{};
    std::atomic<std::uint32_t> count{0};
};

FaultLog faultLog;

bool Protect(std::uint32_t page, bool readOnly) {
    auto* address = reinterpret_cast<void*>(faultLog.base + (std::uintptr_t{page} << ResidencyPageShift));
#ifdef _WIN32
    DWORD previous = 0;
    return VirtualProtect(address, ResidencyPageBytes, readOnly ? PAGE_READONLY : PAGE_READWRITE, &previous) != 0;
#else
    return mprotect(address, ResidencyPageBytes, readOnly ? PROT_READ : PROT_READ | PROT_WRITE) == 0;
#endif
}

bool TakeFault(std::uintptr_t address) {
    if (address < faultLog.base || address >= faultLog.base + faultLog.bytes) return false;
    const auto page = static_cast<std::uint32_t>((address - faultLog.base) >> ResidencyPageShift);
    if (!faultLog.guarded[page].exchange(false)) return false;
    if (!Protect(page, false)) return false;
    const auto slot = faultLog.count.fetch_add(1);
    if (slot < faultLog.pages.size()) faultLog.pages[slot] = (faultLog.base >> ResidencyPageShift) + page;
    return true;
}

#ifdef _WIN32
LONG CALLBACK FaultHandler(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    return TakeFault(record->ExceptionInformation[1]) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
#else
struct sigaction previousAction {};

void FaultHandler(int signal, siginfo_t* info, void* context) {
    if (TakeFault(reinterpret_cast<std::uintptr_t>(info->si_addr))) return;
    sigaction(signal, &previousAction, nullptr);
    raise(signal);
    static_cast<void>(context);
}
#endif

class GuestRegion {
public:
    GuestRegion() {
        bytes = std::size_t{GuestPages} << ResidencyPageShift;
#ifdef _WIN32
        region = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        handler = AddVectoredExceptionHandler(1, FaultHandler);
#else
        region = static_cast<std::uint8_t*>(std::aligned_alloc(ResidencyPageBytes, bytes));
        struct sigaction action {};
        action.sa_sigaction = FaultHandler;
        action.sa_flags = SA_SIGINFO;
        sigemptyset(&action.sa_mask);
        sigaction(SIGSEGV, &action, &previousAction);
#endif
        Require(region != nullptr, "residency driver: cannot allocate the guest region");
        faultLog.base = reinterpret_cast<std::uintptr_t>(region);
        faultLog.bytes = bytes;
        for (std::uint32_t page = 0; page < GuestPages; ++page) {
            auto* words = Words(First() + page);
            for (std::uint32_t word = 0; word < ResidencyPageBytes / 4; ++word) words[word] = Pattern(page, word);
        }
        GuestAllocations::Mutation().Add(region, bytes, true, true, true);
    }

    ~GuestRegion() {
        for (std::uint32_t page = 0; page < GuestPages; ++page) {
            if (faultLog.guarded[page].exchange(false)) Protect(page, false);
        }
        GuestAllocations::Mutation().Remove(region);
#ifdef _WIN32
        RemoveVectoredExceptionHandler(handler);
        VirtualFree(region, 0, MEM_RELEASE);
#else
        sigaction(SIGSEGV, &previousAction, nullptr);
        std::free(region);
#endif
    }

    GuestRegion(const GuestRegion&) = delete;
    GuestRegion& operator=(const GuestRegion&) = delete;

    static std::uint32_t Pattern(std::uint32_t page, std::uint32_t word) { return (page << 24) | word; }
    std::uint64_t First() const { return reinterpret_cast<std::uintptr_t>(region) >> ResidencyPageShift; }
    std::uint64_t Address(std::uint32_t page, std::uint32_t subpage = 0) const { return ((First() + page) << ResidencyPageShift) + (std::uint64_t{subpage} << ResidencySubpageShift); }
    std::uint32_t* Words(std::uint64_t page) const { return reinterpret_cast<std::uint32_t*>(region + ((page - First()) << ResidencyPageShift)); }

private:
    std::uint8_t* region = nullptr;
    std::size_t bytes = 0;
#ifdef _WIN32
    void* handler = nullptr;
#endif
};

class TestGuest final : public IResidencyGuestMemory {
public:
    TestGuest(const Context& context, const GuestRegion& region) : context(context), region(region) {}

    std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) override {
        if (page < region.First() || page >= region.First() + GuestPages || unmapped.contains(page)) return std::nullopt;
        const auto address = page << ResidencyPageShift;
        const auto* import = HostImportFor(context, address, ResidencyPageBytes);
        if (import == nullptr) return std::nullopt;
        const bool whole = import->base <= address && address + ResidencyPageBytes <= import->base + import->bytes;
        ResidencyImport attributes{whole, true, true};
        if (shared.contains(page)) attributes.section = page + 1;
        return ResidencyGuestImport{attributes, import->buffer, address - import->base, import->address + (address - import->base)};
    }

    std::vector<ResidencyRange> Aliases(std::uint64_t, std::uint64_t) override { return {}; }
    std::uint64_t Generation() override { return generation; }

    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override {
        for (std::size_t k = 0; k < generations.size(); ++k) {
            const auto stamp = stamps.find(firstPage + k);
            changed[k] = generations[k] != ResidencyUnchecked && (generations[k] == 0 || (stamp != stamps.end() && stamp->second > generations[k])) ? 1 : 0;
        }
    }

    bool Guard(std::uint64_t page) override {
        const auto index = static_cast<std::uint32_t>(page - region.First());
        if (!Protect(index, true)) return false;
        faultLog.guarded[index] = true;
        return true;
    }

    void Unguard(std::span<const std::uint64_t> pages) override {
        for (const auto page : pages) {
            const auto index = static_cast<std::uint32_t>(page - region.First());
            if (faultLog.guarded[index].exchange(false)) Require(Protect(index, false), "residency driver: unprotecting a guest page failed");
        }
    }

    bool GuardAccess(std::uint64_t, ResidencyMask) override { return true; }

    void UnguardAccess(std::uint64_t) override {}

    std::vector<ResidencyRange> WholeImports(std::uint64_t&) override { return {}; }

    std::uint64_t GuestBytes() override { return std::uint64_t{GuestPages} << ResidencyPageShift; }

    void Stamp(std::uint64_t page) { stamps[page] = ++generation; }

    void Share(std::uint64_t page) { shared.insert(page); }

    void Unmap(std::uint64_t page) { unmapped.insert(page); }

private:
    const Context& context;
    const GuestRegion& region;
    std::uint64_t generation = 1;
    std::map<std::uint64_t, std::uint64_t> stamps;
    std::set<std::uint64_t> shared;
    std::set<std::uint64_t> unmapped;
};

struct ReadPush {
    std::uint64_t table;
    std::uint64_t results;
    std::uint64_t guestBase;
    std::uint64_t first;
    std::uint32_t count;
    std::uint32_t stride;
};

class Harness {
public:
    Harness(AgcDriver::VulkanDevice& device, Recorder& recorder, const Context& context, GuestRegion& region)
        : device(device), recorder(recorder), context(context), region(region), guest(context, region),
          driver(context, recorder, guest, ResidencyPolicy{ChunkSlots, std::uint64_t{ChunkSlots} << ResidencyPageShift, 2, 2}) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(RESIDENCY_READ_COMP_SPV);
        moduleInfo.pCode = RESIDENCY_READ_COMP_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReadPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        pipelineInfo.layout = layout;
        Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
    }

    ~Harness() {
        Drain();
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    }

    void OnStream(std::uint32_t queue, bool locked, const std::function<void()>& action) {
        std::exception_ptr failure;
        const auto body = [&] {
            try {
                if (queue != 0) AgcDriver::GuestMemory::TagGpuLockThread(queue);
                if (locked) {
                    const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                    action();
                } else {
                    action();
                }
            } catch (...) {
                failure = std::current_exception();
            }
        };
        if (queue == 0) body();
        else std::thread(body).join();
        if (failure) std::rethrow_exception(failure);
    }

    void Drain() {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.SubmitRecorded(false);
        device.WaitIdle();
    }

    void Ordering() {
        const auto count = std::min<std::uint32_t>(faultLog.count.exchange(0), static_cast<std::uint32_t>(faultLog.pages.size()));
        std::vector<std::uint64_t> pages;
        for (std::uint32_t k = 0; k < count; ++k) pages.push_back(faultLog.pages[k]);
        driver.Manager().QueueCpuWritten(pages);
        driver.OrderingPoint();
    }

    std::uint64_t Page(std::uint32_t index) const { return region.First() + index; }

    void Use(std::uint32_t first, std::uint32_t pages) {
        driver.Manager().Use(region.Address(first), std::uint64_t{pages} << ResidencyPageShift, ResidencyAccess::Read);
    }

    std::shared_ptr<Buffer> RecordRead(std::uint32_t first, std::uint32_t pages) {
        driver.Publish();
        auto table = std::make_shared<Buffer>(context, GuestPages * sizeof(std::uint64_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        std::vector<std::uint64_t> entries(GuestPages);
        for (std::uint32_t page = 0; page < GuestPages; ++page) {
            const auto mirror = driver.Manager().Translate(Page(page));
            const auto import = guest.ImportOf(Page(page));
            Require(import.has_value(), "residency driver: the guest region is not imported");
            entries[page] = mirror ? *mirror : import->address;
        }
        std::memcpy(table->Bytes().data(), entries.data(), entries.size() * sizeof(std::uint64_t));
        const auto count = pages * ResidencySubpages;
        auto results = std::make_shared<Buffer>(context, count * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        const ReadPush push{table->DeviceAddress(), results->DeviceAddress(), region.Address(0), region.Address(first), count, ResidencySubpageBytes};
        const auto objects = driver.ChunkObjects(false);
        recorder.DeclareSection(objects);
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (count + 63) / 64, 1, 1);
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        recorder.Keep(table);
        recorder.Keep(results);
        driver.KeepChunks();
        return results;
    }

    void FillImport(std::uint32_t page, std::uint32_t subpage, std::uint32_t value) {
        const auto import = guest.ImportOf(Page(page));
        Require(import.has_value(), "residency driver: the page to fill is not imported");
        const auto commands = recorder.Commands();
        const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
        const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, import->buffer, import->offset + (VkDeviceSize{subpage} << ResidencySubpageShift), ResidencySubpageBytes, value);
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) expected[{page, subpage * WordsPerSubpage + word}] = value;
    }

    void FillMirror(std::uint32_t page, std::uint32_t subpage, std::uint32_t value) {
        const auto slot = driver.Manager().Slot(Page(page));
        Require(slot.has_value(), "residency driver: the page to fill is not resident");
        const auto chunk = driver.Mirrors().Chunk(slot->chunk);
        const std::array<Recorder::SectionObject, 1> objects{Recorder::SectionObject{chunk.get(), true}};
        recorder.DeclareSection(objects);
        const auto commands = recorder.Commands();
        const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, chunk->buffer, (VkDeviceSize{slot->slot} << ResidencyPageShift) + (VkDeviceSize{subpage} << ResidencySubpageShift), ResidencySubpageBytes, value);
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        recorder.Keep(chunk);
    }

    void Store(std::uint32_t page, std::uint32_t word, std::uint32_t value) {
        reinterpret_cast<volatile std::uint32_t*>(region.Words(Page(page)))[word] = value;
        guest.Stamp(Page(page));
        expected[{page, word}] = value;
    }

    std::uint32_t Expected(std::uint32_t page, std::uint32_t word) const {
        const auto found = expected.find({page, word});
        return found == expected.end() ? GuestRegion::Pattern(page, word) : found->second;
    }

    void CheckResults(const std::shared_ptr<Buffer>& results, std::uint32_t first, std::uint32_t pages, const std::string& what) {
        std::vector<std::uint32_t> values(pages * ResidencySubpages);
        std::memcpy(values.data(), results->Bytes().data(), values.size() * sizeof(std::uint32_t));
        for (std::uint32_t k = 0; k < values.size(); ++k) {
            const auto page = first + k / ResidencySubpages;
            const auto word = (k % ResidencySubpages) * WordsPerSubpage;
            if (values[k] == Expected(page, word)) continue;
            Expect(false, what + ": page " + std::to_string(page) + " subpage " + std::to_string(k % ResidencySubpages) + " read " + std::to_string(values[k]) + ", expected " + std::to_string(Expected(page, word)));
            return;
        }
    }

    AgcDriver::VulkanDevice& device;
    Recorder& recorder;
    const Context& context;
    GuestRegion& region;
    TestGuest guest;
    ResidencyDriver driver;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> expected;

private:
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void CrossStreamCopyIn(Harness& harness) {
    harness.OnStream(OtherQueue, true, [&] { harness.Use(0, 4); });
    const auto publication = harness.driver.Publish();
    Expect(publication.changed.size() == 4, "the four admitted pages are published");
    for (const auto& page : publication.changed) Expect(page.mirror != 0, "an admitted page is published at its mirror");
    std::shared_ptr<Buffer> results;
    harness.OnStream(0, true, [&] { results = harness.RecordRead(0, 4); });
    harness.Drain();
    harness.CheckResults(results, 0, 4, "a dispatch on one stream after copy-ins recorded on another");
}

void PinsAcrossStreams(Harness& harness) {
    std::shared_ptr<Buffer> results;
    harness.OnStream(0, true, [&] {
        harness.Use(0, 4);
        results = harness.RecordRead(0, 4);
    });
    const auto before = harness.driver.Manager().Counts();
    harness.OnStream(OtherQueue, true, [&] { harness.Use(4, 2); });
    const auto after = harness.driver.Manager().Counts();
    Expect(after.evicted == before.evicted && after.deferred == before.deferred + 2, "pages another stream's open batch reads are pinned: nothing is evicted, the new pages wait");
    harness.Drain();
    harness.CheckResults(results, 0, 4, "the pinned reads");
    const auto slot0 = harness.driver.Manager().Slot(harness.Page(0));
    const auto slot1 = harness.driver.Manager().Slot(harness.Page(1));
    harness.OnStream(OtherQueue, true, [&] {
        harness.Use(4, 2);
        results = harness.RecordRead(4, 2);
    });
    Expect(!harness.driver.Manager().Slot(harness.Page(0)) && !harness.driver.Manager().Slot(harness.Page(1)), "with the batch finished the two least recent pages are evicted");
    const auto slot4 = harness.driver.Manager().Slot(harness.Page(4));
    const auto slot5 = harness.driver.Manager().Slot(harness.Page(5));
    Expect(slot4 && slot5 && ((slot4 == slot0 && slot5 == slot1) || (slot4 == slot1 && slot5 == slot0)), "the new pages took the slots of an idle GPU at once");
    harness.Drain();
    harness.CheckResults(results, 4, 2, "reads of the newly admitted pages");
}

void StoresAcrossStreams(Harness& harness) {
    const auto page = harness.Page(2);
    harness.OnStream(OtherQueue, true, [&] { harness.Use(2, 1); });
    harness.Drain();
    Expect(harness.driver.Manager().Slot(page).has_value(), "the page is resident before the store");
    harness.OnStream(OtherQueue, true, [&] {
        const std::array<ResidencyRange, 1> grants{ResidencyRange{harness.region.Address(2, 3), ResidencySubpageBytes}};
        harness.driver.BeforeWritingDispatch(grants, false);
        harness.FillImport(2, 3, 0x5a5a5a5au);
    });
    Expect(!harness.driver.Manager().Slot(page), "the write grant demoted the private page before the store was recorded");
    harness.Drain();
    harness.OnStream(0, true, [&] { harness.driver.Manager().FlushForCpuRead(harness.region.Address(2, 3), 4); });
    Expect(harness.region.Words(page)[3 * WordsPerSubpage] == 0x5a5a5a5au && harness.region.Words(page)[4 * WordsPerSubpage - 1] == 0x5a5a5a5au, "a store another stream recorded into the import reached guest memory with no write-back");
    Expect(harness.region.Words(page)[2 * WordsPerSubpage] == GuestRegion::Pattern(2, 2 * WordsPerSubpage), "subpages the GPU did not write are unchanged");
    std::shared_ptr<Buffer> results;
    harness.OnStream(0, true, [&] {
        harness.Use(2, 1);
        results = harness.RecordRead(2, 1);
    });
    Expect(harness.driver.Manager().Slot(page).has_value(), "the next use admits the page again");
    harness.Drain();
    harness.CheckResults(results, 2, 1, "a dispatch on one stream after a store recorded on another");
}

void WriteBackAcrossStreams(Harness& harness) {
    const auto page = harness.Page(7);
    harness.guest.Share(page);
    harness.OnStream(OtherQueue, true, [&] {
        harness.Use(7, 1);
        const std::array<ResidencyRange, 1> grants{ResidencyRange{harness.region.Address(7, 3), ResidencySubpageBytes}};
        harness.driver.BeforeWritingDispatch(grants, false);
        harness.FillMirror(7, 3, 0x5b5b5b5bu);
    });
    Expect(harness.driver.Manager().Slot(page).has_value() && harness.driver.Manager().Dirty(page) == 0x0008, "a write grant into a shared page keeps it resident with the granted subpage dirty");
    harness.Drain();
    Expect(harness.region.Words(page)[3 * WordsPerSubpage] == GuestRegion::Pattern(7, 3 * WordsPerSubpage), "guest memory does not have the GPU's bytes before a CPU read");
    harness.OnStream(0, true, [&] { harness.driver.Manager().FlushForCpuRead(harness.region.Address(7, 3), 4); });
    Expect(harness.region.Words(page)[3 * WordsPerSubpage] == 0x5b5b5b5bu && harness.region.Words(page)[4 * WordsPerSubpage - 1] == 0x5b5b5b5bu, "a flush on one stream writes back what another stream wrote into the mirror");
    Expect(harness.region.Words(page)[2 * WordsPerSubpage] == GuestRegion::Pattern(7, 2 * WordsPerSubpage), "subpages the GPU did not write are unchanged");
    Expect(harness.driver.Manager().Dirty(page) == 0, "the page is clean after the flush");
    for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) harness.expected[{7, 3 * WordsPerSubpage + word}] = 0x5b5b5b5bu;
}

void FaultsAtOrderingPoints(Harness& harness) {
    harness.OnStream(0, true, [&] { harness.Use(4, 2); });
    harness.Drain();
    Expect(!faultLog.guarded[4].load(), "a private page is not guarded before an ordering point after a submission issued after its admission completed");
    harness.OnStream(0, true, [&] { harness.driver.Manager().Retire(); });
    Expect(faultLog.guarded[4].load(), "a resident CPU-writable page is guarded once its import settled");
    const auto applied = harness.driver.Counts().appliedAtOrderingPoints;
    harness.Store(4, 5 * WordsPerSubpage, 0x12345678u);
    Expect(faultLog.count.load() == 1 && !faultLog.guarded[4].load(), "the store faulted and disarmed the guard");
    harness.OnStream(OtherQueue, false, [&] { harness.Ordering(); });
    Expect(harness.driver.Counts().appliedAtOrderingPoints == applied + 1, "the ordering point took the lock and applied the fault");
    Expect(faultLog.guarded[4].load(), "the ordering point re-armed the guard");
    std::shared_ptr<Buffer> results;
    harness.OnStream(0, true, [&] { results = harness.RecordRead(4, 1); });
    harness.Drain();
    harness.CheckResults(results, 4, 1, "a dispatch that names no pages, after a refresh recorded on another stream");
    const auto idle = harness.driver.Counts().appliedAtOrderingPoints;
    harness.OnStream(0, false, [&] { harness.driver.OrderingPoint(); });
    Expect(harness.driver.Counts().appliedAtOrderingPoints == idle, "an ordering point with nothing queued does not take the lock");
}

void GrantsAndDrops(Harness& harness) {
    harness.OnStream(0, true, [&] {
        const std::array<ResidencyRange, 1> grants{ResidencyRange{harness.region.Address(5), 16}};
        harness.driver.BeforeWritingDispatch(grants, false);
    });
    Expect(harness.driver.Manager().Slot(harness.Page(5)) && harness.driver.Manager().Dirty(harness.Page(5)) == 0x0001 && harness.driver.Manager().Slot(harness.Page(4)), "a write grant into a settled private page keeps it resident with the granted subpage dirty");
    const auto resident = harness.driver.Manager().Counts().resident;
    harness.OnStream(0, true, [&] { harness.driver.BeforeWritingDispatch({}, true); });
    Expect(resident != 0 && harness.driver.Manager().Counts().resident == resident && harness.driver.Manager().Dirty(harness.Page(5)) == 0x0001, "a dispatch that may write any writable import keeps the settled GPU-writable mirrors and their dirty subpages");
    harness.Drain();
    harness.OnStream(0, true, [&] { harness.Use(6, 1); });
    harness.driver.Publish();
    harness.driver.Manager().DropRange(harness.region.Address(6), ResidencyPageBytes);
    Expect(harness.driver.Publish().changed.empty(), "a drop over a page whose import is unchanged keeps it");
    harness.guest.Unmap(harness.Page(6));
    harness.driver.Manager().DropRange(harness.region.Address(6), ResidencyPageBytes);
    const auto publication = harness.driver.Publish();
    Expect(publication.changed.size() == 1 && publication.changed[0].page == harness.Page(6) && publication.changed[0].mirror == 0, "a dropped page is published back at its import");
    harness.Drain();
    harness.OnStream(0, true, [&] {
        harness.driver.Manager().Retire();
        harness.driver.Manager().SetBudget(0);
    });
    harness.Drain();
    harness.OnStream(0, true, [&] { harness.driver.Manager().Retire(); });
    Expect(harness.driver.Manager().Counts().chunks == 0 && harness.driver.Mirrors().DeviceBytes() == 0, "a zero budget releases the chunk");
}

}

int main() {
    try {
        GuestRegion region;
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Recorder* recorder = nullptr;
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            recorder = Recorder::Active();
        }
        Require(recorder != nullptr, "residency driver: the device has no recorder");
        if (recorder->Streams() < 2) {
            std::printf("skipped, the device gives the recorder one stream\n");
            return VulkanTestSkipped;
        }
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            if (HostImportFor(context, region.Address(0), std::size_t{GuestPages} << ResidencyPageShift) == nullptr) {
                std::printf("skipped, the device does not import guest memory\n");
                return VulkanTestSkipped;
            }
        }
        {
            Harness harness(*device, *recorder, context, region);
            CrossStreamCopyIn(harness);
            PinsAcrossStreams(harness);
            StoresAcrossStreams(harness);
            WriteBackAcrossStreams(harness);
            FaultsAtOrderingPoints(harness);
            GrantsAndDrops(harness);
            const auto counts = harness.driver.Counts();
            std::printf("residency driver: %zu recorder streams, %llu ordering points, %llu applied, %llu vanished copies\n", recorder->Streams(), static_cast<unsigned long long>(counts.orderingPoints),
                static_cast<unsigned long long>(counts.appliedAtOrderingPoints), static_cast<unsigned long long>(counts.vanishedCopies));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d residency driver checks failed\n", failures);
        return 1;
    }
    std::printf("residency driver tests passed\n");
    return 0;
}
