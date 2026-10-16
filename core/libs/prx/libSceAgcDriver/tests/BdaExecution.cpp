#include "BdaShader.hpp"
#include "BdaAbi.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "StorageImagePages.hpp"
#include "BdaPageReference.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

class Pipeline {
public:
    Pipeline(const Context& context, std::span<const std::uint32_t> code, const std::array<Buffer*, 3>& buffers) : context(context) {
        try {
            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount = bindings.size();
            setInfo.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &size;
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
            VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &setLayout};
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
            for (std::uint32_t i = 0; i < buffers.size(); ++i) {
                const VkDescriptorBufferInfo buffer{buffers[i]->Handle(), 0, buffers[i]->Bytes().size()};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = set;
                write.dstBinding = i;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffer;
                context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
            }
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &setLayout;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
            VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize = code.size_bytes();
            moduleInfo.pCode = code.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            pipelineInfo.layout = layout;
            Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        } catch (...) { release(); throw; }
    }

    ~Pipeline() { release(); }

    void Run(std::uint32_t groups) {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &upload, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groups, 1, 1);
        VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &download, 0, nullptr, 0, nullptr);
        batch.SubmitAndWait();
    }

private:
    void release() noexcept {
        if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        if (setLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }

    const Context& context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

}

static void RunWrittenRangeDecodeTests() {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<std::uint32_t> words(Abi::FaultBufferBytes / sizeof(std::uint32_t));
    std::vector<std::pair<std::uint64_t, std::size_t>> marked;
    const auto decode = [&] {
        marked.clear();
        return Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t address, std::size_t bytes) { marked.emplace_back(address, bytes); });
    };
    const auto claim = [&](std::uint32_t index, std::uint32_t slot, std::uint64_t address, std::uint32_t mask) {
        words[Abi::WrittenListWord + index] = slot;
        words[Abi::WrittenSlotsWord + 2 * slot] = Abi::WrittenChunkKey(address);
        words[Abi::WrittenSlotsWord + 2 * slot + 1] = mask;
        words[Abi::WrittenCountWord] = std::max(words[Abi::WrittenCountWord], index + 1);
    };
    constexpr std::uint64_t page = std::uint64_t{1} << Abi::WrittenPageShift;
    Require(decode() && marked.empty(), "an empty written-page record marked pages");
    constexpr std::uint64_t top = 0x7ffffffe0000ULL;
    claim(0, 5, top, 0b1011u);
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{top, 2 * page}, {top + 3 * page, page}}, "written pages near the top of the address space were not decoded as written");
    std::fill(words.begin(), words.end(), 0u);
    constexpr std::uint64_t high = 0x200000001000ULL;
    claim(0, Abi::WrittenPageSlots - 1, high, 1u << ((high >> Abi::WrittenPageShift) & (Abi::WrittenChunkPages - 1u)));
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{high, page}}, "a written page above 16 TiB was decoded at another address");
    std::fill(words.begin(), words.end(), 0u);
    constexpr std::uint64_t chunk = 0x3e0000000ULL;
    claim(0, 7, chunk, 0xffffffffu);
    claim(1, 8, chunk + (std::uint64_t{1} << Abi::WrittenChunkShift), 1u);
    claim(2, 9, chunk - (std::uint64_t{1} << Abi::WrittenChunkShift), 0x80000000u);
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk, 32 * page}, {chunk + 32 * page, page}, {chunk - page, page}}, "whole, first-page and last-page chunks were not decoded exactly");
    words[Abi::WrittenListWord + 1] = Abi::WrittenPageSlots;
    Require(!decode(), "a claim outside the slots was accepted");
    std::fill(words.begin(), words.end(), 0u);
    words[Abi::WrittenCountWord] = 1;
    Require(!decode(), "a claim of an empty slot was accepted");
    words[Abi::WrittenCountWord] = Abi::WrittenPageSlots + 1;
    Require(!decode(), "more claims than slots were accepted");
    std::fill(words.begin(), words.end(), 0u);
    std::vector<std::pair<std::uint64_t, std::size_t>> images;
    const auto decodeImages = [&] {
        images.clear();
        return Abi::ForEachWrittenImageRange(words.data(), [&](std::uint64_t address, std::size_t bytes) { images.emplace_back(address, bytes); });
    };
    claim(0, 3, chunk, 0b0110u);
    words[Abi::WrittenListWord + 1] = 4;
    words[Abi::WrittenSlotsWord + 2 * 4] = Abi::WrittenImageChunkKey(chunk);
    words[Abi::WrittenSlotsWord + 2 * 4 + 1] = 0b1001u;
    words[Abi::WrittenListWord + 2] = 6;
    words[Abi::WrittenSlotsWord + 2 * 6] = Abi::WrittenImageChunkKey(top);
    words[Abi::WrittenSlotsWord + 2 * 6 + 1] = 0x80000000u;
    words[Abi::WrittenCountWord] = 3;
    Require(Abi::WrittenImageChunkKey(chunk) == (Abi::WrittenChunkKey(chunk) | Abi::WrittenImageKeyBit), "an image chunk key is not its memory key with the image bit");
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk + page, 2 * page}}, "the memory ranges of a record included image chunks or lost a memory chunk");
    Require(decodeImages() && images == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk, page}, {chunk + 3 * page, page}, {top + 31 * page, page}}, "the image ranges of a record included memory chunks or were decoded at other addresses");
    Require(!Abi::WrittenImagesWhole(words.data()), "a record without the whole-image key kept its images whole");
    words[Abi::WrittenListWord + 3] = 9;
    words[Abi::WrittenSlotsWord + 2 * 9] = Abi::WrittenImageWholeKey;
    words[Abi::WrittenSlotsWord + 2 * 9 + 1] = 1u;
    words[Abi::WrittenCountWord] = 4;
    Require(Abi::WrittenImagesWhole(words.data()), "the whole-image key did not keep the record's images whole");
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk + page, 2 * page}}, "the whole-image key was decoded as a memory range");
    Require(decodeImages() && images == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk, page}, {chunk + 3 * page, page}, {top + 31 * page, page}}, "the whole-image key was decoded as an image range");
    Require((static_cast<std::uint64_t>((Abi::WrittenImageWholeKey & ~Abi::WrittenImageKeyBit) - 1u) << Abi::WrittenChunkShift) >= Abi::WrittenAddressLimit, "the whole-image key names a chunk a written address can reach");
    words[Abi::WrittenSlotsWord + 2 * 6] = Abi::WrittenImageKeyBit;
    Require(!decode() && !decodeImages(), "a claim of an image slot without a chunk was accepted");
}

static std::shared_ptr<const BdaPageTable> Uploaded(const Context& context, std::shared_ptr<const BdaPageTable> table) {
    if (table == nullptr) return table;
    CommandBatch batch(context);
    if (table->RecordUploads(context, batch.Handle(), BdaPageTable::EveryStream)) RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    batch.SubmitAndWait();
    return table;
}

static std::shared_ptr<const BdaPageTable> PagesFor(const Context& context, bool paged, std::span<const ShaderRecompiler::BdaAbi::Range> ranges) {
    return paged ? Uploaded(context, BdaPageTable::Build(context, ranges)) : nullptr;
}

static std::uint64_t PageTableAddress(const std::shared_ptr<const BdaPageTable>& pages) {
    return pages != nullptr ? pages->Address() : 0u;
}

static std::unique_ptr<Buffer> MakeBinding(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> own, std::span<const ShaderRecompiler::BdaAbi::Grant> grants, std::uint64_t pageTable, std::uint32_t extraCount = 0) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto binding = std::make_unique<Buffer>(context, sizeof(Abi::Header) + own.size_bytes() + grants.size_bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const Abi::Header header{Abi::TableVersion, static_cast<std::uint32_t>(own.size()) + extraCount, static_cast<std::uint32_t>(grants.size()), 0, pageTable, 0};
    std::memcpy(binding->Bytes().data(), &header, sizeof(header));
    if (!own.empty()) std::memcpy(binding->Bytes().data() + sizeof(header), own.data(), own.size_bytes());
    if (!grants.empty()) std::memcpy(binding->Bytes().data() + sizeof(header) + own.size_bytes(), grants.data(), grants.size_bytes());
    return binding;
}

static std::unique_ptr<Buffer> MakeBinding(const Context& context, bool paged, const std::shared_ptr<const BdaPageTable>& pages, std::span<const ShaderRecompiler::BdaAbi::Range> ranges, std::span<const ShaderRecompiler::BdaAbi::Grant> grants = {}, std::uint32_t extraCount = 0) {
    return MakeBinding(context, paged ? std::span<const ShaderRecompiler::BdaAbi::Range>{} : ranges, grants, PageTableAddress(pages), extraCount);
}

static void RunBdaDwordWriteTests(const Context& context, bool paged) {
    RunWrittenRangeDecodeTests();
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340f80ULL;
    constexpr std::size_t firstBytes = 160;
    Buffer first(context, firstBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer readOnly(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    const std::array<Abi::Range, 3> ranges{{
        {guest, guest + firstBytes, first.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + firstBytes, guest + firstBytes + 16, second.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + firstBytes + 16, guest + firstBytes + 32, readOnly.DeviceAddress(), Abi::Read, 0},
    }};
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const std::array<std::uint32_t, 4> values{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    const auto bytesAt = [&](std::uint64_t address) -> std::byte* {
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            if (address < ranges[index].begin || address >= ranges[index].end) continue;
            auto& buffer = index == 0 ? first : index == 1 ? second : readOnly;
            return buffer.Bytes().data() + (address - ranges[index].begin);
        }
        return nullptr;
    };
    const auto notedBytes = [&](std::uint64_t first, std::uint64_t last) {
        std::uint64_t covered = 0, total = 0;
        const auto* faultWords = reinterpret_cast<const std::uint32_t*>(fault.Bytes().data());
        Require(Abi::ForEachWrittenRange(faultWords, [&](std::uint64_t begin, std::size_t bytes) {
            total += bytes;
            for (const auto address : {first, last}) covered += address >= begin && address < begin + bytes ? 1u : 0u;
        }), "a BDA dword write left an invalid written-page record");
        return std::pair{covered, total};
    };
    const auto pages = PagesFor(context, paged, ranges);
    const auto write = [&](std::uint64_t address, std::uint32_t dwords, std::uint32_t storedBytes, Abi::FaultReason reason, std::uint32_t groups = 1, std::vector<Abi::Grant> grants = {}) {
        const auto table = MakeBinding(context, paged, pages, ranges, grants);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        for (auto* buffer : {&first, &second, &readOnly}) std::memset(buffer->Bytes().data(), 0xa5, buffer->Bytes().size());
        Pipeline pipeline(context, MakeBdaDwordWriteTestShader(address, dwords, values.data()), {table.get(), &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        for (std::uint32_t byte = 0; byte < dwords * 4u; ++byte) {
            const auto* stored = bytesAt(address + byte);
            const auto expected = byte < storedBytes ? static_cast<std::byte>(values[byte / 4u] >> ((byte % 4u) * 8u)) : std::byte{0xa5};
            Require(stored != nullptr && *stored == expected, "BDA dword write stored incorrect bytes at offset " + std::to_string(byte) + " of guest + " + std::to_string(address - guest) + (paged ? " (paged, " : " (") + std::to_string(grants.size()) + " write grants, fault reason " + std::to_string(static_cast<std::uint32_t>(report.reason)) + " at guest + " + std::to_string(report.address - guest) + " bytes " + std::to_string(report.bytes) + ")");
        }
        Require(*bytesAt(address - 1) == std::byte{0xa5}, "BDA dword write touched the byte before it");
        if (address + dwords * 4u < guest + firstBytes + 32) Require(*bytesAt(address + dwords * 4u) == std::byte{0xa5}, "BDA dword write touched the byte after it");
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty, "a mapped BDA dword write published a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "a BDA dword write did not publish its fault");
        }
        const auto [covered, total] = notedBytes(address, address + (storedBytes != 0 ? storedBytes - 1u : 0u));
        if (storedBytes == 0) {
            Require(total == 0, "a BDA dword write that stored nothing noted pages");
        } else {
            const auto pages = ((address + storedBytes - 1u) >> Abi::WrittenPageShift) - (address >> Abi::WrittenPageShift) + 1u;
            Require(covered == 2 && total == pages << Abi::WrittenPageShift, "a BDA dword write did not note exactly its written pages");
            const auto chunks = ((address + storedBytes - 1u) >> Abi::WrittenChunkShift) - (address >> Abi::WrittenChunkShift) + 1u;
            std::uint32_t claims = 0;
            std::memcpy(&claims, fault.Bytes().data() + Abi::WrittenCountWord * sizeof(std::uint32_t), sizeof(claims));
            Require(claims == chunks, "a BDA dword write claimed a written-page slot more than once per chunk");
        }
    };
    write(guest + 16, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 20, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 8, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 12, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 32, 3, 12, static_cast<Abi::FaultReason>(0));
    write(guest + 0x78, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 41, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes - 8, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes + 12, 4, 4, Abi::FaultReason::Permission);
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0), 64);
    constexpr auto readOnlyBegin = guest + firstBytes + 16;
    write(guest + firstBytes + 12, 4, 16, static_cast<Abi::FaultReason>(0), 1, {{readOnlyBegin, readOnlyBegin + 16}});
    write(guest + firstBytes + 12, 4, 8, Abi::FaultReason::Permission, 1, {{readOnlyBegin, readOnlyBegin + 4}});
    write(guest + firstBytes + 12, 2, 4, Abi::FaultReason::Permission, 1, {{readOnlyBegin + 1, readOnlyBegin + 16}});
    write(readOnlyBegin + 4, 2, 8, static_cast<Abi::FaultReason>(0), 1, {{guest, guest + 8}, {readOnlyBegin + 4, readOnlyBegin + 12}});
    write(readOnlyBegin + 4, 2, 4, Abi::FaultReason::Permission, 1, {{guest, guest + 8}, {readOnlyBegin + 4, readOnlyBegin + 11}});
}

static void RunBdaReadTests(const Context& context, bool paged) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    first.Bytes()[0] = std::byte{0x11};
    first.Bytes()[1] = std::byte{0x22};
    first.Bytes()[2] = std::byte{0x33};
    second.Bytes()[0] = std::byte{0x44};
    second.Bytes()[1] = std::byte{0x55};
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    std::array<Abi::Range, 2> ranges{{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0}, {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0}}};
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto run = [&](std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t extraCount = 0, std::uint32_t groups = 1, std::int64_t offset = 0) {
        const auto pages = PagesFor(context, paged, ranges);
        const auto table = MakeBinding(context, paged, pages, ranges, {}, extraCount);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::uint32_t sentinel = 0xdeadbeef;
        std::memcpy(output.Bytes().data(), &sentinel, sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaTestShader(address, bits, offset), {table.get(), &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::uint32_t result = 0;
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(&result, output.Bytes().data(), sizeof(result));
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty && result == expected, "BDA GPU read produced incorrect data or a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "BDA GPU fault was not published correctly");
            Require(result == sentinel, "faulting BDA shader continued to output a substitute value");
        }
    };
    run(guest, 8, 0x11, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 16, 0x3322, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 32, 0x55443322, static_cast<Abi::FaultReason>(0));
    run(guest + 5, 8, 0, Abi::FaultReason::Unmapped, 0, 64);
    run(guest - 1, 8, 0, Abi::FaultReason::Unmapped);
    run(std::numeric_limits<std::uint64_t>::max() - 1, 32, 0, Abi::FaultReason::Overflow);
    run(guest, 8, 0, Abi::FaultReason::InvalidTable, 1);
    run(std::numeric_limits<std::uint64_t>::max() - 2, 8, 0, Abi::FaultReason::Overflow, 0, 1, 4);
    run(1, 8, 0, Abi::FaultReason::Overflow, 0, 1, -4);
    Buffer words(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto read = [&](std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, std::uint32_t value, bool faults) {
        const auto pages = PagesFor(context, paged, ranges);
        const auto table = MakeBinding(context, paged, pages, ranges);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaDwordReadTestShader(address, dwords, coherent, stops), {table.get(), &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        auto expected = sentinel;
        if (!faults || !stops) {
            expected[0] = value;
            for (std::uint32_t dword = 1; dword < dwords; ++dword) expected[dword] = 0;
        }
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::Unmapped && report.address == guest + 5 && report.bytes == 1 && report.instruction == 0x1234, "BDA dword read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA dword read published a fault");
        }
        Require(result == expected, faults && stops ? "faulting BDA dword read continued to output a substitute value" : "BDA dword read produced incorrect data");
    };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            read(guest + 1, 1, coherent, stops, 0x55443322, false);
            read(guest + 3, 1, coherent, stops, 0x5544, true);
            read(guest + 1, 4, coherent, stops, 0x55443322, true);
        }
    }
    Buffer wide(context, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    for (std::uint32_t dword = 0; dword < 8u; ++dword) {
        const std::uint32_t value = 0xa0b0c000u + dword;
        std::memcpy(wide.Bytes().data() + dword * 4u, &value, sizeof(value));
    }
    constexpr std::uint64_t spanGuest = 0x7fff56780000ULL;
    const std::array<Abi::Range, 3> spanRanges{{ranges[0], ranges[1], {spanGuest, spanGuest + 32, wide.DeviceAddress(), Abi::Read, 0}}};
    const auto span = [&](std::uint64_t address, std::int32_t offset, std::uint32_t extracted, bool coherent, bool stops, std::array<std::uint32_t, 4> values, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const auto pages = PagesFor(context, paged, spanRanges);
        const auto spanTable = MakeBinding(context, paged, pages, spanRanges);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaSpanReadTestShader(address, static_cast<std::uint32_t>(offset), extracted, coherent, stops), {spanTable.get(), &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        const bool faults = static_cast<std::uint32_t>(reason) != 0u;
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.instruction == 0x1234, "BDA span read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA span read published a fault");
        }
        Require(result == (faults && stops ? sentinel : values), faults && stops ? "faulting BDA span read continued to output a substitute value" : "BDA span read produced incorrect data");
    };
    const auto at = [&](std::uint32_t dword) { return 0xa0b0c000u + dword; };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            span(spanGuest + 16, -12, 0b1010u, coherent, stops, {0, at(2), 0, at(4)});
            span(spanGuest, 4, 0b0101u, coherent, stops, {at(1), 0, at(3), 0});
            span(spanGuest + 24, 0, 0b0011u, coherent, stops, {at(6), at(7), 0, 0});
            span(spanGuest + 20, 8, 0b0001u, coherent, stops, {at(7), 0, 0, 0});
            span(spanGuest + 28, 0, 0b0101u, coherent, stops, {at(7), 0, 0, 0}, Abi::FaultReason::Unmapped, spanGuest + 36, 1);
            span(guest, 1, 0b1001u, coherent, stops, {0x55443322, 0, 0, 0}, Abi::FaultReason::Unmapped, guest + 13, 1);
            span(4, -8, 0b0001u, coherent, stops, {0, 0, 0, 0}, Abi::FaultReason::Overflow, 4, 0);
        }
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
    RunBdaDwordWriteTests(context, paged);
}

static void RunBdaPageTableTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    namespace Ref = BdaPageReference;
    struct Probe {
        std::uint64_t address;
        std::uint32_t bytes;
        std::uint32_t permission;
    };
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto translate = [&](std::span<const Abi::Range> own, std::span<const Abi::Grant> grants, std::uint64_t pageTable, std::span<const Probe> probes) {
        const auto root = MakeBinding(context, own, grants, pageTable);
        const auto count = static_cast<std::uint32_t>(probes.size());
        Buffer io(context, std::size_t{count} * 8u * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* words = reinterpret_cast<std::uint32_t*>(io.Bytes().data());
        for (std::uint32_t index = 0; index < count; ++index) {
            words[4u * index] = static_cast<std::uint32_t>(probes[index].address);
            words[4u * index + 1u] = static_cast<std::uint32_t>(probes[index].address >> 32u);
            words[4u * index + 2u] = probes[index].bytes;
            words[4u * index + 3u] = probes[index].permission;
        }
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        Pipeline pipeline(context, MakeBdaTranslateTestShader(count), {root.get(), &fault, &io});
        pipeline.Run((count + 63u) / 64u);
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        Require(report.state == Abi::FaultState::Empty, "a BDA translation published a fault");
        std::vector<Ref::Translation> results(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto* result = words + 4u * count + 4u * index;
            results[index] = {result[0] | (static_cast<std::uint64_t>(result[1]) << 32u), result[2]};
        }
        return results;
    };
    const auto hex = [](std::uint64_t value) {
        char text[24];
        std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
        return std::string(text);
    };
    const auto probesOf = [](std::span<const Abi::Range> list, std::initializer_list<std::uint32_t> sizes) {
        std::vector<Probe> probes;
        for (const auto address : Ref::Probes(list)) {
            for (const auto bytes : sizes) {
                for (const auto permission : {Abi::Read, Abi::Write, Abi::Read | Abi::Write}) probes.push_back({address, bytes, permission});
            }
        }
        return probes;
    };
    const auto check = [&](const std::string& name, std::span<const Abi::Range> list, const std::shared_ptr<const BdaPageTable>& pages, std::span<const Abi::Range> held, std::span<const Abi::Grant> grants = {}, std::span<const Abi::Grant> bound = {}, bool boundGiven = false) {
        auto probes = probesOf(list, {1u, 4u, 16u, 64u, 0xfffffff0u});
        const auto own = Ref::Without(list, held);
        const auto gpuGrants = boundGiven ? bound : grants;
        for (const auto& set : {grants, gpuGrants}) {
            for (const auto& grant : set) {
                for (const auto edge : {grant.begin, grant.end}) {
                    for (const std::int64_t delta : {-17, -16, -4, -1, 0, 1, 4, 16}) {
                        for (const auto bytes : {1u, 4u, 16u}) {
                            for (const auto permission : {Abi::Write, Abi::Read | Abi::Write}) probes.push_back({edge + static_cast<std::uint64_t>(delta), bytes, permission});
                        }
                    }
                }
            }
        }
        const auto results = translate(own, gpuGrants, PageTableAddress(pages), probes);
        const auto alone = pages != nullptr ? translate({}, gpuGrants, pages->Address(), probes) : std::vector<Ref::Translation>{};
        const auto leaves = Ref::EncodeLeaves(held, [](std::uint32_t window, std::span<const Abi::Range> ranges) { return EncodeBdaWindow(window, ranges); });
        const Ref::Leaves none;
        std::size_t differs = 0;
        for (std::size_t index = 0; index < probes.size(); ++index) {
            const auto& probe = probes[index];
            const auto expected = Ref::Translate(leaves, own, probe.address, probe.bytes, probe.permission, gpuGrants);
            Require(results[index] == expected, name + ": the GPU translated " + hex(probe.address) + " (" + std::to_string(probe.bytes) + " bytes, permission " + std::to_string(probe.permission) + ") to " + hex(results[index].device) + " reason " + std::to_string(results[index].reason) + ", expected " + hex(expected.device) + " reason " + std::to_string(expected.reason));
            if (!(results[index] == Ref::Translate(none, list, probe.address, probe.bytes, probe.permission, grants))) ++differs;
            if (pages == nullptr) continue;
            const auto paged = Ref::Translate(none, held, probe.address, probe.bytes, probe.permission, gpuGrants);
            Require(alone[index] == paged, name + ": the GPU page table alone translated " + hex(probe.address) + " (" + std::to_string(probe.bytes) + " bytes, permission " + std::to_string(probe.permission) + ") to " + hex(alone[index].device) + " reason " + std::to_string(alone[index].reason) + ", its ranges to " + hex(paged.device) + " reason " + std::to_string(paged.reason));
        }
        return differs;
    };
    std::mt19937_64 random(0x9a6e7ab1eu);
    for (const auto base : {0x1fffd0000ull, 0x7ff6e4600000ull, 0x2000001234ull}) {
        const auto table = Ref::RandomTable(random, base, 60u);
        const auto grants = Ref::RandomGrants(random, table.list, 8u);
        Require(check("list alone", table.list, nullptr, {}) == 0u, "the BDA list lookup differs from its reference");
        Require(check("list alone with write grants", table.list, nullptr, {}, grants) == 0u, "the BDA list lookup with write grants differs from its reference");
        Require(check("random pages", table.list, Uploaded(context, BdaPageTable::Build(context, table.pages)), table.pages) == 0u, "the page table and the own ranges translated a random table unlike its ranges");
        Require(check("random pages with write grants", table.list, Uploaded(context, BdaPageTable::Build(context, table.pages)), table.pages, grants) == 0u, "the page table and the own ranges with write grants translated a random table unlike its ranges");
        auto shifted = grants;
        for (auto& grant : shifted) ++grant.end;
        Require(check("wrong write grants", table.list, Uploaded(context, BdaPageTable::Build(context, table.pages)), table.pages, grants, shifted, true) != 0u, "write grants one byte too long went unnoticed");
    }
    constexpr std::uint64_t base = 0x3fffff0000ull;
    const std::vector<Abi::Range> shapes{
        {base + 0x100u, base + 0x3000u, 0x100000000ull, Abi::Read, 0u},
        {base + 0x3000u, base + 0x9001u, 0x200000000ull, Abi::Read | Abi::Write, 0u},
        {base + 0x9001u, base + 0x9002u, 0x300000000ull, Abi::Read, 0u},
        {base + 0x9010u, base + 0x10000u, 0x400000000ull, 0u, 0u},
        {base + 0x10000u, base + 0x10000u + 0x240000000ull, 0x500000000ull, Abi::Read | Abi::Write, 0u},
    };
    Require(check("partial and split pages", shapes, Uploaded(context, BdaPageTable::Build(context, shapes)), shapes) == 0u, "the page table changed a BDA translation of partial and split pages");
    const std::vector<Abi::Range> shared{shapes[0], shapes[2], shapes[4]};
    Require(check("split pages shared with own ranges", shapes, Uploaded(context, BdaPageTable::Build(context, shared)), shared) == 0u, "the page table and the own ranges of one page translated unlike their ranges");
    auto wrong = shapes;
    wrong[1].deviceAddress += 0x40u;
    Require(check("wrong translation", shapes, Uploaded(context, BdaPageTable::Build(context, wrong)), wrong) != 0u, "a page table with a wrong translation went unnoticed");
    auto writable = shapes;
    writable[0].permissions |= Abi::Write;
    Require(check("wrong permission", shapes, Uploaded(context, BdaPageTable::Build(context, writable)), writable) != 0u, "a page table that made a read-only range writable went unnoticed");
    constexpr std::uint64_t split = 0x6000001000ull;
    constexpr std::uint64_t splitDevice = 0xa00000000ull;
    const std::vector<Abi::Range> whole{{split, split + 0x30000u, splitDevice, Abi::Read, 0u}};
    const std::vector<Abi::Grant> middle{{split + 0x10010u, split + 0x10050u}};
    const auto wholePages = Uploaded(context, BdaPageTable::Build(context, whole));
    Require(check("write grant inside a read-only range", whole, wholePages, whole, middle) == 0u, "a write grant inside a read-only range translated unlike its reference");
    const auto grantProbes = probesOf(std::vector<Abi::Range>{{split + 0x10010u, split + 0x10050u, splitDevice + 0x10010u, Abi::Read, 0u}}, {1u, 4u, 16u, 64u});
    const auto granted = translate({}, middle, wholePages->Address(), grantProbes);
    const auto refused = translate({}, {}, wholePages->Address(), grantProbes);
    std::size_t permitted = 0;
    for (std::size_t index = 0; index < grantProbes.size(); ++index) {
        const auto& probe = grantProbes[index];
        const bool inside = probe.address >= split && probe.address + probe.bytes <= split + 0x30000u;
        const bool covered = probe.address >= middle[0].begin && probe.address + probe.bytes <= middle[0].end;
        if (!inside) continue;
        if (probe.permission == Abi::Read) {
            Require(granted[index] == Ref::Translation{splitDevice + (probe.address - split), 0u} && refused[index] == granted[index], "a read inside a read-only range with a write grant at " + hex(probe.address) + " was not translated");
            continue;
        }
        Require(refused[index] == Ref::Fault(Abi::FaultReason::Permission), "a store to a read-only range without a write grant at " + hex(probe.address) + " was not a permission fault");
        Require(granted[index] == (covered ? Ref::Translation{splitDevice + (probe.address - split), 0u} : Ref::Fault(Abi::FaultReason::Permission)), "a store at " + hex(probe.address) + " (" + std::to_string(probe.bytes) + " bytes) was not decided by the write grant");
        if (covered) ++permitted;
    }
    Require(permitted != 0u, "no probe was permitted by the write grant");
    constexpr std::uint64_t window = 0x500000000ull;
    const Abi::Range kept{window + 0x200000000ull, window + 0x200040000ull, 0x600000000ull, Abi::Read, 0u};
    const std::vector<Abi::Range> before{{window + 0x10u, window + 0x8000u, 0x700000000ull, Abi::Read, 0u}, {window + 0x8000u, window + 0x30000u, 0x800000000ull, Abi::Read | Abi::Write, 0u}, kept};
    const std::vector<Abi::Range> after{{window + 0x10u, window + 0x8000u, 0x700000000ull, Abi::Read, 0u}, {window + 0x9000u, window + 0x9100u, 0x900000000ull, Abi::Read, 0u}, kept};
    const auto first = Uploaded(context, BdaPageTable::Build(context, before));
    const auto second = Uploaded(context, BdaPageTable::Build(context, after));
    const auto keptWindow = static_cast<std::uint32_t>(kept.begin >> Abi::WindowShift);
    const auto changedWindow = static_cast<std::uint32_t>(window >> Abi::WindowShift);
    Require(first->LeafAddress(keptWindow) != 0u && first->LeafAddress(keptWindow) == second->LeafAddress(keptWindow), "a page table update did not share the leaf of an unchanged window");
    Require(first->LeafAddress(changedWindow) != second->LeafAddress(changedWindow) && first->Address() != second->Address(), "a page table update changed the table it was derived from");
    Require(check("before the update", before, first, before) == 0u, "the page table before an update changed a BDA translation");
    Require(check("after the update", after, second, after) == 0u, "the page table after an update changed a BDA translation");
    Require(check("before the update, again", before, first, before) == 0u, "an update changed the page table a recorded dispatch still uses");
}

static void RunBdaWrittenPageTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7ffe00000000ULL;
    constexpr std::uint64_t page = std::uint64_t{1} << Abi::WrittenPageShift;
    struct Case {
        const char* name;
        BdaLaneStore store;
        std::uint64_t stride;
        std::uint64_t offset;
        std::uint64_t active;
    };
    const auto bytesOf = [](BdaLaneStore store) { return store == BdaLaneStore::Byte ? 1u : store == BdaLaneStore::Short ? 2u : store == BdaLaneStore::Dwordx4 ? 16u : 4u; };
    const std::array<Case, 10> cases{{
        {"unaligned dwords across pages, alternate lanes", BdaLaneStore::Dword, 0x1000u, 0xffeu, 0xaaaaaaaaaaaaaaaaull},
        {"unaligned runtime dwords across pages, other lanes", BdaLaneStore::RuntimeDword, 0x1000u, 0xffdu, 0x5555555555555555ull},
        {"dwordx4 across chunks, scattered lanes", BdaLaneStore::Dwordx4, 0x20000u, 0x1fff8u, 0x8421084210842108ull},
        {"unaligned dwordx4 across pages", BdaLaneStore::Dwordx4, 0x3000u, 0x2ff6u, 0xffffffff00000000ull},
        {"aligned dwords of all lanes in one page", BdaLaneStore::Dword, 16u, 0u, ~0ull},
        {"bytes of every third lane", BdaLaneStore::Byte, 0x1001u, 1u, 0x9249249249249249ull},
        {"shorts across pages", BdaLaneStore::Short, 0x2000u, 0xfffu, 0x00000000ffff0000ull},
        {"atomics of all lanes", BdaLaneStore::Atomic, 0x3000u, 0x10u, ~0ull},
        {"one lane", BdaLaneStore::Dword, 0x20000u, 0x1fffeu, 1ull << 37u},
        {"no lane", BdaLaneStore::Dword, 0x1000u, 0xffeu, 0ull},
    }};
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (const bool paged : {false, true}) {
        for (const auto& test : cases) {
            const auto bytes = bytesOf(test.store);
            const auto span = test.offset + 64u * test.stride + 64u;
            Buffer memory(context, static_cast<std::size_t>(span), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            const std::array<Abi::Range, 1> ranges{{{guest, guest + span, memory.DeviceAddress(), Abi::Read | Abi::Write, 0}}};
            const auto pages = PagesFor(context, paged, ranges);
            const auto table = MakeBinding(context, paged, pages, ranges);
            std::memset(memory.Bytes().data(), 0, memory.Bytes().size());
            std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
            Pipeline pipeline(context, MakeBdaLaneStoreTestShader(guest + test.offset, test.stride, test.active, test.store), {table.get(), &fault, &output});
            pipeline.Run(1);
            const std::string label = std::string("written pages, ") + test.name + (paged ? " (paged): " : ": ");
            Abi::Fault report{};
            std::memcpy(&report, fault.Bytes().data(), sizeof(report));
            Require(report.state == Abi::FaultState::Empty, label + "a mapped store published a fault");
            std::set<std::uint64_t> expectedPages;
            std::set<std::uint32_t> expectedChunks;
            std::vector<std::uint8_t> expected(memory.Bytes().size(), 0u);
            for (std::uint32_t lane = 0; lane < 64u; ++lane) {
                if (((test.active >> lane) & 1u) == 0u) continue;
                const auto address = guest + test.offset + lane * test.stride;
                for (std::uint32_t byte = 0; byte < bytes; ++byte) {
                    expectedPages.insert((address + byte) >> Abi::WrittenPageShift);
                    expectedChunks.insert(Abi::WrittenChunkKey(address + byte));
                    expected[address + byte - guest] = static_cast<std::uint8_t>(BdaLaneStoreValue >> ((byte % 4u) * 8u));
                }
            }
            Require(std::memcmp(memory.Bytes().data(), expected.data(), expected.size()) == 0, label + "the stores of the active lanes landed incorrectly or an inactive lane stored");
            std::array<std::uint32_t, Abi::FaultBufferBytes / sizeof(std::uint32_t)> words{};
            std::memcpy(words.data(), fault.Bytes().data(), sizeof(words));
            Require(words[Abi::WrittenOverflowWord] == 0, label + "the written-page table overflowed");
            std::set<std::uint64_t> recorded;
            Require(Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t begin, std::size_t length) {
                for (std::uint64_t at = begin; at < begin + length; at += page) Require(recorded.insert(at >> Abi::WrittenPageShift).second, label + "a page is recorded twice");
            }), label + "invalid written-page record");
            Require(recorded == expectedPages, label + "the records name " + std::to_string(recorded.size()) + " pages, the active lanes wrote " + std::to_string(expectedPages.size()));
            Require(words[Abi::WrittenCountWord] == expectedChunks.size(), label + "the records claim " + std::to_string(words[Abi::WrittenCountWord]) + " chunks, the active lanes wrote " + std::to_string(expectedChunks.size()));
        }
    }
}

static std::vector<BdaPageReference::Translation> TranslateThroughTable(const Context& context, std::uint64_t pageTable, std::span<const std::array<std::uint64_t, 3>> probes) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto root = MakeBinding(context, {}, {}, pageTable);
    const auto count = static_cast<std::uint32_t>(probes.size());
    Buffer io(context, std::size_t{count} * 8u * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto* words = reinterpret_cast<std::uint32_t*>(io.Bytes().data());
    for (std::uint32_t index = 0; index < count; ++index) {
        words[4u * index] = static_cast<std::uint32_t>(probes[index][0]);
        words[4u * index + 1u] = static_cast<std::uint32_t>(probes[index][0] >> 32u);
        words[4u * index + 2u] = static_cast<std::uint32_t>(probes[index][1]);
        words[4u * index + 3u] = static_cast<std::uint32_t>(probes[index][2]);
    }
    std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
    Pipeline pipeline(context, MakeBdaTranslateTestShader(count), {root.get(), &fault, &io});
    pipeline.Run((count + 63u) / 64u);
    Abi::Fault report{};
    std::memcpy(&report, fault.Bytes().data(), sizeof(report));
    Require(report.state == Abi::FaultState::Empty, "a BDA page table version: a translation published a fault");
    std::vector<BdaPageReference::Translation> results(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto* result = words + 4u * count + 4u * index;
        results[index] = {result[0] | (static_cast<std::uint64_t>(result[1]) << 32u), result[2]};
    }
    return results;
}

static void RunBdaPageTableVersionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    namespace Ref = BdaPageReference;
    using Ranges = std::vector<Abi::Range>;
    using Windows = std::map<std::uint32_t, Ranges>;
    constexpr std::uint64_t Page = std::uint64_t{1} << Abi::PageShift;
    constexpr std::uint32_t Rw = Abi::Read | Abi::Write;
    const auto hex = [](std::uint64_t value) {
        char text[24];
        std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
        return std::string(text);
    };
    const auto windowsOf = [](const Ranges& list) {
        Windows windows;
        for (const auto& range : list) {
            for (auto window = static_cast<std::uint32_t>(range.begin >> Abi::WindowShift); window <= static_cast<std::uint32_t>((range.end - 1u) >> Abi::WindowShift); ++window) windows[window].push_back(range);
        }
        return windows;
    };
    const auto same = [](const Ranges& left, const Ranges& right) {
        return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](const Abi::Range& a, const Abi::Range& b) { return a.begin == b.begin && a.end == b.end && a.deviceAddress == b.deviceAddress && a.permissions == b.permissions && a.reserved == b.reserved; });
    };
    struct Version {
        std::string name;
        Ranges list;
        std::shared_ptr<const BdaPageTable> table;
    };
    const auto expectTranslations = [&](const Version& version, const std::string& when) {
        std::vector<std::uint64_t> addresses;
        for (const auto& range : version.list) {
            for (const auto probe : Ref::Probes(std::span<const Abi::Range>(&range, 1))) addresses.push_back(probe);
            for (auto page = range.begin & ~(Page - 1u); page < range.end; page += Page) {
                addresses.push_back(std::max(page, range.begin));
                addresses.push_back(std::min(page + Page / 2u, range.end - 1u));
                addresses.push_back(std::min(page + Page, range.end) - 1u);
            }
        }
        std::sort(addresses.begin(), addresses.end());
        addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
        std::size_t cpuWrong = 0;
        std::string firstCpu;
        std::vector<std::array<std::uint64_t, 3>> probes;
        for (const auto address : addresses) {
            if (address >= Abi::PageTableEnd) continue;
            const auto found = std::upper_bound(version.list.begin(), version.list.end(), address, [](std::uint64_t value, const Abi::Range& range) { return value < range.begin; });
            const Abi::Range* holder = found != version.list.begin() && address < std::prev(found)->end ? &*std::prev(found) : nullptr;
            const auto looked = version.table->Find(address);
            const bool right = holder == nullptr ? !looked.has_value() : looked.has_value() && looked->begin == holder->begin && looked->end == holder->end && looked->deviceAddress == holder->deviceAddress && looked->permissions == holder->permissions;
            if (!right && cpuWrong++ == 0u) firstCpu = hex(address) + " found " + (looked ? "[" + hex(looked->begin) + ", " + hex(looked->end) + ") at " + hex(looked->deviceAddress) : std::string("nothing"));
            for (const auto bytes : {1u, 4u, 16u}) {
                for (const auto permission : {Abi::Read, Abi::Write}) probes.push_back({address, bytes, permission});
            }
        }
        Require(cpuWrong == 0u, version.name + ", " + when + ": " + std::to_string(cpuWrong) + " addresses were looked up in another range than the version's list holds them, first " + firstCpu);
        const auto results = TranslateThroughTable(context, version.table->Address(), probes);
        const Ref::Leaves none;
        std::size_t gpuWrong = 0;
        std::string firstGpu;
        for (std::size_t index = 0; index < probes.size(); ++index) {
            const auto& probe = probes[index];
            const auto expected = Ref::Translate(none, version.list, probe[0], static_cast<std::uint32_t>(probe[1]), static_cast<std::uint32_t>(probe[2]));
            if (results[index] == expected || gpuWrong++ != 0u) continue;
            firstGpu = hex(probe[0]) + " (" + std::to_string(probe[1]) + " bytes, permission " + std::to_string(probe[2]) + ") to " + hex(results[index].device) + " reason " + std::to_string(results[index].reason) + ", expected " + hex(expected.device) + " reason " + std::to_string(expected.reason);
        }
        Require(gpuWrong == 0u, version.name + ", " + when + ": the GPU translated " + std::to_string(gpuWrong) + " of " + std::to_string(probes.size()) + " probes unlike the version's ranges, first " + firstGpu);
    };
    const auto build = [&](std::string name, Ranges list, const std::vector<const Version*>& shares, const std::set<std::uint32_t>& changed) {
        std::vector<std::shared_ptr<const BdaPageTable>> tables;
        for (const auto* share : shares) tables.push_back(share != nullptr ? share->table : nullptr);
        Version version;
        version.name = std::move(name);
        version.list = std::move(list);
        version.table = Uploaded(context, BdaPageTable::Build(context, version.list, tables));
        Require(version.table != nullptr, version.name + ": no table was built");
        const auto windows = windowsOf(version.list);
        std::set<std::uint32_t> encoded;
        for (const auto& [window, held] : windows) {
            const Version* source = nullptr;
            for (const auto* share : shares) {
                if (share == nullptr) continue;
                const auto theirs = windowsOf(share->list);
                if (const auto found = theirs.find(window); found != theirs.end() && same(found->second, held)) {
                    source = share;
                    break;
                }
            }
            const auto leaf = version.table->LeafAddress(window);
            Require(leaf != 0u, version.name + ": window " + std::to_string(window) + " has no leaf");
            if (source != nullptr) {
                Require(leaf == source->table->LeafAddress(window), version.name + ": window " + std::to_string(window) + ", whose ranges " + source->name + " holds unchanged, did not share its leaf");
                continue;
            }
            encoded.insert(window);
            for (const auto* share : shares) {
                if (share != nullptr) Require(leaf != share->table->LeafAddress(window), version.name + ": window " + std::to_string(window) + ", whose ranges changed, kept the leaf of " + share->name);
            }
        }
        for (const auto* share : shares) {
            if (share == nullptr) continue;
            for (const auto& [window, held] : windowsOf(share->list)) {
                if (!windows.contains(window)) Require(version.table->LeafAddress(window) == 0u, version.name + ": window " + std::to_string(window) + ", which holds none of its ranges, has a leaf");
            }
        }
        std::string named;
        for (const auto window : encoded) named += " " + std::to_string(window);
        Require(encoded == changed, version.name + ": the windows encoded anew are" + named + ", not the windows the change touches");
        expectTranslations(version, "when built");
        return version;
    };
    const auto splice = [](const Ranges& resident, const Ranges& overrides) {
        Ranges spliced;
        for (const auto& range : resident) {
            auto cursor = range.begin;
            for (const auto& cut : overrides) {
                if (cut.end <= cursor || cut.begin >= range.end) continue;
                if (cut.begin > cursor) {
                    auto part = range;
                    part.begin = cursor;
                    part.end = cut.begin;
                    part.deviceAddress = range.deviceAddress + (cursor - range.begin);
                    spliced.push_back(part);
                }
                cursor = std::max(cursor, std::min(cut.end, range.end));
            }
            if (cursor < range.end) {
                auto part = range;
                part.begin = cursor;
                part.deviceAddress = range.deviceAddress + (cursor - range.begin);
                spliced.push_back(part);
            }
        }
        spliced.insert(spliced.end(), overrides.begin(), overrides.end());
        std::sort(spliced.begin(), spliced.end(), [](const Abi::Range& left, const Abi::Range& right) { return left.begin < right.begin; });
        return spliced;
    };
    constexpr std::uint64_t First = 0x200000000ull;
    constexpr std::uint64_t Second = 0x300000000ull;
    constexpr std::uint64_t Third = 0x600000000ull;
    constexpr std::uint32_t FirstWindow = static_cast<std::uint32_t>(First >> Abi::WindowShift);
    constexpr std::uint32_t SecondWindow = static_cast<std::uint32_t>(Second >> Abi::WindowShift);
    constexpr std::uint32_t ThirdWindow = static_cast<std::uint32_t>(Third >> Abi::WindowShift);
    std::uint64_t device = 0x1000000000ull;
    const auto at = [&](std::uint64_t begin, std::uint64_t end, std::uint32_t permissions) {
        const Abi::Range range{begin, end, device, permissions, 0u};
        device += 0x1000000ull;
        return range;
    };
    const Ranges baseList{
        at(First + 0x10000u, First + 0x50000u, Abi::Read),
        at(First + 0x60000u, First + 0x60100u, Rw),
        at(First + 0x60100u, First + 0x63000u, Rw),
        at(First + 0x63010u, First + 0x72000u, Abi::Read),
        at(First + 0x80000u, First + 0xa0000u, Rw),
        at(Second, Second + 0x30000u, Abi::Read),
        at(Second + 0x40000u, Second + 0x40800u, Rw),
        at(Second + (std::uint64_t{1} << Abi::WindowShift) - 0x20000u, Second + (std::uint64_t{1} << Abi::WindowShift) + 0x30000u, Rw),
        at(Second + (std::uint64_t{1} << Abi::WindowShift) + 0x40000u, Second + (std::uint64_t{1} << Abi::WindowShift) + 0x41000u, Abi::Read),
        at(Third + 0x100000u, Third + 0x100010u, Abi::Read),
        at(Third + 0x100020u, Third + 0x108000u, Rw),
        at(Third + 0x200000u, Third + 0x240000u, Abi::Read),
    };
    const auto base = build("the base table", baseList, {}, {FirstWindow, SecondWindow, SecondWindow + 1u, ThirdWindow});
    auto moved = base.list;
    moved[0].deviceAddress += 0x10000u;
    const auto first = build("the version after one range's device address changed", moved, {&base}, {FirstWindow});
    auto added = first.list;
    added.insert(added.begin() + 10, Abi::Range{Third + 0x100010u, Third + 0x100018u, 0x2000000000ull, Rw, 0u});
    const auto second = build("the version after a range was added inside a split page", added, {&first}, {ThirdWindow});
    auto removed = second.list;
    removed.erase(removed.begin() + 6);
    const auto third = build("the version after a range was removed", removed, {&second}, {SecondWindow});
    auto narrowed = third.list;
    narrowed[6].permissions = Abi::Read;
    const auto fourth = build("the version after the permissions of the range across a window boundary changed", narrowed, {&third}, {SecondWindow, SecondWindow + 1u});
    const Ranges overrides{{First + 0x20800u, First + 0x21000u, 0x3000000000ull, Rw, 0u}, {Third + 0x200000u, Third + 0x240000u, 0x3100000000ull, Rw, 0u}};
    const auto overridden = build("the override list spliced over the last version", splice(fourth.list, overrides), {nullptr, &fourth}, {FirstWindow, ThirdWindow});
    const Ranges moreOverrides{overrides[0], {Third + 0x210000u, Third + 0x220000u, 0x3200000000ull, Rw, 0u}};
    const auto reoverridden = build("another override list spliced over the last version", splice(fourth.list, moreOverrides), {&overridden, &fourth}, {ThirdWindow});
    for (const auto* version : {&base, &first, &second, &third, &fourth, &overridden, &reoverridden}) expectTranslations(*version, "after every later version was built");
}

static void RunBdaCachedReadTests(const Context& context, bool paged) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340000ULL;
    std::array<std::unique_ptr<Buffer>, 3> memory;
    for (std::uint32_t index = 0; index < memory.size(); ++index) {
        memory[index] = std::make_unique<Buffer>(context, 72, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        for (std::uint32_t byte = 0; byte < 72u; ++byte) memory[index]->Bytes()[byte] = static_cast<std::byte>(byte * 5u + index * 64u + 1u);
    }
    const std::array<Abi::Range, 3> ranges{{
        {guest, guest + 64, memory[0]->DeviceAddress(), Abi::Read, 0},
        {guest + 64, guest + 128, memory[1]->DeviceAddress(), Abi::Read, 0},
        {guest + 128, guest + 192, memory[2]->DeviceAddress() + 1u, Abi::Read, 0},
    }};
    const auto byteAt = [&](std::uint64_t address) {
        const auto range = static_cast<std::uint32_t>((address - guest) / 64u);
        return static_cast<std::uint32_t>(memory[range]->Bytes()[address - guest - range * 64u + (range == 2u ? 1u : 0u)]);
    };
    const std::vector<BdaCachedRead> reads{{guest, 1}, {guest + 4, 1}, {guest + 64, 1}, {guest + 60, 1}, {guest + 62, 1}, {guest + 68, 1}, {guest + 56, 3}, {guest + 132, 1}, {guest + 136, 2}, {guest + 4, 1}, {guest + 192, 1}};
    std::vector<std::uint32_t> expected;
    for (const auto& read : reads) {
        for (std::uint32_t dword = 0; dword < read.dwords; ++dword) {
            const auto address = read.address + dword * 4u;
            std::uint32_t value = 0;
            if (address + 4u <= guest + 192u) {
                for (std::uint32_t byte = 0; byte < 4u; ++byte) value |= byteAt(address + byte) << (byte * 8u);
            }
            expected.push_back(value);
        }
    }
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, expected.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    {
        const std::string label = std::string("BDA cached reads") + (paged ? " (paged): " : ": ");
        const auto pages = PagesFor(context, paged, ranges);
        const auto table = MakeBinding(context, paged, pages, ranges);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0xee, output.Bytes().size());
        Pipeline pipeline(context, MakeBdaCachedReadTestShader(reads), {table.get(), &fault, &output});
        pipeline.Run(1);
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        Require(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::Unmapped && report.address == guest + 192u && report.bytes == 1u && report.instruction == 0x1234, label + "the read past the last range did not publish its fault");
        for (std::size_t index = 0; index < expected.size(); ++index) {
            std::uint32_t value = 0;
            std::memcpy(&value, output.Bytes().data() + index * sizeof(value), sizeof(value));
            Require(value == expected[index], label + "dword " + std::to_string(index) + " read " + std::to_string(value) + ", expected " + std::to_string(expected[index]));
        }
    }
}

struct ImageDescriptor {
    std::uint64_t base;
    std::uint32_t format;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t tileMode;
    std::uint32_t type;
    std::uint32_t maxMip;
    std::uint32_t baseLevel;
    std::uint32_t lastLevel;
    std::uint32_t depth;
    std::uint32_t baseArray;
};

static std::array<std::uint32_t, 8> ImageWords(const ImageDescriptor& image) {
    return {static_cast<std::uint32_t>(image.base >> 8u), static_cast<std::uint32_t>((image.base >> 40u) & 0xffu) | image.format << 20u | ((image.width - 1u) & 3u) << 30u,
        ((image.width - 1u) >> 2u) | (image.height - 1u) << 14u, 0xfacu | image.baseLevel << 12u | image.lastLevel << 16u | image.tileMode << 20u | image.type << 28u,
        image.depth | image.baseArray << 16u, image.maxMip << 4u, 0u, 0u};
}

static std::uint32_t StandardBlockOffset(std::uint32_t elementBytes, bool large, std::uint32_t x, std::uint32_t y) {
    std::uint32_t offset = 0;
    switch (elementBytes) {
        case 1u: offset = ((y << 4) & 0x1f0u) ^ ((y << 5) & 0x400u) ^ (x & 0x00fu) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u); break;
        case 2u: offset = ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 1) & 0x00eu) ^ ((x << 4) & 0x080u) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u); break;
        case 4u: offset = ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 2) & 0x00cu) ^ ((x << 5) & 0x080u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u); break;
        case 8u: offset = ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 3) & 0x008u) ^ ((x << 5) & 0x0c0u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u); break;
        default: offset = ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 6) & 0x0c0u) ^ ((x << 7) & 0x200u) ^ ((x << 8) & 0x800u); break;
    }
    if (!large) return offset;
    switch (elementBytes) {
        case 1u: return offset ^ ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 6) & 0x1000u) ^ ((y << 7) & 0x4000u);
        case 2u: return offset ^ ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
        case 4u: return offset ^ ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
        case 8u: return offset ^ ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
        default: return offset ^ ((x << 9) & 0x2000u) ^ ((x << 10) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
    }
}

static std::uint64_t DriverElementAddress(const std::array<std::uint32_t, 8>& words, std::uint32_t x, std::uint32_t y, std::uint32_t layer, std::uint32_t lod) {
    const auto descriptor = DecodeTextureResource(words);
    const auto geometry = DescribeSurface(descriptor);
    const auto elementBytes = BytesPerElement(descriptor.format);
    const auto& mip = geometry.mips.at(descriptor.baseLevel + lod);
    const bool arrayed = descriptor.dimension == TextureDimension::k2DArray || descriptor.dimension == TextureDimension::k1DArray || descriptor.dimension == TextureDimension::kCube;
    const auto start = descriptor.baseAddress + geometry.GuestLayerOffset(arrayed ? descriptor.baseArray + layer : 0u) + mip.tiledOffset;
    if (descriptor.tileMode == TextureTileMode::kLinear) return start + static_cast<std::uint64_t>(y) * mip.pitchBytes + static_cast<std::uint64_t>(x) * elementBytes;
    const auto block = ThinBlockLayout(descriptor.tileMode, elementBytes);
    auto swizzleX = x;
    auto swizzleY = y;
    std::uint64_t blockStart = start;
    if (mip.tail) {
        swizzleX += mip.tailX;
        swizzleY += mip.tailY;
    } else {
        blockStart += (static_cast<std::uint64_t>(y / block[2]) * mip.blocksPerRow + x / block[1]) * block[0];
    }
    std::uint32_t offset = 0;
    if (const auto mode = EquationSwizzleMode(descriptor.tileMode); mode != 0u) {
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        Require(equation != nullptr, "no swizzle equation for a tested image");
        for (std::uint32_t bit = 0; bit < 16u; ++bit) {
            const auto mask = equation->bits[bit];
            const auto selected = (swizzleX & (mask & 0xfffu)) ^ ((swizzleY << 12u) & (mask & 0xfff000u)) ^ (((arrayed ? descriptor.baseArray + layer : 0u) << 24u) & (mask & 0xff000000u));
            offset |= (static_cast<std::uint32_t>(std::popcount(selected)) & 1u) << bit;
        }
        offset ^= descriptor.pipeBankXor;
    } else {
        offset = StandardBlockOffset(elementBytes, block[0] > 4096u, swizzleX, swizzleY);
    }
    return blockStart + (offset & (block[0] - 1u));
}

static void RunImageStoreRecordTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    namespace Pages = ShaderRecompiler::StorageImagePages;
    struct Case {
        const char* name;
        ImageDescriptor image;
        ImageStoreRect rect;
        bool placed;
    };
    const std::array<Case, 22> cases{{
        {"4KB S 8-bit", {0x200000000ull, 1, 256, 200, 5, 9, 0, 0, 0, 0, 0}, {13, 7, 128, 70, 0, 0}, true},
        {"64KB S_X 32-bit with a pipe-bank XOR, level 1", {0x210003000ull, 56, 300, 150, 25, 9, 4, 0, 4, 0, 0}, {5, 3, 140, 70, 0, 1}, true},
        {"64KB S_X 32-bit tail level", {0x210003000ull, 56, 300, 150, 25, 9, 8, 0, 8, 0, 0}, {0, 0, 2, 2, 0, 7}, true},
        {"64KB S 16-bit array layer 2, level 5", {0x220000000ull, 13, 128, 128, 9, 13, 6, 0, 6, 2, 0}, {0, 0, 4, 4, 2, 5}, true},
        {"64KB S 16-bit array layer 1, level 0", {0x220000000ull, 13, 128, 128, 9, 13, 6, 0, 6, 2, 0}, {17, 9, 100, 110, 1, 0}, true},
        {"64KB D_X 64-bit", {0x230000000ull, 62, 64, 64, 26, 9, 0, 0, 0, 0, 0}, {0, 0, 64, 64, 0, 0}, true},
        {"64KB R_X 128-bit with a pipe-bank XOR", {0x24000a000ull, 77, 96, 40, 27, 9, 0, 0, 0, 0, 0}, {3, 1, 90, 37, 0, 0}, true},
        {"64KB Z_X 8-bit mips through the tail", {0x250000000ull, 1, 200, 100, 24, 9, 7, 0, 7, 0, 0}, {0, 0, 8, 8, 0, 6}, true},
        {"64KB Z_X 8-bit level 2", {0x250000000ull, 1, 200, 100, 24, 9, 7, 0, 7, 0, 0}, {1, 1, 48, 24, 0, 2}, true},
        {"linear 32-bit level 2", {0x260000100ull, 20, 512, 16, 0, 9, 2, 0, 2, 0, 0}, {10, 0, 110, 4, 0, 2}, true},
        {"linear 1D array layer 3", {0x270000000ull, 1, 200, 1, 0, 12, 0, 0, 0, 3, 0}, {7, 0, 150, 1, 3, 0}, true},
        {"256B S 16-bit", {0x280000000ull, 7, 80, 60, 1, 9, 0, 0, 0, 0, 0}, {0, 0, 80, 60, 0, 0}, true},
        {"256B D 32-bit", {0x281000000ull, 56, 70, 50, 2, 9, 0, 0, 0, 0, 0}, {5, 5, 60, 40, 0, 0}, true},
        {"4KB D 32-bit", {0x290000000ull, 20, 130, 90, 6, 9, 0, 0, 0, 0, 0}, {0, 0, 130, 90, 0, 0}, true},
        {"4KB S_X 16-bit with a pipe-bank XOR", {0x2a0000700ull, 7, 100, 100, 21, 9, 0, 0, 0, 0, 0}, {0, 0, 100, 100, 0, 0}, true},
        {"4KB D_X 64-bit view base level 1", {0x2b0000000ull, 64, 120, 80, 22, 9, 3, 1, 3, 0, 0}, {0, 0, 60, 40, 0, 0}, true},
        {"64KB D 32-bit", {0x2c0000000ull, 20, 200, 120, 10, 9, 0, 0, 0, 0, 0}, {0, 0, 200, 120, 0, 0}, true},
        {"64KB D_T 16-bit", {0x2d0000000ull, 7, 300, 300, 18, 9, 0, 0, 0, 0, 0}, {40, 50, 200, 210, 0, 0}, true},
        {"rectangle past the level's edge", {0x2e0000000ull, 20, 100, 60, 25, 9, 0, 0, 0, 0, 0}, {80, 40, 50, 50, 0, 0}, true},
        {"4KB Z (no equation)", {0x2f0000000ull, 20, 64, 64, 4, 9, 0, 0, 0, 0, 0}, {0, 0, 8, 8, 0, 0}, false},
        {"3D 64KB S", {0x300000000ull, 20, 64, 64, 9, 10, 0, 0, 0, 7, 0}, {0, 0, 8, 8, 0, 0}, false},
        {"96-bit elements", {0x310000000ull, 72, 64, 64, 9, 9, 0, 0, 0, 0, 0}, {0, 0, 8, 8, 0, 0}, false},
    }};
    Buffer table(context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    constexpr std::uint64_t page = std::uint64_t{1} << Abi::WrittenPageShift;
    for (const auto& test : cases) {
        const auto words = ImageWords(test.image);
        const std::string label = std::string("image store records, ") + test.name + ": ";
        std::set<std::uint64_t> expected;
        for (std::uint32_t row = 0; row < test.rect.height; ++row) {
            for (std::uint32_t column = 0; column < test.rect.width; ++column) {
                const auto x = test.rect.x + column;
                const auto y = test.rect.y + row;
                const auto store = Pages::Locate(words, x, y, test.rect.layer, test.rect.lod);
                Require((store.placement != Pages::Placement::Unplaced) == test.placed, label + "the reference placed a store it should not, or left one unplaced");
                if (store.placement != Pages::Placement::Recorded) continue;
                const auto level = test.image.baseLevel + test.rect.lod;
                Require(x < std::max(test.image.width >> level, 1u) && y < std::max(test.image.height >> level, 1u), label + "the reference recorded a store outside the level");
                const auto driver = DriverElementAddress(words, x, y, test.rect.layer, test.rect.lod);
                Require(store.address >> Abi::WrittenPageShift == driver >> Abi::WrittenPageShift, label + "element (" + std::to_string(x) + ", " + std::to_string(y) + ") is in page 0x" + std::to_string(driver >> Abi::WrittenPageShift) + " of the driver's layout, not 0x" + std::to_string(store.address >> Abi::WrittenPageShift));
                expected.insert(driver >> Abi::WrittenPageShift);
            }
        }
        for (const bool subgroups : {false, true}) {
        const std::string mode = label + (subgroups ? "subgroup notes: " : "lane notes: ");
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        Pipeline pipeline(context, MakeImageStoreRecordTestShader(words, test.rect, subgroups), {&table, &fault, &output});
        pipeline.Run((test.rect.width * test.rect.height + 63u) / 64u);
        std::vector<std::uint32_t> recordWords(Abi::FaultBufferBytes / sizeof(std::uint32_t));
        std::memcpy(recordWords.data(), fault.Bytes().data(), Abi::FaultBufferBytes);
        Require(recordWords[Abi::WrittenOverflowWord] == 0u, mode + "the record overflowed");
        std::set<std::uint64_t> recorded;
        Require(Abi::ForEachWrittenImageRange(recordWords.data(), [&](std::uint64_t begin, std::size_t bytes) {
            for (std::uint64_t at = begin; at < begin + bytes; at += page) recorded.insert(at >> Abi::WrittenPageShift);
        }), mode + "invalid written-page record");
        bool memory = false;
        Require(Abi::ForEachWrittenRange(recordWords.data(), [&](std::uint64_t, std::size_t) { memory = true; }) && !memory, mode + "image stores recorded memory pages");
        Require(Abi::WrittenImagesWhole(recordWords.data()) == !test.placed, mode + (test.placed ? "a placed image store kept its image whole" : "an unplaced image store did not keep its image whole"));
        Require(recorded == expected, mode + "the record names " + std::to_string(recorded.size()) + " pages, the driver's layout " + std::to_string(expected.size()));
        }
    }
}

static void RunBdaPlacementTests(const Context& device) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto context = device;
    context.bdaArena = std::make_shared<BdaArena>();
    constexpr VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const auto table = BdaArena::Take(context, 256);
    Require((table.slab->Properties() & deviceLocal) != 0u, "a BDA table slice was placed outside device-local memory");
    if (BdaArena::MappedTables(context)) {
        Require((table.slab->Properties() & hostVisible) != 0u && table.host == nullptr && table.data == table.slab->Bytes().data() + table.offset, "a BDA table slice on a device with device-local host-visible memory was not written in place");
    } else {
        Require(table.host != nullptr && (table.host->Properties() & hostVisible) != 0u && (table.slab->Properties() & hostVisible) == 0u, "a BDA table slice on a device without device-local host-visible memory has no staging slice");
    }
    const auto record = BdaArena::Take(context, Abi::FaultBufferBytes, BdaArena::Use::Record);
    Require(record.slab->Properties() == deviceLocal, "a fault and written-page record was placed outside device-local memory");
    Require(record.host != nullptr && (record.host->Properties() & hostVisible) != 0u && record.data == record.host->Bytes().data() + record.hostOffset, "a fault and written-page record has no host copy for the CPU to read after its batch");
    constexpr std::uint64_t base = 0x123450000ull;
    const std::array ranges{Abi::Range{base, base + 0x30000u, 0x2000000000ull, Abi::Read, 0u}};
    const auto pages = BdaPageTable::Build(context, ranges);
    Require(pages->Properties() == deviceLocal, "a BDA page table was placed outside device-local memory");
    Require(pages->NeedsUploads(0) && pages->NeedsUploads(1), "a new page table needed no upload");
    CommandBatch batch(context);
    Require(pages->RecordUploads(context, batch.Handle(), 0), "a new page table recorded no upload");
    Require(!pages->NeedsUploads(0) && !pages->RecordUploads(context, batch.Handle(), 0), "a page table was uploaded twice on one stream");
    Require(pages->NeedsUploads(1), "a page table uploaded on one stream counted as uploaded on another");
    RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    batch.SubmitAndWait();
    const std::array<std::array<std::uint64_t, 3>, 1> probes{{{base + 0x10008u, 4u, Abi::Read}}};
    const auto translated = TranslateThroughTable(context, pages->Address(), probes);
    Require(translated.size() == 1u && translated.front().device == 0x2000000000ull + 0x10008u && translated.front().reason == 0u, "a dispatch did not read the uploaded page table");
}

void RunBdaExecutionTests(const Context& context) {
    RunBdaPlacementTests(context);
    RunBdaPageTableTests(context);
    RunBdaPageTableVersionTests(context);
    for (const bool paged : {false, true}) RunBdaReadTests(context, paged);
    for (const bool paged : {false, true}) RunBdaCachedReadTests(context, paged);
    RunImageStoreRecordTests(context);
    RunBdaWrittenPageTests(context);
}

void RunBdaStoreExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    constexpr std::uint64_t guest = 0x7fff1235fffdULL;
    constexpr std::uint32_t writable = Abi::Read | Abi::Write;
    std::array<Abi::Range, 2> ranges{};
    std::vector<Abi::Grant> grants;
    bool runtime = false;
    bool paged = false;
    bool enabled = true;
    const auto run = [&](const char* name, std::uint64_t address, std::uint32_t bits, bool coherent, const std::array<std::uint8_t, 5>& expected, std::initializer_list<std::uint64_t> writtenAddresses, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const std::string label = std::string(runtime ? "BDA runtime write " : "BDA store ") + name + (coherent ? " coherent" : "") + (paged ? " paged: " : ": ");
        std::memset(first.Bytes().data(), 0xcd, first.Bytes().size());
        std::memset(second.Bytes().data(), 0xcd, second.Bytes().size());
        const auto pages = PagesFor(context, paged, ranges);
        const auto table = MakeBinding(context, paged, pages, ranges, grants);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0, output.Bytes().size());
        Pipeline pipeline(context, runtime ? MakeBdaWriteTestShader(address, 0xa1b2c3d4u, bits, coherent, enabled) : MakeBdaStoreTestShader(address, 0xa1b2c3d4u, bits, coherent), {table.get(), &fault, &output});
        pipeline.Run(1);
        std::array<std::uint8_t, 5> actual{};
        std::memcpy(actual.data(), first.Bytes().data(), 2);
        std::memcpy(actual.data() + 2, second.Bytes().data(), 3);
        Require(actual == expected, label + "incorrect stored bytes or writes outside the mapped span");
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        if (reason == Abi::FaultReason{}) {
            Require(report.state == Abi::FaultState::Empty && report.reason == Abi::FaultReason{} && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.reserved == 0, label + "mapped store published a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.stage == static_cast<std::uint32_t>(ShaderRecompiler::IrShaderStage::Compute) && report.instruction == 0x1234 && report.reserved == 0, label + "incorrect first fault metadata");
        }
        std::array<std::uint32_t, Abi::FaultBufferBytes / sizeof(std::uint32_t)> words{};
        std::memcpy(words.data(), fault.Bytes().data(), sizeof(words));
        Require(words[Abi::WrittenOverflowWord] == 0, label + "written-page table overflowed");
        std::set<std::uint64_t> recordedPages;
        Require(Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t address, std::size_t bytes) {
            for (std::size_t offset = 0; offset < bytes; offset += std::size_t{1} << Abi::WrittenPageShift) {
                Require(recordedPages.insert((address + offset) >> Abi::WrittenPageShift).second, label + "written-page table contains duplicate pages");
            }
        }), label + "invalid written-page record");
        std::set<std::uint64_t> expectedPages;
        std::set<std::uint32_t> expectedChunks;
        for (const auto writtenAddress : writtenAddresses) {
            expectedPages.insert(writtenAddress >> Abi::WrittenPageShift);
            expectedChunks.insert(Abi::WrittenChunkKey(writtenAddress));
        }
        Require(recordedPages == expectedPages, label + "incorrect written-page set");
        Require(words[Abi::WrittenCountWord] == expectedChunks.size(), label + "incorrect written-chunk claim count");
        std::uint32_t completed = 0;
        std::memcpy(&completed, output.Bytes().data(), sizeof(completed));
        Require(completed == 1, label + "store did not finish its per-byte accesses");
    };
    for (const auto& [coherent, writes, withPages] : std::initializer_list<std::tuple<bool, bool, bool>>{{false, false, false}, {false, true, false}, {true, false, false}, {true, true, false}, {false, false, true}, {false, true, true}, {true, false, true}, {true, true, true}}) {
        runtime = writes;
        paged = withPages;
        ranges = {{{guest, guest + 2, first.DeviceAddress(), writable, 0}, {guest + 2, guest + 5, second.DeviceAddress(), writable, 0}}};
        run("byte", guest + 1, 8, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1});
        run("split short", guest + 1, 16, coherent, {0xcd, 0xd4, 0xc3, 0xcd, 0xcd}, {guest + 1, guest + 2});
        run("split unaligned dword", guest, 32, coherent, {0xd4, 0xc3, 0xb2, 0xa1, 0xcd}, {guest, guest + 3});
        if (runtime) {
            enabled = false;
            run("disabled split unaligned dword", guest, 32, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {});
            run("disabled byte", guest + 1, 8, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {});
            enabled = true;
        }
        ranges[1].permissions = Abi::Read;
        run("read-only byte", guest + 2, 8, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only short suffix", guest + 1, 16, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only suffix", guest, 32, coherent, {0xd4, 0xc3, 0xcd, 0xcd, 0xcd}, {guest}, Abi::FaultReason::Permission, guest + 2, 1);
        if (!coherent) {
            grants = {{guest + 2, guest + 5}};
            run("granted byte", guest + 2, 8, coherent, {0xcd, 0xcd, 0xd4, 0xcd, 0xcd}, {guest + 2});
            run("granted suffix", guest, 32, coherent, {0xd4, 0xc3, 0xb2, 0xa1, 0xcd}, {guest, guest + 3});
            grants = {{guest + 3, guest + 5}};
            run("partly granted suffix", guest, 32, coherent, {0xd4, 0xc3, 0xcd, 0xa1, 0xcd}, {guest, guest + 3}, Abi::FaultReason::Permission, guest + 2, 1);
            grants = {{guest, guest + 3}};
            run("grant ending inside a store", guest + 1, 32, coherent, {0xcd, 0xd4, 0xc3, 0xcd, 0xcd}, {guest + 1, guest + 2}, Abi::FaultReason::Permission, guest + 3, 1);
            grants.clear();
        }
        ranges[0].end = guest + 1;
        ranges[1].permissions = writable;
        run("unmapped middle byte", guest, 32, coherent, {0xd4, 0xcd, 0xb2, 0xa1, 0xcd}, {guest, guest + 3}, Abi::FaultReason::Unmapped, guest + 1, 1);
        ranges = {{{guest + 3, guest + 5, first.DeviceAddress(), writable, 0}, {guest + 5, guest + 8, second.DeviceAddress(), writable, 0}}};
        run("split aligned dword", guest + 3, 32, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Unmapped, guest + 3, 4);
        if (runtime) {
            enabled = false;
            run("disabled unmapped dword", guest, 32, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {});
            enabled = true;
        }
    }
}
