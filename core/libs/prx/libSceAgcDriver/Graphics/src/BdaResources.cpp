#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "ImageTableAbi.hpp"
#include "DeviceImageTableAbi.hpp"
#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <iomanip>
#include <sstream>

namespace AgcDriver::Graphics {

namespace {

std::uint64_t NextStoreRecord() {
    static std::atomic<std::uint64_t> next{0};
    return next.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> WrittenImageRanges(const std::uint32_t* words) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    if (words[Abi::WrittenOverflowWord] != 0 || Abi::WrittenImagesWhole(words) || !Abi::ForEachWrittenImageRange(words, [&](std::uint64_t begin, std::size_t bytes) { ranges.emplace_back(begin, begin + bytes); })) return std::nullopt;
    return ranges;
}

}

bool BdaArena::MappedTables(const Context& context) {
    constexpr VkMemoryPropertyFlags mapped = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t type = 0; type < context.memory.memoryTypeCount; ++type) {
        if ((context.memory.memoryTypes[type].propertyFlags & mapped) == mapped) return true;
    }
    return false;
}

std::pair<std::shared_ptr<Buffer>, VkDeviceSize> BdaArena::take(const std::shared_ptr<BdaArena>& self, const Context& context, Pool pool, std::size_t bytes, std::vector<std::shared_ptr<Buffer>>& retired) {
    const VkBufferUsageFlags addressable = context.bufferDeviceAddress ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u;
    constexpr VkBufferUsageFlags transfers = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags properties = host;
    std::size_t slabBytes = SlabBytes;
    if (pool == TableDevice) {
        usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | addressable | transfers;
        properties = MappedTables(context) ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | host : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        slabBytes = TableSlabBytes;
    } else if (pool == TableStaging) {
        slabBytes = TableSlabBytes;
    } else if (pool == RecordDevice) {
        usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | addressable | transfers;
        properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
        usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    auto slabContext = context;
    slabContext.bdaArena.reset();
    if (bytes > slabBytes) return {std::make_shared<Buffer>(slabContext, bytes, usage, properties), 0};
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16);
    auto& slabs = pools[pool];
    auto offset = (slabs.used + alignment - 1u) / alignment * alignment;
    if (slabs.current == nullptr || offset + bytes > slabBytes) {
        retired.push_back(std::move(slabs.current));
        std::unique_ptr<Buffer> slab;
        if (!slabs.free.empty()) {
            slab = std::move(slabs.free.back());
            slabs.free.pop_back();
        } else {
            slab = std::make_unique<Buffer>(slabContext, slabBytes, usage, properties);
        }
        const std::weak_ptr<BdaArena> owner = self;
        slabs.current = std::shared_ptr<Buffer>(slab.release(), [owner, pool](Buffer* released) {
            if (const auto live = owner.lock()) {
                std::lock_guard returned(live->mutex);
                live->pools[pool].free.emplace_back(released);
                return;
            }
            delete released;
        });
        slabs.used = 0;
        offset = 0;
    }
    slabs.used = offset + bytes;
    return {slabs.current, offset};
}

BdaArena::Slice BdaArena::Take(const Context& context, std::size_t bytes, Use use) {
    Require(bytes != 0, "an empty BDA table or fault record");
    if (!context.bdaArena) context.bdaArena = std::make_shared<BdaArena>();
    const auto arena = context.bdaArena;
    const bool mapped = use == Use::Table && MappedTables(context);
    std::vector<std::shared_ptr<Buffer>> retired;
    std::lock_guard lock(arena->mutex);
    Slice slice;
    slice.bytes = bytes;
    std::tie(slice.slab, slice.offset) = arena->take(arena, context, use == Use::Table ? TableDevice : RecordDevice, bytes, retired);
    if (mapped) {
        slice.data = slice.slab->Bytes().data() + slice.offset;
        return slice;
    }
    std::tie(slice.host, slice.hostOffset) = arena->take(arena, context, use == Use::Table ? TableStaging : RecordHost, bytes, retired);
    slice.data = slice.host->Bytes().data() + slice.hostOffset;
    return slice;
}

bool BdaArena::RecordUpload(const Context& context, VkCommandBuffer commands, const Slice& slice) {
    if (slice.host == nullptr) return false;
    const VkBufferCopy region{slice.hostOffset, slice.offset, slice.bytes};
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, slice.host->Handle(), slice.slab->Handle(), 1, &region);
    return true;
}

BatchFaultRecord::BatchFaultRecord(const Context& context, std::uint32_t stream, std::uint64_t batch) : slice(BdaArena::Take(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, BdaArena::Use::Record)), stream(stream), batch(batch), id(NextStoreRecord()) {
    std::memset(slice.data, 0, ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t));
}

VkDescriptorBufferInfo BatchFaultRecord::Descriptor() const {
    return {slice.slab->Handle(), slice.offset, ShaderRecompiler::BdaAbi::FaultBufferBytes};
}

bool BatchFaultRecord::Admit(std::uint32_t written) {
    if (written > Capacity - chunks) return false;
    chunks += written;
    return true;
}

BdaResources::BdaResources(const Context& context, std::shared_ptr<BatchFaultRecord> shared) : shared(std::move(shared)), deviceImages(context.deviceImages) {
    static_assert(std::endian::native == std::endian::little);
    Require(ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA fault buffer exceeds storage buffer range limit");
    if (this->shared != nullptr) return;
    fault = BdaArena::Take(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, BdaArena::Use::Record);
    std::memset(fault.data, 0, ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t));
    ownRecord = NextStoreRecord();
}

void BdaResources::UseRecord(const Context& context, std::shared_ptr<BatchFaultRecord> record) {
    shared = std::move(record);
    if (shared != nullptr) {
        fault = {};
        return;
    }
    if (fault.slab != nullptr) return;
    fault = BdaArena::Take(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, BdaArena::Use::Record);
    std::memset(fault.data, 0, ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t));
    ownRecord = NextStoreRecord();
}

BdaResources::BdaResources(const Context& context, const GuestBufferMemory& memory, std::span<const ShaderRecompiler::BdaAbi::Range> overrides, std::shared_ptr<BatchFaultRecord> shared) : BdaResources(context, std::move(shared)) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<Abi::Range> mapped(overrides.begin(), overrides.end());
    std::sort(mapped.begin(), mapped.end(), [](const Abi::Range& left, const Abi::Range& right) { return left.begin < right.begin; });
    pages = mapped.empty() ? memory.PageTable(context) : memory.PageTableWith(context, mapped);
    ownRanges = memory.OwnRanges();
    writeGrants = memory.WriteGrants();
    Require(ownRanges.size() <= std::numeric_limits<std::uint32_t>::max() && writeGrants.size() <= std::numeric_limits<std::uint32_t>::max(), "BDA table range count overflow");
    const auto tableBytes = sizeof(Abi::Header) + ownRanges.size() * sizeof(Abi::Range) + writeGrants.size() * sizeof(Abi::Grant);
    Require(tableBytes <= context.limits.maxStorageBufferRange, "BDA descriptors exceed storage buffer range limit");
    for (const auto& range : ownRanges) Require((range.permissions & Abi::Write) == 0 || range.end <= Abi::WrittenAddressLimit, "a writable BDA range ends past the addresses the written-page record can hold");
    for (const auto& grant : writeGrants) Require(grant.end <= Abi::WrittenAddressLimit, "a BDA write grant ends past the addresses the written-page record can hold");
    ValidateBdaRangeList(ownRanges);
    ValidateBdaGrants(writeGrants);
    table = BdaArena::Take(context, tableBytes);
    const auto* residency = ResidencyDriver::Active();
    const Abi::Header header{Abi::TableVersion, static_cast<std::uint32_t>(ownRanges.size()), static_cast<std::uint32_t>(writeGrants.size()), 0, pages != nullptr ? pages->Address() : 0u, residency != nullptr ? residency->TouchTableAddress() : 0u};
    auto* bytes = table.data;
    std::memcpy(bytes, &header, sizeof(header));
    if (!ownRanges.empty()) std::memcpy(bytes + sizeof(header), ownRanges.data(), ownRanges.size() * sizeof(Abi::Range));
    if (!writeGrants.empty()) std::memcpy(bytes + sizeof(header) + ownRanges.size() * sizeof(Abi::Range), writeGrants.data(), writeGrants.size() * sizeof(Abi::Grant));
}

void BdaResources::PrepareUse(const Context& context, VkCommandBuffer commands, Recorder* recorder) {
    const auto& record = shared != nullptr ? shared->Record() : fault;
    bool transfers = false;
    if (record.slab != nullptr) {
        constexpr VkDeviceSize cleared = ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t);
        if (recorder != nullptr) {
            transfers = recorder->UseRecord(record.slab, record.offset, cleared, record.host, record.hostOffset, record.bytes);
        } else {
            context.Resolved(&DeviceFunctions::cmdFillBuffer, "vkCmdFillBuffer")(commands, record.slab->Handle(), record.offset, cleared, 0u);
            transfers = true;
        }
    }
    const auto stream = recorder != nullptr ? recorder->CallerStream() : BdaPageTable::EveryStream;
    if (table.host != nullptr) {
        const bool fresh = stream == BdaPageTable::EveryStream || (tableStreams & (1u << stream)) == 0u;
        if (stream != BdaPageTable::EveryStream) tableStreams |= 1u << stream;
        if (fresh) transfers = BdaArena::RecordUpload(context, commands, table) || transfers;
    }
    if (pages != nullptr) transfers = pages->RecordUploads(context, commands, stream) || transfers;
    if (transfers) RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
}

bool BdaResources::NeedsTransfers(const Recorder& recorder) const {
    const auto& record = shared != nullptr ? shared->Record() : fault;
    if (record.slab != nullptr && !recorder.RecordUsed(record.slab.get(), record.offset)) return true;
    const auto stream = recorder.CallerStream();
    if (table.host != nullptr && (tableStreams & (1u << stream)) == 0u) return true;
    return pages != nullptr && pages->NeedsUploads(stream);
}

void BdaResources::FinishUse(const Context& context, VkCommandBuffer commands) const {
    const auto& record = shared != nullptr ? shared->Record() : fault;
    if (record.slab == nullptr || record.host == nullptr) return;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const VkBufferCopy region{record.offset, record.hostOffset, record.bytes};
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, record.slab->Handle(), record.host->Handle(), 1, &region);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
}

VkDescriptorBufferInfo BdaResources::Table() const {
    Require(table.slab != nullptr, "BDA page table was not requested");
    return {table.slab->Handle(), table.offset, table.bytes};
}

VkDescriptorBufferInfo BdaResources::Fault() const {
    if (shared != nullptr) return shared->Descriptor();
    return {fault.slab->Handle(), fault.offset, ShaderRecompiler::BdaAbi::FaultBufferBytes};
}

namespace {
std::atomic<bool> loopGuardTripped{false};

bool loopGuardReported(const ShaderRecompiler::BdaAbi::Fault& report) {
    if (report.reason != ShaderRecompiler::BdaAbi::FaultReason::LoopLimit) return false;
    loopGuardTripped.store(true, std::memory_order_relaxed);
    std::fprintf(stderr, "[gpu] loop guard: the loop exit at pc 0x%x of shader 0x%llx ran past %u evaluations\n", report.instruction, static_cast<unsigned long long>(report.address), report.bytes);
    return true;
}
}

bool LoopGuardTripped() {
    return loopGuardTripped.load(std::memory_order_relaxed);
}

bool BdaResources::published(const ShaderRecompiler::BdaAbi::Fault& report) {
    if (report.state == ShaderRecompiler::BdaAbi::FaultState::Empty) {
        Require(static_cast<std::uint32_t>(report.reason) == 0 && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.reserved == 0, "BDA fault record has data without publication");
        return false;
    }
    Require(report.state == ShaderRecompiler::BdaAbi::FaultState::Ready && (report.reserved == 0 || report.reason == ShaderRecompiler::BdaAbi::FaultReason::InvalidBufferDescriptor), "incomplete or invalid BDA fault record");
    return true;
}

void BdaResources::CheckFault() const {
    if (shared != nullptr) return;
    const BdaResources* self = this;
    auto* words = reinterpret_cast<std::uint32_t*>(fault.data);
    const bool recorded = ownRecord != 0 && StorageTexture::StoresRecordedFor(ownRecord);
    const auto images = recorded ? WrittenImageRanges(words) : std::nullopt;
    markWrittenPages(words, std::span(&self, 1));
    if (recorded) StorageTexture::ResolveRecordedStores(ownRecord, images ? &*images : nullptr);
    ShaderRecompiler::BdaAbi::Fault report{};
    std::memcpy(&report, fault.data, sizeof(report));
    if (!published(report)) return;
    if (loopGuardReported(report)) {
        std::memset(fault.data, 0, ShaderRecompiler::BdaAbi::FaultBufferBytes);
        return;
    }
    throwFault(report);
}

namespace {
std::uint32_t RecordedStage(std::uint32_t stage) {
    using Ir = ShaderRecompiler::IrShaderStage;
    using Stage = ShaderRecompiler::ShaderStage;
    switch (static_cast<Ir>(stage)) {
        case Ir::Vertex: case Ir::Fetch: return static_cast<std::uint32_t>(Stage::Vertex);
        case Ir::Pixel: return static_cast<std::uint32_t>(Stage::Fragment);
        case Ir::Compute: return static_cast<std::uint32_t>(Stage::Compute);
        case Ir::Mesh: return static_cast<std::uint32_t>(Stage::Mesh);
        case Ir::Local: return static_cast<std::uint32_t>(Stage::Local);
        case Ir::TessellationControl: return static_cast<std::uint32_t>(Stage::TessellationControl);
        case Ir::TessellationEvaluation: return static_cast<std::uint32_t>(Stage::TessellationEvaluation);
        default: return stage;
    }
}
}

bool BdaResources::owns(const ShaderRecompiler::BdaAbi::Fault& report) const {
    if (report.reason != ShaderRecompiler::BdaAbi::FaultReason::ImageTable) return false;
    return (report.bytes & ShaderRecompiler::DeviceImageTableAbi::FaultFlag) != 0u ? deviceImages != nullptr : !imageTables.empty();
}

void BdaResources::throwFault(const ShaderRecompiler::BdaAbi::Fault& report) const {
    Require(report.reason != ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle, "rect-list requires finite nondegenerate axis-aligned positions with equal positive W");
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::ImageTable) {
        const auto table = std::find_if(imageTables.begin(), imageTables.end(), [&](const ImageTableFaults& candidate) { return candidate.stage == report.stage; });
        const auto* tables = table != imageTables.end() ? &*table : imageTables.size() == 1 ? &imageTables.front() : nullptr;
        std::ostringstream message;
        message << "image table: shader 0x" << std::hex << (tables != nullptr ? tables->shader : 0u) << std::dec << " stage " << report.stage << " pc 0x" << std::hex << report.instruction;
        if ((report.bytes & ShaderRecompiler::DeviceImageTableAbi::FaultFlag) != 0u) {
            const auto rejection = deviceImages != nullptr ? deviceImages->FindRejection(report.bytes & ~ShaderRecompiler::DeviceImageTableAbi::FaultFlag) : std::nullopt;
            message << " selected a descriptor the device image table rejected" << std::dec;
            if (rejection.has_value()) {
                message << " (words";
                for (const auto word : rejection->words) message << ' ' << std::hex << std::setw(8) << std::setfill('0') << word;
                message << std::dec << "): " << ShaderRecompiler::ImageTableAbi::PoisonReasonName(static_cast<ShaderRecompiler::ImageTableAbi::PoisonReason>(rejection->reason)) << ": " << rejection->message;
            } else {
                message << ": rejection " << (report.bytes & ~ShaderRecompiler::DeviceImageTableAbi::FaultFlag);
            }
            throw std::runtime_error(message.str());
        }
        message << " selected the descriptor at 0x" << report.address << std::dec;
        if (tables != nullptr && report.bytes < tables->poison.size()) {
            const auto& poison = tables->poison[report.bytes];
            if (poison.dwordCount != 0) {
                message << " (words";
                for (std::uint32_t index = 0; index < poison.dwordCount && index < poison.words.size(); ++index) message << ' ' << std::hex << std::setw(8) << std::setfill('0') << poison.words[index];
                message << std::dec << ")";
            }
            message << ": " << ShaderRecompiler::ImageTableAbi::PoisonReasonName(static_cast<ShaderRecompiler::ImageTableAbi::PoisonReason>(poison.reason));
            if (report.bytes < tables->messages.size() && !tables->messages[report.bytes].empty()) message << ": " << tables->messages[report.bytes];
        } else {
            message << ": poison entry " << report.bytes;
        }
        throw std::runtime_error(message.str());
    }
    std::ostringstream message;
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::InvalidBufferDescriptor) {
        const auto stage = RecordedStage(report.stage & ((1u << ShaderRecompiler::BdaAbi::DescriptorFaultResourceShift) - 1u));
        const auto resource = report.stage >> ShaderRecompiler::BdaAbi::DescriptorFaultResourceShift;
        std::uint64_t shader = 0;
        for (const auto& [named, address] : stageShaders) {
            if (named == stage) shader = address;
        }
        char text[192];
        std::snprintf(text, sizeof(text), "shader 0x%llx stage %u pc 0x%x: buffer descriptor uses an unsupported type (buffer %u V# %08x %08x %08x %08x)", static_cast<unsigned long long>(shader), stage, report.instruction, resource, static_cast<std::uint32_t>(report.address), static_cast<std::uint32_t>(report.address >> 32u), report.bytes, report.reserved);
        throw std::runtime_error(text);
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::UnlinkedCall) {
        message << "shader stage " << report.stage << " called address 0x" << std::hex << report.address << " at program counter 0x" << report.instruction << ", which no linked callee of that call matches";
        throw std::runtime_error(message.str());
    }
    message << "BDA access failed: address=0x" << std::hex << report.address << " instruction=0x" << report.instruction << std::dec << " bytes=" << report.bytes << " stage=" << report.stage << " reason=" << static_cast<std::uint32_t>(report.reason);
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::Permission && table.slab != nullptr) {
        auto range = pages != nullptr ? pages->Find(report.address) : std::nullopt;
        if (!range.has_value()) {
            const auto found = std::upper_bound(ownRanges.begin(), ownRanges.end(), report.address, [](std::uint64_t value, const ShaderRecompiler::BdaAbi::Range& candidate) { return value < candidate.begin; });
            if (found != ownRanges.begin() && report.address < std::prev(found)->end) range = *std::prev(found);
        }
        if (range.has_value()) message << std::hex << "; the store hit 0x" << range->begin << "+0x" << range->end - range->begin << ", read-only in the BDA table: stores through GPU-selected descriptors and FLAT/GLOBAL addresses reach only writable ranges imported in place, not ones served by a mirror or a copy (past the import budget, or refused by the driver)";
        message << std::dec << "; this dispatch's write grants: " << writeGrants.size() << std::hex;
        for (std::size_t index = 0; index < writeGrants.size() && index < 8u; ++index) message << " 0x" << writeGrants[index].begin << "-0x" << writeGrants[index].end;
    }
    throw std::runtime_error(message.str());
}

}

namespace AgcDriver::Graphics {

void BdaResources::SetImageTables(std::vector<ImageTableFaults> tables, std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) {
    imageTables = std::move(tables);
    imageTableRanges = std::move(ranges);
    imageTableBytes.clear();
    imageTableBytes.reserve(imageTableRanges.size());
    for (const auto& [base, size] : imageTableRanges) {
        const auto* bytes = reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(base));
        imageTableBytes.emplace_back(bytes, bytes + size);
    }
}

void BatchFaultRecord::Check(std::span<const BatchFaultUser> users) const {
    namespace Abi = ShaderRecompiler::BdaAbi;
    if (users.empty()) return;
    auto* words = reinterpret_cast<std::uint32_t*>(slice.data);
    std::vector<const BdaResources*> sharing;
    sharing.reserve(users.size());
    for (const auto& user : users) sharing.push_back(user.bda);
    const bool recorded = StorageTexture::StoresRecordedFor(id);
    const auto images = recorded ? WrittenImageRanges(words) : std::nullopt;
    BdaResources::applyWrittenPages(words, sharing);
    if (recorded) StorageTexture::ResolveRecordedStores(id, images ? &*images : nullptr);
    if (words[Abi::WrittenOverflowWord] != 0) {
        for (const auto& user : users) {
            for (const auto& [begin, end] : user.writes) {
                if (end > begin) BdaResources::markWrittenRange(begin, static_cast<std::size_t>(end - begin), sharing);
            }
        }
        words[Abi::WrittenOverflowWord] = 0;
    }
    Abi::Fault report{};
    std::memcpy(&report, slice.data, sizeof(report));
    if (!BdaResources::published(report) || loopGuardReported(report)) return;
    const auto holds = [&](const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
        return std::any_of(ranges.begin(), ranges.end(), [&](const std::pair<std::uint64_t, std::uint64_t>& range) { return range.first <= report.address && report.address < range.second; });
    };
    std::vector<std::size_t> candidates;
    for (std::size_t index = 0; index < users.size(); ++index) {
        const bool match = report.reason == Abi::FaultReason::ImageTable ? users[index].bda->owns(report) : holds(users[index].reads) || holds(users[index].writes);
        if (match) candidates.push_back(index);
    }
    const auto owner = candidates.empty() ? std::size_t{0} : candidates.front();
    try {
        users[owner].bda->throwFault(report);
    } catch (const std::runtime_error& error) {
        std::ostringstream message;
        message << error.what();
        if (candidates.size() == 1 || users.size() == 1) {
            if (users[owner].program != 0) message << " (program 0x" << std::hex << users[owner].program << std::dec << ")";
            if (users.size() > 1) message << " (one of " << users.size() << " builds sharing the batch's fault record)";
        } else {
            std::vector<std::size_t> named = candidates;
            if (named.empty()) {
                for (std::size_t index = 0; index < users.size(); ++index) named.push_back(index);
            }
            message << " (one of " << named.size() << " candidate builds of the " << users.size() << " sharing the batch's fault record; programs" << std::hex;
            for (std::size_t index = 0; index < named.size() && index < 4u; ++index) message << " 0x" << users[named[index]].program;
            message << std::dec << (named.size() > 4u ? " and more)" : ")");
        }
        throw std::runtime_error(message.str());
    }
}

void BdaResources::markWrittenPages(std::uint32_t* words, std::span<const BdaResources* const> users) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Require(words[Abi::WrittenOverflowWord] == 0, "more than " + std::to_string(Abi::WrittenPageSlots) + " chunks of " + std::to_string(Abi::WrittenChunkPages) + " pages stored to through the BDA table in one use are not implemented");
    applyWrittenPages(words, users);
}

void BdaResources::applyWrittenPages(std::uint32_t* words, std::span<const BdaResources* const> users) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    const auto count = words[Abi::WrittenCountWord];
    if (count == 0) return;
    Require(Abi::ForEachWrittenRange(words, [&](std::uint64_t begin, std::size_t bytes) { markWrittenRange(begin, bytes, users); }), "invalid BDA written-page record");
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[Abi::WrittenListWord + claim];
        words[Abi::WrittenSlotsWord + 2 * slot] = 0;
        words[Abi::WrittenSlotsWord + 2 * slot + 1] = 0;
    }
    words[Abi::WrittenCountWord] = 0;
}

void BdaResources::markWrittenRange(std::uint64_t begin, std::size_t bytes, std::span<const BdaResources* const> users) {
    const auto end = begin + bytes;
    if (auto* residency = ResidencyDriver::Active()) residency->NoteRecordedWrites(begin, bytes);
    for (const auto* user : users) {
        for (std::size_t index = 0; index < user->imageTableRanges.size(); ++index) {
            const auto [base, size] = user->imageTableRanges[index];
            const auto low = std::max(begin, base);
            const auto high = std::min(end, base + size);
            if (low >= high) continue;
            const auto* current = reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(low));
            if (std::memcmp(current, user->imageTableBytes[index].data() + (low - base), static_cast<std::size_t>(high - low)) != 0) {
                std::ostringstream message;
                message << "image table at 0x" << std::hex << base << " is written by its own dispatch";
                Require(false, message.str());
            }
        }
    }
    GuestMemory::MarkWritten(begin, bytes);
}

}
