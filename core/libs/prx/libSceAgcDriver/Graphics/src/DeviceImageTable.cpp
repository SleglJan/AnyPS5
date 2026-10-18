#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "DeviceImageTableAbi.hpp"
#include "ImageTableAbi.hpp"
#include "StorageImageMemory.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <span>
#include <utility>

namespace AgcDriver::Graphics {

namespace Abi = ShaderRecompiler::DeviceImageTableAbi;

namespace {

std::atomic<std::uint64_t>& MemoryEntryCount() {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

std::atomic<std::uint64_t>& MemoryDemotionCount() {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

constexpr VkBufferUsageFlags DeviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr std::size_t RingWords = Abi::RingRecordsWord + static_cast<std::size_t>(DeviceImageTable::RingCapacity) * Abi::RecordWords;
constexpr std::size_t UpdateBufferBytes = 65536u;

bool Profiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

std::string HexWords(std::span<const std::uint32_t> words) {
    std::string text;
    for (const auto word : words) {
        char hex[12];
        std::snprintf(hex, sizeof(hex), text.empty() ? "%08x" : " %08x", word);
        text += hex;
    }
    return text;
}

bool PointSampled(const ShaderRecompiler::ImageResource& mode) {
    return mode.numericClass == ShaderRecompiler::IrTextureNumericClass::Uint || mode.numericClass == ShaderRecompiler::IrTextureNumericClass::Sint || mode.conversionFormat != ShaderRecompiler::IrBufferFormat::Invalid || mode.depthBits;
}

bool Multisampled(const ShaderRecompiler::ImageResource& mode) {
    return mode.dimension == ShaderRecompiler::RdnaImageDimension::Dim2DMsaa || mode.dimension == ShaderRecompiler::RdnaImageDimension::Dim2DMsaaArray;
}

bool IntegerMode(const ShaderRecompiler::ImageResource& mode) {
    return mode.numericClass == ShaderRecompiler::IrTextureNumericClass::Uint || mode.numericClass == ShaderRecompiler::IrTextureNumericClass::Sint;
}

std::shared_ptr<Sampler> ModeSampler(const Context& build, std::span<const std::uint32_t, 4> words, bool compare, bool integerBorder, bool degammaPaired, bool unnormalized) {
    if (build.samplerCache != nullptr) return build.samplerCache->Get(build, words, compare, unnormalized, degammaPaired, integerBorder);
    auto resource = DecodeSamplerResource(words, unnormalized, degammaPaired, integerBorder);
    resource.compareEnable = compare;
    return std::make_shared<Sampler>(build, resource);
}

}

bool UsesDeviceImageTable(const ShaderRecompiler::RecompileResult& program) {
    return std::any_of(program.bindings.begin(), program.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) {
        return binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap;
    });
}

DeviceImageTable::DeviceImageTable(const Context& context) : context(context) {
    Require(context.bufferDeviceAddress, "the device image table needs buffer device addresses");
    header = std::make_shared<Buffer>(context, Abi::HeaderWords * sizeof(std::uint32_t), DeviceUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

DeviceImageTable::~DeviceImageTable() = default;

std::uint64_t DeviceImageTable::HeaderAddress() const {
    return header->DeviceAddress();
}

void DeviceImageTable::Expect(std::span<const Abi::Expectation> images, bool complete) {
    std::lock_guard lock(mutex);
    if (!complete) incomplete = true;
    const auto presents = Recorder::Presents();
    for (const auto& image : images) {
        current.push_back(image.key);
        const auto found = keys.Find(image.key);
        if (!found.has_value()) {
            (image.candidate ? candidates : expected).push_back({image.key, image.pc});
            continue;
        }
        const auto held = entries.find(image.key);
        if (held != entries.end()) held->second.used = presents;
        if (found->fault != 0u && !image.candidate && (held == entries.end() || !held->second.parked)) throw std::runtime_error(rejectionText(image.key, found->fault));
    }
}

void DeviceImageTable::BeforeDispatch(const Context& build, Recorder& recorder) {
    std::lock_guard lock(mutex);
    const bool unnamed = std::exchange(incomplete, false);
    if (unnamed && unnamedBatch.has_value()) {
        const auto started = std::chrono::steady_clock::now();
        const auto batch = *std::exchange(unnamedBatch, std::nullopt);
        recorder.Submit();
        recorder.FinishToken(batch);
        ++totals.missWaits;
        totals.missWaitNanoseconds += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    }
    const auto named = current;
    resolve(build, recorder);
    refresh(build, recorder, unnamed ? nullptr : &named);
    evict(build, recorder);
    record(recorder);
    if (unnamed) unnamedBatch = ringBatch;
    report();
}

void DeviceImageTable::resolve(const Context& build, Recorder& recorder) {
    std::vector<DeviceImageKeys::Key> resolved;
    const auto add = [&](std::vector<DeviceImageMiss>& queued, std::uint64_t& counter, bool named, bool required) {
        if (queued.empty()) return;
        const auto misses = DistinctDeviceImageMisses(queued);
        queued.clear();
        for (const auto& miss : misses) {
            if (keys.Find(miss.key).has_value()) continue;
            ++counter;
            Entry entry;
            const auto value = lookup(build, miss.key, miss.pc, entry, named, required || !named);
            insert(miss.key, value, recorder);
            if (required && value.fault != 0u) throw std::runtime_error(rejectionText(miss.key, value.fault));
            if (entry.texture == nullptr && entry.storage == nullptr && !entry.memory) continue;
            if (entry.perUse) {
                perUseKeys.push_back(miss.key);
                resolved.push_back(miss.key);
            }
            entry.used = Recorder::Presents();
            entry.checked = currentState();
            entries[miss.key] = std::move(entry);
            ++totals.inserted;
        }
    };
    add(pending, totals.distinct, false, false);
    add(expected, totals.expected, true, true);
    add(candidates, totals.expected, true, false);
    refreshPerUse(build, recorder, resolved);
    demoteViewedMemory(build);
    for (const auto& key : current) {
        const auto held = entries.find(key);
        if (held == entries.end() || !held->second.memory || !held->second.written) continue;
        dispatchMemoryWrites.emplace_back(held->second.address, held->second.address + held->second.guestBytes);
    }
    for (const auto& key : current) {
        const auto held = entries.find(key);
        if (held != entries.end() && held->second.texture != nullptr && std::ranges::find(dispatchTextures, held->second.texture) == dispatchTextures.end()) dispatchTextures.push_back(held->second.texture);
        if (held == entries.end() || held->second.storage == nullptr) continue;
        if (std::ranges::none_of(dispatchStorage, [&](const DispatchStorageUse& used) { return used.key == key; })) dispatchStorage.push_back({key, held->second.storage, held->second.written});
    }
    current.clear();
}

void DeviceImageTable::AfterDispatch(const Context& build, Recorder& recorder, std::uint64_t storeRecord) {
    std::lock_guard lock(mutex);
    const auto uses = std::exchange(dispatchStorage, {});
    dispatchTextures.clear();
    for (const auto& [begin, end] : std::exchange(dispatchMemoryWrites, {})) {
        GuestMemory::MarkWritten(begin, static_cast<std::size_t>(end - begin));
        for (auto& [key, entry] : entries) {
            if (entry.texture != nullptr && entry.address < end && begin < entry.address + entry.guestBytes) entry.overwritten = true;
        }
    }
    std::exception_ptr failure;
    for (const auto& use : uses) {
        recorder.Keep(use.image);
        if (const auto held = entries.find(use.key); held != entries.end() && held->second.storage == use.image) held->second.storage.reset();
        if (!use.written) continue;
        use.image->OwnAsTarget();
        bool depthPlane = false;
        try {
            depthPlane = DepthSurfaceStoreBack(build, *use.image);
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
        if (storeRecord != 0 && !depthPlane) use.image->MarkStoresRecorded(storeRecord);
        else use.image->MarkDirty();
    }
    if (failure) std::rethrow_exception(failure);
}

void DeviceImageTable::evict(const Context& build, Recorder& recorder) {
    constexpr std::uint64_t CheckPresents = 60u;
    constexpr std::uint64_t IdlePresents = 120u;
    const auto presents = Recorder::Presents();
    if (presents < evictionCheck + CheckPresents) return;
    evictionCheck = presents;
    std::uint64_t pinned = 0;
    for (const auto& [key, entry] : entries) {
        if (entry.texture != nullptr) pinned += entry.guestBytes;
    }
    const auto budget = SampledTextureCacheBudget(build) / 2u;
    if (pinned <= budget) return;
    std::vector<std::pair<std::uint64_t, DeviceImageKeys::Key>> idle;
    for (const auto& [key, entry] : entries) {
        if (entry.texture != nullptr && !entry.perUse && entry.used + IdlePresents <= presents) idle.emplace_back(entry.used, key);
    }
    std::ranges::sort(idle, [](const auto& left, const auto& right) { return left.first < right.first; });
    std::set<DeviceImageKeys::Key> dropped;
    for (const auto& [used, key] : idle) {
        if (pinned <= budget / 2u * 3u / 2u) break;
        auto held = entries.find(key);
        pinned -= std::min(pinned, held->second.guestBytes);
        retire(held->second, recorder);
        entries.erase(held);
        dropped.insert(key);
    }
    if (dropped.empty()) return;
    totals.evicted += keys.Retain([&](const DeviceImageKeys::Key& key) { return !dropped.contains(key); });
}

std::vector<std::shared_ptr<Texture>> DeviceImageTable::DispatchTextures() const {
    std::lock_guard lock(mutex);
    return dispatchTextures;
}

std::vector<std::shared_ptr<StorageTexture>> DeviceImageTable::DispatchStorage() const {
    std::lock_guard lock(mutex);
    std::vector<std::shared_ptr<StorageTexture>> images;
    images.reserve(dispatchStorage.size());
    for (const auto& use : dispatchStorage) images.push_back(use.image);
    return images;
}

void DeviceImageTable::refreshPerUse(const Context& build, Recorder& recorder, const std::vector<DeviceImageKeys::Key>& resolved) {
    for (auto it = perUseKeys.begin(); it != perUseKeys.end();) {
        const auto& key = *it;
        auto held = entries.find(key);
        if (held == entries.end()) {
            it = perUseKeys.erase(it);
            continue;
        }
        auto& entry = held->second;
        if (std::ranges::find(resolved, key) != resolved.end()) {
            ++it;
            continue;
        }
        if (std::ranges::find(current, key) == current.end()) {
            if (entry.memory && !entry.written) {
                Entry view;
                const auto value = lookup(build, key, 0u, view, false, false);
                Require(keys.Insert(key, value) == DeviceImageKeys::Insertion::Updated, "a memory-path device image entry was not in the table");
                ++totals.replaced;
                if (view.texture == nullptr) {
                    entries.erase(held);
                    it = perUseKeys.erase(it);
                    continue;
                }
                view.used = entry.used;
                view.checked = currentState();
                const bool perUse = view.perUse;
                entry = std::move(view);
                if (!perUse) {
                    it = perUseKeys.erase(it);
                    continue;
                }
                ++it;
                continue;
            }
            if (!entry.parked) {
                const auto fault = reject(key, 0u, static_cast<std::uint32_t>(ShaderRecompiler::ImageTableAbi::RejectionReason::DriverRejected), "a depth surface or a copy remade per use is read through the device image table by a dispatch whose CPU pre-pass did not name it, which is not implemented", false);
                Require(keys.Insert(key, DeviceImageValue{0u, 0u, fault + 1u, 0u}) == DeviceImageKeys::Insertion::Updated, "a parked device image entry was not in the table");
                entry.parked = true;
            }
            ++it;
            continue;
        }
        ++totals.refreshed;
        Entry fresh;
        const auto value = lookup(build, key, 0u, fresh, true, true);
        Require(keys.Insert(key, value) == DeviceImageKeys::Insertion::Updated, "a per-use device image entry was not in the table");
        if (value.fault != 0u) throw std::runtime_error(rejectionText(key, value.fault));
        if (fresh.texture != entry.texture || fresh.sampler != entry.sampler || fresh.storage != entry.storage) {
            retire(entry, recorder);
            ++totals.replaced;
        }
        const bool perUse = fresh.perUse;
        entry = std::move(fresh);
        if (!perUse) {
            it = perUseKeys.erase(it);
            continue;
        }
        ++it;
    }
}

void DeviceImageTable::demoteViewedMemory(const Context& build) {
    for (bool demoted = true; demoted;) {
        demoted = false;
        for (const auto& key : current) {
            const auto held = entries.find(key);
            if (held == entries.end() || !held->second.memory) continue;
            const auto& memory = held->second;
            const bool viewed = std::ranges::any_of(current, [&](const DeviceImageKeys::Key& other) {
                const auto view = entries.find(other);
                if (view == entries.end() || view->second.memory || (view->second.texture == nullptr && view->second.storage == nullptr)) return false;
                if (!memory.written && !view->second.written) return false;
                return view->second.address < memory.address + memory.guestBytes && memory.address < view->second.address + view->second.guestBytes;
            });
            if (!viewed) continue;
            viewOnly.insert(key);
            Entry fresh;
            const auto value = lookup(build, key, 0u, fresh, true, true);
            Require(keys.Insert(key, value) == DeviceImageKeys::Insertion::Updated, "a device image entry demoted to its view was not in the table");
            if (value.fault != 0u) throw std::runtime_error(rejectionText(key, value.fault));
            fresh.used = held->second.used;
            fresh.checked = currentState();
            held->second = std::move(fresh);
            if (!held->second.perUse) std::erase(perUseKeys, key);
            ++totals.demoted;
            MemoryDemotionCount().fetch_add(1, std::memory_order_relaxed);
            demoted = true;
            break;
        }
    }
}

namespace {

constexpr std::uint32_t GuestFormatRgba8UInt = 60u;
constexpr std::uint32_t GuestFormatRgba8Srgb = 130u;

std::string forcedDegammaWords(const ShaderRecompiler::DeviceImageInterface& interface, const DeviceImageKeys::Key& key, ShaderRecompiler::DescriptorValue& descriptor) {
    auto& words = descriptor.dwords;
    if ((key[Abi::SamplerClassWord] & Abi::RuntimeSampler) == 0u || (key[Abi::SamplerWord] & (1u << 20u)) == 0u) return {};
    if (((words[1] >> 20u) & 0x1ffu) != GuestFormatRgba8UInt) return {};
    const auto& image = interface.image;
    if (!image.singleMipPlaneBiasCompatible || image.dimension != ShaderRecompiler::RdnaImageDimension::Dim2D || image.resourceClass != ShaderRecompiler::ImageResourceClass::Sampled || image.depthCompare || image.packed) return "FORCE_DEGAMMA on RGBA8_UINT requires a direct plain biased pixel sample";
    if ((words[3] >> 28u) != 10u || (words[4] & 0x1fffu) != 0u || (words[3] & 0xff000u) != 0u || (words[5] & 0xf0u) != 0u) return "FORCE_DEGAMMA on RGBA8_UINT requires a one-slice volume at mip zero";
    words[1] = (words[1] & ~(0x1ffu << 20u)) | (GuestFormatRgba8Srgb << 20u);
    return {};
}

}

DeviceImageValue DeviceImageTable::lookup(const Context& build, const DeviceImageKeys::Key& key, std::uint32_t pc, Entry& entry, bool perUse, bool report) {
    using Reason = ShaderRecompiler::ImageTableAbi::RejectionReason;
    const auto id = static_cast<std::uint64_t>(key[Abi::InterfaceWord]) | (static_cast<std::uint64_t>(key[Abi::InterfaceWord + 1u]) << 32u);
    const auto interface = ShaderRecompiler::ResourceMaterializer::FindDeviceImageInterface(id);
    Require(interface != nullptr, "a device image table miss names an image interface no compiled program registered (" + HexWords(std::span<const std::uint32_t>(key).subspan(Abi::InterfaceWord, 2u)) + ")");
    ShaderRecompiler::DescriptorValue descriptor;
    descriptor.dwordCount = Abi::DescriptorWords;
    std::copy_n(key.begin(), Abi::DescriptorWords, descriptor.dwords.begin());
    const auto rejected = [&](Reason reason, const std::string& message) {
        const bool runtimeSampler = (key[Abi::SamplerClassWord] & Abi::RuntimeSampler) != 0u;
        const auto text = runtimeSampler ? message + " (S# " + HexWords(std::span<const std::uint32_t>(key).subspan(Abi::SamplerWord, Abi::SamplerWords)) + ")" : message;
        return DeviceImageValue{0u, 0u, reject(key, pc, static_cast<std::uint32_t>(reason), text, report) + 1u};
    };
    if (((key[1] >> 20u) & 0x1ffu) == GuestFormatRgba8UInt && (key[Abi::SamplerClassWord] & Abi::RuntimeSampler) != 0u) {
        const auto mixed = std::ranges::any_of(current, [&](const DeviceImageKeys::Key& other) {
            return (other[Abi::SamplerClassWord] & Abi::RuntimeSampler) != 0u && ((other[Abi::SamplerWord] ^ key[Abi::SamplerWord]) & (1u << 20u)) != 0u && other[Abi::InterfaceWord] == key[Abi::InterfaceWord] && other[Abi::InterfaceWord + 1u] == key[Abi::InterfaceWord + 1u] && std::equal(key.begin(), key.begin() + Abi::DescriptorWords, other.begin());
        });
        if (mixed) return rejected(Reason::DriverRejected, "FORCE_DEGAMMA on RGBA8_UINT with mixed sampler states is not implemented");
    }
    if (const auto forced = forcedDegammaWords(*interface, key, descriptor); !forced.empty()) return rejected(Reason::DriverRejected, forced);
    const auto match = ShaderRecompiler::ResourceMaterializer::TryRuntimeImageMode(interface->image, descriptor, interface->modes);
    if (!match.mode.has_value()) return rejected(match.reason, match.detail);
    const auto& mode = interface->modes.at(*match.mode);
    if (interface->image.directRuntime && interface->image.read && !interface->image.written && !interface->image.atomic && ((descriptor.dwords[3] >> 28u) & 0xfu) <= 1u) return {0u, *match.mode, 0u, 0u};
    if (mode.constantSwizzle) return {0u, *match.mode, 0u, 0u};
    if (interface->image.singleMipPlaneBiasCompatible && interface->image.dimension == ShaderRecompiler::RdnaImageDimension::Dim2D && mode.dimension == ShaderRecompiler::RdnaImageDimension::Dim3D && (key[Abi::SamplerClassWord] & Abi::RuntimeSampler) != 0u) {
        const auto first = key[Abi::SamplerWord];
        const auto third = key[Abi::SamplerWord + 2u];
        if ((first & Abi::SamplerUnnormalized) != 0u || ((first >> 29u) & 3u) > 2u || ((third >> 20u) & 0xfu) != 0u || ((third >> 26u) & 3u) > 1u) return rejected(Reason::DriverRejected, "2D biased sampling of a one-slice volume requires a normalized point sampler");
    }
    if (perUse && !viewOnly.contains(key) && ShaderRecompiler::MemoryPathRuntimeImage(interface->image) && ShaderRecompiler::StorageImageMemory::SurfaceThroughMemory(descriptor.dwords, ShaderRecompiler::StorageImageMemory::AccessOf(interface->image)) && !DepthSurfaceAt(DecodeTextureResource(descriptor.dwords).baseAddress)) {
        const auto resource = DecodeTextureResource(descriptor.dwords);
        entry = {};
        entry.address = resource.baseAddress;
        entry.guestBytes = DescribeSurface(resource).guestBytes;
        entry.perUse = true;
        entry.memory = true;
        entry.written = interface->image.written || interface->image.atomic;
        ++totals.memory;
        MemoryEntryCount().fetch_add(1, std::memory_order_relaxed);
        return {ShaderRecompiler::StorageImageMemory::MemorySlot, *match.mode, 0u, 0u};
    }
    if (interface->image.resourceClass == ShaderRecompiler::ImageResourceClass::Storage) {
        if (!perUse) return rejected(Reason::DriverRejected, "a storage image is read or written through the device image table by a dispatch whose CPU pre-pass did not name it, which is not implemented");
        try {
            DeviceTableView view;
            view.shape = ShaderRecompiler::ImageShapeForResource(mode);
            view.atomic = mode.atomic;
            view.atomic64 = mode.atomic64;
            view.mipRun = mode.mipMode == ShaderRecompiler::ImageMipMode::DynamicStorage;
            view.written = interface->image.written || interface->image.atomic;
            auto storage = LookupDeviceTableStorage(build, std::span<const std::uint32_t, 8>(descriptor.dwords), view);
            entry = {};
            entry.address = storage.address;
            entry.guestBytes = storage.guestBytes;
            entry.perUse = true;
            entry.storage = std::move(storage.image);
            entry.written = interface->image.written || interface->image.atomic;
            return {storage.slot, *match.mode, 0u, 0u};
        } catch (const std::runtime_error& error) {
            return rejected(Reason::DriverRejected, error.what());
        }
    }
    if (Multisampled(mode) && mode.dimension == ShaderRecompiler::RdnaImageDimension::Dim2DMsaaArray) return rejected(Reason::Multisampled, "an image table entry is a multisampled array");
    const auto reduction = "a min or max reduction sampler that filters between texels or mip levels samples an image that needs point filtering (sint, converted or depth-bits format), which is not implemented";
    const bool runtimeSampler = (key[Abi::SamplerClassWord] & Abi::RuntimeSampler) != 0u;
    std::array<std::uint32_t, Abi::SamplerWords> samplerWords{};
    std::copy_n(key.begin() + Abi::SamplerWord, Abi::SamplerWords, samplerWords.begin());
    const bool reducing = runtimeSampler ? ShaderRecompiler::ReducesBetweenTexels(samplerWords[0], samplerWords[2]) : (key[Abi::SamplerClassWord] & Abi::SamplerReduction) != 0u;
    const bool unnormalized = runtimeSampler && (samplerWords[0] & Abi::SamplerUnnormalized) != 0u;
    if (PointSampled(mode) && reducing) return rejected(Reason::Reduction, reduction);
    try {
        auto samplerClass = key[Abi::SamplerClassWord];
        std::shared_ptr<Sampler> sampler;
        bool forcedSrgb = false;
        if (runtimeSampler) {
            const auto descriptorWords = std::span<const std::uint32_t>(descriptor.dwords);
            const bool singleLevel = DescriptorSingleLevel(descriptorWords);
            if (const auto paired = SingleLevelSamplerWords(samplerWords, singleLevel, !singleLevel)) samplerWords = *paired;
            if (PointSampled(mode)) samplerWords[2] = ShaderRecompiler::PointFilteredSamplerWord(samplerWords[0], samplerWords[2]);
            if (unnormalized && mode.depthCompare) return rejected(Reason::Unnormalized, "an S# with unnormalized coordinates compares against the image it samples, which is not implemented");
            if (const auto* mismatch = unnormalized ? ShaderRecompiler::ResourceMaterializer::UnnormalizedSampleMismatch(mode) : nullptr) return rejected(Reason::Unnormalized, std::string("an S# with unnormalized coordinates ") + mismatch + ", which is not implemented");
            if (!unnormalized && interface->image.foldTexelOffsetsOnly) return rejected(Reason::UnsupportedOperation, "a normalized S# is used with an image sample that has a texel offset that is not a constant, which image sampling takes only with VK_KHR_maintenance8 and shaderImageGatherExtended");
            sampler = ModeSampler(build, samplerWords, mode.depthCompare, IntegerMode(mode), true, unnormalized);
            forcedSrgb = ForcesDegammaOn(DecodeTextureResource(descriptorWords).format, std::array<std::uint32_t, 1>{0u}, std::array<std::shared_ptr<Sampler>, 1>{sampler});
            samplerClass = DeviceTableSamplerClass(sampler.get(), samplerWords, unnormalized, IntegerMode(mode)) & ~Abi::SamplerAnisoOverride;
        }
        DeviceTableView view{ShaderRecompiler::ImageShapeForResource(mode), mode.depthCompare, mode.conversionFormat != ShaderRecompiler::IrBufferFormat::Invalid || mode.depthBits, ShaderRecompiler::ResourceMaterializer::BindsDecodedCopy(mode, descriptor), IntegerMode(mode), unnormalized, (key[Abi::SamplerClassWord] & Abi::SamplerExactAtBase) != 0u && (samplerWords[1] & 0xfffu) == 0u && (samplerWords[2] & 0xfffffu) == 0u};
        view.srgb = forcedSrgb;
        auto texture = LookupDeviceTableTexture(build, std::span<const std::uint32_t, 8>(descriptor.dwords), view, samplerClass, perUse);
        const auto samplerSlot = sampler != nullptr ? sampler->Slot() : 0u;
        auto modeWord = *match.mode;
        if ((mode.emulatedCompare & ShaderRecompiler::EmulatedCompare::Enabled) != 0u) {
            if (!runtimeSampler) return rejected(Reason::ColorCompare, "comparison sampling of a color texture through a sampler the device image table does not read is not implemented");
            std::uint32_t compare = 0;
            try {
                compare = ShaderRecompiler::ResourceMaterializer::ColorCompareState(interface->image, descriptor, std::span<const std::uint32_t, 4>(key.data() + Abi::SamplerWord, Abi::SamplerWords));
            } catch (const std::runtime_error& error) {
                return rejected(Reason::ColorCompare, error.what());
            }
            modeWord |= compare << Abi::ModeCompareShift;
        }
        entry = {std::move(texture.texture), std::move(sampler), texture.address, texture.guestBytes, texture.generation, texture.snapshot, texture.perUse};
        return {texture.slot, modeWord, 0u, samplerSlot};
    } catch (const std::runtime_error& error) {
        return rejected(Reason::DriverRejected, error.what());
    }
}

std::uint32_t DeviceImageTable::reject(const DeviceImageKeys::Key& key, std::uint32_t pc, std::uint32_t reason, const std::string& message, bool report) {
    const auto [found, inserted] = rejectionIndex.try_emplace({key, reason, message}, static_cast<std::uint32_t>(rejections.size()));
    if (!inserted) return found->second;
    Require(rejections.size() < Abi::FaultFlag, "the device image table holds more rejections than a fault record can name");
    Rejection rejection;
    std::copy_n(key.begin(), Abi::DescriptorWords, rejection.words.begin());
    rejection.reason = reason;
    rejection.pc = pc;
    rejection.message = message;
    rejections.push_back(std::move(rejection));
    if (!report) return found->second;
    ++totals.rejected;
    const auto name = ShaderRecompiler::ImageTableAbi::RejectionReasonName(static_cast<ShaderRecompiler::ImageTableAbi::RejectionReason>(reason));
    std::fprintf(stderr, "[device-images] the descriptor (words %s) first sampled at pc 0x%x is rejected: %.*s: %s\n", HexWords(std::span<const std::uint32_t>(key).first(Abi::DescriptorWords)).c_str(), pc, static_cast<int>(name.size()), name.data(), message.c_str());
    return found->second;
}

void DeviceImageTable::insert(const DeviceImageKeys::Key& key, const DeviceImageValue& value, Recorder& recorder) {
    if (keys.Insert(key, value) != DeviceImageKeys::Insertion::Full) return;
    ++totals.clears;
    std::fprintf(stderr, "[device-images] the table is full at %u entries in %u buckets: every entry is dropped and resolved again on its next use\n", keys.Entries(), keys.Buckets());
    for (auto& [held, entry] : entries) retire(entry, recorder);
    entries.clear();
    perUseKeys.clear();
    keys.Clear();
    Require(keys.Insert(key, value) != DeviceImageKeys::Insertion::Full, "the device image table refused a key while empty");
}

void DeviceImageTable::retire(const Entry& entry, Recorder& recorder) {
    if (entry.texture != nullptr) recorder.Keep(entry.texture);
    if (entry.sampler != nullptr) recorder.Keep(entry.sampler);
    if (entry.storage != nullptr) recorder.Keep(entry.storage);
}

DeviceImageTable::ValidationState DeviceImageTable::currentState() {
    return {GuestMemory::ThreadCollectEpoch(), GuestMemory::UnwatchSerial(), DepthSurfacesSerial(), StorageTexture::PendingSerial(), GuestMemory::OwnWriteSerial()};
}

void DeviceImageTable::refresh(const Context& build, Recorder& recorder, const std::vector<DeviceImageKeys::Key>* scope) {
    if (entries.empty()) return;
    const auto state = currentState();
    const auto due = [&](const Entry& entry) { return !entry.perUse && (entry.overwritten || state.epoch == 0u || entry.checked != state); };
    std::vector<std::map<DeviceImageKeys::Key, Entry>::iterator> checked;
    if (scope != nullptr) {
        for (const auto& key : *scope) {
            const auto it = entries.find(key);
            if (it != entries.end() && due(it->second) && std::ranges::find(checked, it) == checked.end()) checked.push_back(it);
        }
    } else {
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (due(it->second)) checked.push_back(it);
        }
    }
    if (checked.empty()) return;
    const auto everything = [&](const Entry& entry) { return entry.checked.unwatch != state.unwatch || entry.checked.depthSurfaces != state.depthSurfaces; };
    const auto collect = [&](const Entry& entry) { return everything(entry) || state.epoch == 0u || entry.checked.epoch != state.epoch; };
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    for (const auto it : checked) {
        if (collect(it->second)) spans.emplace_back(it->second.address, it->second.address + it->second.guestBytes);
    }
    std::ranges::sort(spans);
    std::size_t merged = 0;
    for (const auto& span : spans) {
        if (merged != 0u && span.first <= spans[merged - 1u].second) spans[merged - 1u].second = std::max(spans[merged - 1u].second, span.second);
        else spans[merged++] = span;
    }
    spans.resize(merged);
    for (const auto& [begin, end] : spans) static_cast<void>(GuestMemory::CollectWrites(begin, static_cast<std::size_t>(end - begin)));
    for (const auto it : checked) {
        auto& [key, entry] = *it;
        const bool all = everything(entry);
        const bool written = collect(entry) || entry.checked.ownWrites != state.ownWrites;
        const bool pendingChanged = all || entry.checked.pending != state.pending;
        bool stale = all || entry.overwritten || (written && !GuestMemory::UnchangedSince(entry.address, static_cast<std::size_t>(entry.guestBytes), entry.generation));
        if (!stale && pendingChanged) {
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{entry.address, entry.address + entry.guestBytes}}};
            stale = !entry.snapshot || StorageTexture::AnyPendingOverlaps(range);
        }
        if (!stale) {
            entry.checked = state;
            continue;
        }
        ++totals.refreshed;
        Entry fresh;
        const auto value = lookup(build, key, 0u, fresh, false, true);
        Require(keys.Insert(key, value) == DeviceImageKeys::Insertion::Updated, "a refreshed device image entry was not in the table");
        if (fresh.texture != entry.texture || fresh.sampler != entry.sampler) {
            retire(entry, recorder);
            ++totals.replaced;
        }
        if (fresh.texture == nullptr) {
            entries.erase(it);
            continue;
        }
        fresh.checked = state;
        entry = std::move(fresh);
    }
}

void DeviceImageTable::record(Recorder& recorder) {
    const auto batch = recorder.CallerToken();
    const bool assign = ring == nullptr || ringBatch != batch;
    const bool grown = table == nullptr || keys.Buckets() != tableBuckets;
    const bool rebuilt = grown || keys.Rebuilt();
    const auto changed = keys.TakeChanged();
    if (!assign && !rebuilt && changed.empty()) return;
    recorder.DeclareSection({{this, true}});
    const auto commands = recorder.Commands();
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    const auto words = keys.Words();
    const auto bytes = words.size() * sizeof(std::uint32_t);
    if (grown) {
        if (table != nullptr) recorder.Keep(table);
        table = std::make_shared<Buffer>(context, bytes, DeviceUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        tableBuckets = keys.Buckets();
    }
    if (rebuilt) {
        auto staging = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(staging->Bytes().data(), words.data(), bytes);
        const VkBufferCopy region{0, 0, bytes};
        context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, staging->Handle(), table->Handle(), 1, &region);
        recorder.Keep(std::move(staging));
    } else {
        for (std::size_t first = 0; first < changed.size();) {
            auto last = first;
            while (last + 1u < changed.size() && changed[last + 1u] == changed[last] + 1u && (last + 2u - first) * Abi::BucketBytes <= UpdateBufferBytes) ++last;
            const auto offset = static_cast<VkDeviceSize>(changed[first]) * Abi::BucketBytes;
            context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, table->Handle(), offset, static_cast<VkDeviceSize>(last + 1u - first) * Abi::BucketBytes, words.data() + static_cast<std::size_t>(changed[first]) * Abi::BucketWords);
            first = last + 1u;
        }
    }
    if (assign || grown) {
        if (assign) {
            if (freeRings.empty()) freeRings.push_back(std::make_shared<Buffer>(context, RingWords * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
            ring = std::move(freeRings.back());
            freeRings.pop_back();
            std::memset(ring->Bytes().data(), 0, Abi::RingRecordsWord * sizeof(std::uint32_t));
            ringBatch = batch;
            recorder.OnComplete([this, held = ring] { drain(held); });
        }
        std::array<std::uint32_t, Abi::HeaderWords> fields{};
        const auto tableAddress = table->DeviceAddress();
        const auto ringAddress = ring->DeviceAddress();
        fields[Abi::HeaderVersion] = Abi::Version;
        fields[Abi::HeaderTableLow] = static_cast<std::uint32_t>(tableAddress);
        fields[Abi::HeaderTableHigh] = static_cast<std::uint32_t>(tableAddress >> 32u);
        fields[Abi::HeaderMask] = keys.Buckets() - 1u;
        fields[Abi::HeaderRingLow] = static_cast<std::uint32_t>(ringAddress);
        fields[Abi::HeaderRingHigh] = static_cast<std::uint32_t>(ringAddress >> 32u);
        fields[Abi::HeaderRingCapacity] = RingCapacity;
        context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, header->Handle(), 0, sizeof(fields), fields.data());
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

void DeviceImageTable::drain(const std::shared_ptr<Buffer>& drained) {
    std::lock_guard lock(mutex);
    const auto bytes = drained->Bytes();
    const auto decoded = DecodeDeviceImageRing(std::span<const std::uint32_t>(reinterpret_cast<const std::uint32_t*>(bytes.data()), RingWords), RingCapacity);
    totals.misses += decoded.written;
    totals.dropped += decoded.dropped;
    pending.insert(pending.end(), decoded.misses.begin(), decoded.misses.end());
    freeRings.push_back(drained);
}

std::string DeviceImageTable::rejectionText(const DeviceImageKeys::Key& key, std::uint32_t fault) const {
    const auto& rejection = rejections.at(fault - 1u);
    const auto name = ShaderRecompiler::ImageTableAbi::RejectionReasonName(static_cast<ShaderRecompiler::ImageTableAbi::RejectionReason>(rejection.reason));
    return "the descriptor (words " + HexWords(std::span<const std::uint32_t>(key).first(Abi::DescriptorWords)) + ") sampled at pc 0x" + HexWords(std::span<const std::uint32_t>(&rejection.pc, 1u)) + " is rejected: " + std::string(name) + ": " + rejection.message;
}

std::optional<DeviceImageTable::Rejection> DeviceImageTable::FindRejection(std::uint32_t index) const {
    std::lock_guard lock(mutex);
    if (index >= rejections.size()) return std::nullopt;
    return rejections[index];
}

std::uint64_t DeviceImageTable::MemoryEntries() {
    return MemoryEntryCount().load(std::memory_order_relaxed);
}

std::uint64_t DeviceImageTable::MemoryDemotions() {
    return MemoryDemotionCount().load(std::memory_order_relaxed);
}

DeviceImageTable::Counts DeviceImageTable::Totals() const {
    std::lock_guard lock(mutex);
    auto counts = totals;
    counts.entries = keys.Entries();
    counts.buckets = keys.Buckets();
    return counts;
}

void DeviceImageTable::report() {
    if (!Profiled()) return;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (lastReport == 0) {
        lastReport = now;
        reported = totals;
        reportedPresents = Recorder::Presents();
        const auto waits = GuestMemory::PendingReadWaitCounts();
        reportedWaitReads = waits.reads;
        reportedWaitNanoseconds = waits.nanoseconds;
        return;
    }
    if (now - lastReport < 10'000'000'000ll) return;
    const auto presents = Recorder::Presents() - reportedPresents;
    const auto misses = totals.misses - reported.misses;
    const auto waits = GuestMemory::PendingReadWaitCounts();
    std::fprintf(stderr, "[device-images] (10 s): %llu misses (%llu distinct, %llu dropped), %.2f per present over %llu presents, %llu expected, %llu inserted, %llu rejected, %llu refreshed (%llu replaced), %u entries in %u buckets, %llu clears, %llu pre-pass reads waited for pending GPU writes (%.1f ms), %llu waits for the misses of a dispatch with unnamed uses (%.1f ms)\n",
        static_cast<unsigned long long>(misses), static_cast<unsigned long long>(totals.distinct - reported.distinct), static_cast<unsigned long long>(totals.dropped - reported.dropped),
        presents != 0u ? static_cast<double>(misses) / static_cast<double>(presents) : 0.0, static_cast<unsigned long long>(presents), static_cast<unsigned long long>(totals.expected - reported.expected),
        static_cast<unsigned long long>(totals.inserted - reported.inserted), static_cast<unsigned long long>(totals.rejected - reported.rejected), static_cast<unsigned long long>(totals.refreshed - reported.refreshed), static_cast<unsigned long long>(totals.replaced - reported.replaced),
        keys.Entries(), keys.Buckets(), static_cast<unsigned long long>(totals.clears - reported.clears), static_cast<unsigned long long>(waits.reads - reportedWaitReads), static_cast<double>(waits.nanoseconds - reportedWaitNanoseconds) / 1e6,
        static_cast<unsigned long long>(totals.missWaits - reported.missWaits), static_cast<double>(totals.missWaitNanoseconds - reported.missWaitNanoseconds) / 1e6);
    reportedWaitReads = waits.reads;
    reportedWaitNanoseconds = waits.nanoseconds;
    reported = totals;
    reportedPresents = Recorder::Presents();
    lastReport = now;
}

}
