#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker/SrtFlatSlotClasses.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "ImageTableAbi.hpp"
#include "ResultKey.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

using ImageTableAbi::PoisonReason;

std::atomic<std::uint64_t> specializationNanoseconds{0};

bool MaterializeProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct DecodedImage {
    IrTextureNumericClass numericClass = IrTextureNumericClass::Unsupported;
    RdnaImageDimension dimension = RdnaImageDimension::Unknown;
    std::uint32_t mipCount = 1;
    IrBufferFormat conversionFormat = IrBufferFormat::Invalid;
    std::uint32_t shaderSwizzle = ShaderImageIdentitySwizzle;
    bool cube = false;
    bool fmask = false;
    bool depthBits = false;
    bool depthUnorm16 = false;
    IrBufferFormat packedFormat = IrBufferFormat::Invalid;
    bool srgbDecode = false;
};

ShaderBufferResource decodeBufferDescriptor(const DescriptorValue& value) {
    if (value.dwordCount != 4u) {
        throw std::runtime_error("buffer descriptor has an invalid width");
    }
    ShaderBufferResource result;
    for (std::uint32_t i = 0; i < 4u; i++) {
        result.fields[i] = value.dwords[i];
    }
    return result;
}

bool nullImageDescriptor(const DescriptorValue& descriptor) {
    return descriptor.dwords[0] == 0u && (descriptor.dwords[1] & 0xffu) == 0u;
}

ImageType rawImageType(const DescriptorValue& descriptor) {
    return static_cast<ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
}

IrBufferFormat rawImageFormat(const DescriptorValue& descriptor) {
    return static_cast<IrBufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
}

std::uint32_t descriptorImageSwizzle(const DescriptorValue& descriptor) {
    return descriptor.dwords[3] & 0xfffu;
}

bool descriptorIsCube(const DescriptorValue& descriptor) {
    return rawImageType(descriptor) == ImageType::Cube;
}

RdnaImageDimension descriptorDimension(const DescriptorValue& descriptor, RdnaImageDimension requested) {
    const bool wantArray = requested == RdnaImageDimension::Dim1DArray || requested == RdnaImageDimension::Dim2DArray || requested == RdnaImageDimension::Dim2DMsaaArray;
    switch (rawImageType(descriptor)) {
        case ImageType::Color1D:
            return RdnaImageDimension::Dim1D;
        case ImageType::Color1DArray:
            return wantArray ? RdnaImageDimension::Dim1DArray : RdnaImageDimension::Dim1D;
        case ImageType::Color3D:
            return RdnaImageDimension::Dim3D;
        case ImageType::Cube:
            return RdnaImageDimension::Dim2DArray;
        case ImageType::Color2DArray:
            return wantArray || requested == RdnaImageDimension::Dim3D ? RdnaImageDimension::Dim2DArray : RdnaImageDimension::Dim2D;
        case ImageType::Color2DMsaaArray:
            return wantArray ? RdnaImageDimension::Dim2DMsaaArray : RdnaImageDimension::Dim2DMsaa;
        case ImageType::Color2D:
            return RdnaImageDimension::Dim2D;
        case ImageType::Color2DMsaa:
            return RdnaImageDimension::Dim2DMsaa;
        default:
            throw std::runtime_error("image descriptor has an unsupported type");
    }
}

bool validImageDescriptor(const DescriptorValue& descriptor, bool r128) {
    const auto type = rawImageType(descriptor);
    const auto format = rawImageFormat(descriptor);
    if (type < ImageType::Color1D || format == IrBufferFormat::Invalid) {
        return false;
    }
    if (r128 && type != ImageType::Color1D && type != ImageType::Color2D && type != ImageType::Color2DMsaa) {
        return false;
    }
    if (type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) {
        const auto baseLevel = (descriptor.dwords[3] >> 12u) & 0xfu;
        const auto fragments = (descriptor.dwords[3] >> 16u) & 0xfu;
        const auto maxMip = (descriptor.dwords[5] >> 4u) & 0xfu;
        return baseLevel == 0u && fragments >= 1u && fragments <= 3u && (r128 || maxMip == fragments);
    }
    return true;
}

std::uint32_t storageMipCount(const ImageResource& base, const DescriptorValue& descriptor) {
    if (base.mipMode != ImageMipMode::DynamicStorage || nullImageDescriptor(descriptor)) {
        return 1u;
    }
    const auto mipBase = (descriptor.dwords[3] >> 12u) & 0xfu;
    const auto mipLast = (descriptor.dwords[3] >> 16u) & 0xfu;
    return mipBase <= mipLast ? mipLast - mipBase + 1u : 0u;
}

struct ImageDecode {
    DecodedImage decoded;
    std::optional<PoisonReason> reason;
    std::string detail;
};

ImageDecode rejectImage(PoisonReason reason, std::string detail) {
    ImageDecode result;
    result.reason = reason;
    result.detail = std::move(detail);
    return result;
}

PoisonReason invalidImageReason(const DescriptorValue& descriptor) {
    const auto type = rawImageType(descriptor);
    if (type < ImageType::Color1D) {
        return PoisonReason::NotImage;
    }
    if (rawImageFormat(descriptor) == IrBufferFormat::Invalid) {
        return PoisonReason::InvalidFormat;
    }
    if (type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) {
        return PoisonReason::Multisampled;
    }
    return PoisonReason::Dimension;
}

ImageDecode inspectImageDescriptor(const DescriptorValue& descriptor, const ImageResource& base, std::uint32_t srgbDecodeFormats) {
    ImageDecode result;
    auto& decoded = result.decoded;
    decoded.mipCount = storageMipCount(base, descriptor);
    if (decoded.mipCount == 0u) {
        return rejectImage(PoisonReason::InvalidFormat, "storage image descriptor has an invalid mip range");
    }
    if (nullImageDescriptor(descriptor)) {
        decoded.numericClass = base.atomic ? IrTextureNumericClass::Uint : IrTextureNumericClass::Float;
        decoded.dimension = RdnaImageDimension::Dim2D;
        decoded.cube = false;
        return result;
    }
    if (base.resourceClass == ImageResourceClass::None || (base.atomic && base.resourceClass != ImageResourceClass::Storage)) {
        throw std::runtime_error("image resource has an invalid class");
    }
    if (!validImageDescriptor(descriptor, base.r128)) {
        return rejectImage(invalidImageReason(descriptor), "image descriptor is invalid");
    }
    decoded.dimension = descriptorDimension(descriptor, base.dimension);
    decoded.cube = descriptorIsCube(descriptor);
    const auto format = rawImageFormat(descriptor);
    if (base.atomic64 && format != IrBufferFormat::Format32_32UInt && format != IrBufferFormat::Format32_32SInt && format != IrBufferFormat::Format32_32Float) {
        return rejectImage(PoisonReason::InvalidFormat, "64-bit atomic image descriptor uses an unsupported format " + std::to_string(static_cast<std::uint32_t>(format)));
    }
    if (base.atomic && !base.atomic64 && format != IrBufferFormat::Format32UInt && format != IrBufferFormat::Format32SInt && format != IrBufferFormat::Format32Float) {
        return rejectImage(PoisonReason::InvalidFormat, "atomic image descriptor uses an unsupported format " + std::to_string(static_cast<std::uint32_t>(format)));
    }
    const bool storage = base.resourceClass == ImageResourceClass::Storage;
    const bool table = base.table != NoTable;
    decoded.fmask = IsFmaskTextureFormat(format);
    if (decoded.fmask && !base.fmaskCompatible) return rejectImage(PoisonReason::Fmask, "FMASK requires a direct 32-bit image load");
    if (decoded.fmask && (storage || base.depthCompare || table)) {
        return rejectImage(PoisonReason::Fmask, "FMASK requires a direct sampled image load");
    }
    if (base.packed) {
        if (table || (!storage && descriptorImageSwizzle(descriptor) != ShaderImageIdentitySwizzle)) {
            return rejectImage(PoisonReason::InvalidFormat, "packed image access requires a direct image, with identity swizzle when sampled");
        }
        decoded.packedFormat = format;
    }
    if (base.byElements != 0u) {
        const bool eightBit = format == IrBufferFormat::Format8UNorm || format == IrBufferFormat::Format8SNorm || format == IrBufferFormat::Format8UInt || format == IrBufferFormat::Format8SInt;
        const bool sixteenBit = format == IrBufferFormat::Format16UNorm || format == IrBufferFormat::Format16SNorm || format == IrBufferFormat::Format16UInt || format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format16Float;
        const bool eightBitPair = format == IrBufferFormat::Format8_8UNorm || format == IrBufferFormat::Format8_8SNorm || format == IrBufferFormat::Format8_8UInt || format == IrBufferFormat::Format8_8SInt;
        const bool packedEight = format == IrBufferFormat::Format8UNorm || format == IrBufferFormat::Format8UInt || format == IrBufferFormat::Format8SInt;
        const bool packedSixteen = format == IrBufferFormat::Format16UNorm || format == IrBufferFormat::Format16UInt || format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format8_8UNorm || format == IrBufferFormat::Format8_8UInt || format == IrBufferFormat::Format8_8SInt;
        const bool measured = base.packed ? packedEight || (base.byElements == 2u && packedSixteen)
                                          : base.byElements == 4u ? base.byComponents == 1u && eightBit : base.byElements == 2u && (base.byComponents == 1u ? eightBit || sixteenBit : base.byComponents == 2u && eightBitPair);
        if (!measured || descriptorImageSwizzle(descriptor) != ShaderImageIdentitySwizzle || rawImageType(descriptor) != ImageType::Color2D || table) {
            return rejectImage(PoisonReason::InvalidFormat, base.packed ? "MIMG PCK2/PCK4 requires a direct, identity-swizzled 2D UNORM, UINT or SINT image whose elements fill one dword: R8, R16 or RG8 for PCK2, R8 for PCK4"
                                                                     : "MIMG BY2/BY4 requires a direct, identity-swizzled 2D image whose elements fill one dword: R8, R16 or RG8 for BY2, R8 for BY4");
        }
    }
    decoded.conversionFormat = RemapTextureFormat(format) != format ? format : IrBufferFormat::Invalid;
    if (format == IrBufferFormat::Format11_11_10UNorm || format == IrBufferFormat::Format10_11_11Float) {
        const bool floating = format == IrBufferFormat::Format10_11_11Float;
        if (!base.srgbDecodeCompatible) return rejectImage(PoisonReason::Conversion, floating ? "samples or gathers a converted float image, or queries its LOD, which is not implemented" : "sampling, gathering or querying LOD of a converted unorm image is not implemented");
        if (!base.depthBitsCompatible) return rejectImage(PoisonReason::Conversion, floating ? "reads or writes a converted float image with 16-bit data, which is not implemented" : "reads or writes a converted unorm image with 16-bit data, which is not implemented");
        for (std::uint32_t component = 0; component < 4u; ++component) {
            if (((descriptorImageSwizzle(descriptor) >> (component * 3u)) & 7u) == 7u) return rejectImage(PoisonReason::Conversion, "selects a channel the converted image format does not have");
        }
    }
    decoded.srgbDecode = !storage && !base.depthCompare && (srgbDecodeFormats & SrgbDecodeBit(format)) != 0u;
    if (decoded.srgbDecode && !base.srgbDecodeCompatible) {
        return rejectImage(PoisonReason::SrgbUnsupported, "samples or gathers an sRGB image the device cannot sample, which is not implemented");
    }
    if (storage || decoded.conversionFormat != IrBufferFormat::Invalid) {
        decoded.shaderSwizzle = descriptorImageSwizzle(descriptor);
    }
    const bool wideSint = format == IrBufferFormat::Format32SInt || format == IrBufferFormat::Format32_32SInt || format == IrBufferFormat::Format32_32_32_32SInt;
    const bool narrowSint = format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format8_8SInt || format == IrBufferFormat::Format16_16SInt || format == IrBufferFormat::Format8_8_8_8SInt || format == IrBufferFormat::Format16_16_16_16SInt;
    const bool rawSintStorage = storage && (wideSint || (narrowSint && !base.packed)) && base.written && !base.read && !base.atomic;
    decoded.numericClass = base.atomic ? IrTextureNumericClass::Uint : SampledTextureNumericClass(format);
    if (!storage && !base.depthCompare && IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3])) {
        decoded.depthBits = true;
        if (!base.depthBitsCompatible) return rejectImage(PoisonReason::DepthBits16, "runtime image reads depth bits as unsupported 16-bit results");
        decoded.depthUnorm16 = DepthBitsTextureWidth(descriptor.dwords[1], descriptor.dwords[3]) == 16u;
        decoded.numericClass = IrTextureNumericClass::Float;
        decoded.shaderSwizzle = descriptorImageSwizzle(descriptor);
    }
    if (storage) {
        if ((!rawSintStorage && decoded.numericClass == IrTextureNumericClass::Sint) || decoded.numericClass == IrTextureNumericClass::Unsupported) {
            return rejectImage(PoisonReason::NumericClass, "storage image descriptor uses an unsupported format");
        }
        if (rawSintStorage) {
            decoded.numericClass = IrTextureNumericClass::Uint;
        }
        if ((rawSintStorage || (base.atomic && format == IrBufferFormat::Format32SInt)) && !base.packed) {
            if (!base.depthBitsCompatible) return rejectImage(PoisonReason::InvalidFormat, "stores 16-bit data to an image of a SINT format");
            decoded.conversionFormat = format;
        }
    } else if (decoded.numericClass == IrTextureNumericClass::Unsupported || (base.depthCompare && decoded.numericClass != IrTextureNumericClass::Float)) {
        return rejectImage(PoisonReason::NumericClass, "sampled image descriptor uses an unsupported format");
    }
    return result;
}

DecodedImage decodeImageDescriptor(const DescriptorValue& descriptor, const ImageResource& base, std::uint32_t srgbDecodeFormats) {
    auto result = inspectImageDescriptor(descriptor, base, srgbDecodeFormats);
    if (result.reason.has_value()) {
        throw std::runtime_error(result.detail);
    }
    return result.decoded;
}

bool requiresPointSampler(const ImageResource& mode) {
    return mode.numericClass == IrTextureNumericClass::Uint || mode.numericClass == IrTextureNumericClass::Sint || mode.conversionFormat != IrBufferFormat::Invalid || mode.depthBits;
}

bool reducesBetweenTexels(std::uint32_t word0, std::uint32_t filter) {
    return ((word0 >> 29u) & 3u) != 0u && (((filter >> 20u) & 0xfu) != 0u || ((filter >> 26u) & 3u) == 2u);
}

std::string hexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

std::uint32_t TableKeyScanLimit() {
    static const std::uint32_t limit = [] {
        const char* text = std::getenv("APS5_IMAGE_TABLE_KEY_SCAN");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : 256u;
    }();
    return limit;
}

bool TableStrict() {
    static const bool strict = [] {
        const char* text = std::getenv("APS5_IMAGE_TABLE_STRICT");
        return text != nullptr && std::strcmp(text, "0") != 0;
    }();
    return strict;
}

bool TableTraced() {
    static const bool traced = std::getenv("APS5_TRACE_IMAGE_TABLE") != nullptr;
    return traced;
}

constexpr std::size_t PoisonReasonCount = static_cast<std::size_t>(PoisonReason::Count);

struct TableCounters {
    std::atomic<std::uint64_t> snapshots{0};
    std::atomic<std::uint64_t> tables{0};
    std::atomic<std::uint64_t> keys{0};
    std::array<std::atomic<std::uint64_t>, PoisonReasonCount> poison{};
    std::atomic<std::uint64_t> narrowed{0};
    std::atomic<std::uint64_t> narrowSkipped{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    // Column memo outcomes: the previous walk reused whole, its read plan reused with changed
    // words, a fresh read plan; and the columns checked against the word-by-word walk with their
    // differences (APS5_VERIFY_TABLE_MEMO=1).
    std::atomic<std::uint64_t> memoHits{0};
    std::atomic<std::uint64_t> memoChanged{0};
    std::atomic<std::uint64_t> memoPlanned{0};
    // Memo hits that skipped the reads (the column's pages proved unchanged and replayed), and the
    // memos armed for that (page images taken after a hit).
    std::atomic<std::uint64_t> memoSkipped{0};
    std::atomic<std::uint64_t> memoArmed{0};
    // Phase times (APS5_PROFILE_DRAW): a column's setup (its table descriptor, key count, narrowing),
    // the skip's proof and page replay, the skipped column's output (lookups or the cached codes),
    // the full walks, and the per-key codes of the snapshot; skipped columns that took the cached
    // codes whole.
    std::atomic<std::uint64_t> setupNanoseconds{0};
    std::atomic<std::uint64_t> proofNanoseconds{0};
    std::atomic<std::uint64_t> outputNanoseconds{0};
    std::atomic<std::uint64_t> walkNanoseconds{0};
    std::atomic<std::uint64_t> codesNanoseconds{0};
    std::atomic<std::uint64_t> outputsReused{0};
    // Snapshots reused whole (TableSnapshotter::ReuseWhole), and those checked against the full
    // path with their differences (APS5_VERIFY_TABLE_MEMO=1).
    std::atomic<std::uint64_t> wholeReused{0};
    std::atomic<std::uint64_t> wholeVerified{0};
    std::atomic<std::uint64_t> wholeMismatches{0};
    std::atomic<std::uint64_t> verified{0};
    std::atomic<std::uint64_t> verifyMismatches{0};
    std::atomic<long long> lastReport{0};
};

TableCounters& tableCounters() {
    static TableCounters counters;
    return counters;
}

void reportTables() {
    if (!MaterializeProfiled()) return;
    auto& counters = tableCounters();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ll || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto snapshots = counters.snapshots.exchange(0, std::memory_order_relaxed);
    if (snapshots == 0) return;
    std::string poison;
    for (std::size_t reason = 0; reason < PoisonReasonCount; reason++) {
        const auto count = counters.poison[reason].exchange(0, std::memory_order_relaxed);
        if (count == 0) continue;
        poison += " " + std::string(ImageTableAbi::PoisonReasonName(static_cast<PoisonReason>(reason))) + "=" + std::to_string(count);
    }
    std::fprintf(stderr, "[image-table] (10 s): %llu snapshots, %llu tables, %llu keys, structural poison:%s, narrowing %llu applied / %llu skipped, %.1f ms; column memo %llu hits (%llu skipped the reads, %llu armed, %llu outputs reused), %llu changed words, %llu planned; verified %llu columns, %llu mismatches; whole snapshots reused %llu (verified %llu, %llu mismatches); phases ms: setup %.1f, skip proof+replay %.1f, skip output %.1f, full walks %.1f, codes %.1f\n",
        static_cast<unsigned long long>(snapshots), static_cast<unsigned long long>(counters.tables.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.keys.exchange(0, std::memory_order_relaxed)), poison.empty() ? " none" : poison.c_str(),
        static_cast<unsigned long long>(counters.narrowed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.narrowSkipped.exchange(0, std::memory_order_relaxed)), static_cast<double>(counters.nanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6,
        static_cast<unsigned long long>(counters.memoHits.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoSkipped.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoArmed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.outputsReused.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoChanged.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoPlanned.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(counters.verified.load(std::memory_order_relaxed)), static_cast<unsigned long long>(counters.verifyMismatches.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(counters.wholeReused.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.wholeVerified.load(std::memory_order_relaxed)), static_cast<unsigned long long>(counters.wholeMismatches.load(std::memory_order_relaxed)),
        static_cast<double>(counters.setupNanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6, static_cast<double>(counters.proofNanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6, static_cast<double>(counters.outputNanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6, static_cast<double>(counters.walkNanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6, static_cast<double>(counters.codesNanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6);
}

enum class RecordState : std::uint8_t { OutsideDomain, Read, Unmapped };

struct ColumnRecords {
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint32_t records = 0;
    std::uint32_t keys = 0;
    std::uint32_t fault = 0;
    bool narrowed = false;
    bool outside = false;
    std::vector<RecordState> states;
    std::vector<std::uint32_t> words;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> readRanges;
    // Every record was read (no poison code): the snapshot's codes are `words` as they are.
    bool allRead = false;
    // The memo whose skip gave this column's output whole (its codes taken as they were); null
    // when anything else made it (a whole-snapshot memo is built only from such columns).
    struct ColumnMemo* pure = nullptr;
};

bool TableMemo() {
    static const bool memo = std::getenv("APS5_NO_TABLE_MEMO") == nullptr;
    return memo;
}

bool TableMemoVerified() {
    static const bool verify = std::getenv("APS5_VERIFY_TABLE_MEMO") != nullptr;
    return verify;
}

// The most pages a column memo keeps images of for skipping its reads (16 MiB of table).
constexpr std::size_t TableSkipPages = 4096;

// A page of a column's table as read after the memo's generation was collected, and the dwords the
// column's runs read in it.
struct TablePageImage {
    std::uint64_t base = 0;
    std::array<std::uint32_t, 1024> words{};
    std::array<std::uint64_t, 16> mask{};
};

// What a column's record walk reads, without the words: the accessibility answers it depended on
// (one per change of page, in walk order: PageAccessible for a buffer column, the whole-page answer
// for a pointer column), the dwords it reads as contiguous runs in read order, and the outcome that
// follows from those answers alone (record states, which dwords of each record were read, the
// pointer column's read ranges and bounds). Equal inputs and equal answers give an equal walk.
struct ColumnPlan {
    std::vector<std::pair<std::uint64_t, bool>> checks;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    std::vector<RecordState> states;
    std::vector<std::uint32_t> first;
    std::vector<std::uint8_t> mask;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> readRanges;
    std::uint64_t low = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t high = 0;
    std::uint64_t words = 0;
    // A pointer column that met a page not accessible whole asked per dword: not replayed.
    bool replayable = true;
};

// A column's previous walk on this thread: its inputs, plan, the words it read, and the distinct
// descriptors in first-occurrence order with each record's index among them. Reused when the
// inputs match, every recorded accessibility answer is the same and the plan's runs read the same
// words again; the reads themselves always happen, so the capture records the same dwords.
struct ColumnMemo {
    bool address = false;
    bool sampler = false;
    std::uint32_t dwordCount = 0;
    std::uint32_t stride = 0;
    std::uint32_t addend = 0;
    std::uint32_t offset = 0;
    std::uint32_t keys = 0;
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::vector<RecordState> statesIn;
    ColumnPlan plan;
    bool valued = false;
    std::vector<std::uint32_t> words;
    std::vector<DescriptorValue> distinct;
    std::vector<std::uint32_t> local;
    std::uint64_t used = 0;
    // Armed (nonzero): the write generation collected before the reads of a hit whose words these
    // are, the page-aligned spans the plan's runs read, and those pages' images taken after it.
    // While SrtTableMemory::unchanged holds for every span, a capture replays the images instead of
    // reading (WalkRecords); anything else disarms the memo until a later hit arms it again.
    std::uint64_t generation = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    // Shared with the captures that replayed them (their pages refer to the dwords).
    std::vector<std::shared_ptr<const TablePageImage>> images;
    // The column's codes as a hit's Reuse produced them by appending its descriptors at
    // `codeBase` (the snapshot's words before it, with none of their width yet): a later hit at the
    // same position takes them whole. npos: none kept.
    std::vector<std::uint32_t> codes;
    std::size_t codeBase = std::numeric_limits<std::size_t>::max();
    bool allRead = false;
    // Moves with every change of what a skip of this memo outputs (a disarm, a walk that was not a
    // pure skip, the slot taken for another column): a whole-snapshot memo records it.
    std::uint64_t version = 0;

    void Disarm() {
        ++version;
        generation = 0;
        spans.clear();
        images.clear();
        codes.clear();
        codeBase = std::numeric_limits<std::size_t>::max();
    }

    bool SameColumn(const TableColumn& column) const {
        return address == column.address && sampler == column.sampler && dwordCount == column.dwordCount && stride == column.stride && addend == column.addend && offset == column.offset;
    }

    bool Matches(const TableColumn& column, const ColumnRecords& records) const {
        // The states as bytes: the element-wise compare of the enum vector ran a byte at a time
        // (~19k compares per matched column of a large table).
        return address == column.address && sampler == column.sampler && dwordCount == column.dwordCount && stride == column.stride && addend == column.addend && offset == column.offset && keys == records.keys && base == records.base && size == records.size && statesIn.size() == records.states.size() && (statesIn.empty() || std::memcmp(statesIn.data(), records.states.data(), statesIn.size() * sizeof(RecordState)) == 0);
    }
};

constexpr std::size_t ColumnMemoEntries = 8;

struct ColumnMemos {
    std::vector<ColumnMemo> entries;
    std::uint64_t clock = 0;
};

ColumnMemos& columnMemos() {
    // Reserved whole: a whole-snapshot memo refers to its column memos by address.
    thread_local ColumnMemos memos = [] {
        ColumnMemos made;
        made.entries.reserve(ColumnMemoEntries);
        return made;
    }();
    return memos;
}

// A plan's whole snapshot made of pure skips only (every column's output its memo's, unchanged):
// while every column's memo is the same (version), its table the same (base, size, records, keys)
// and its skip proof holds again, the snapshot is this output again, hash included
// (TableSnapshotter::ReuseWhole; APS5_NO_WHOLE_TABLE_REUSE=1 disables).
struct SnapshotMemoColumn {
    std::uint32_t source = 0;
    ColumnMemo* memo = nullptr;
    std::uint64_t version = 0;
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint32_t records = 0;
    std::uint32_t keys = 0;
    bool outside = false;
};

struct SnapshotMemo {
    const IrResourcePlan* plan = nullptr;
    std::uint64_t shaderHash = 0;
    std::vector<SnapshotMemoColumn> columns;
    ImageTableSnapshot output;
    std::uint64_t used = 0;
};

constexpr std::size_t SnapshotMemoEntries = 8;

struct SnapshotMemos {
    std::vector<SnapshotMemo> entries;
    std::uint64_t clock = 0;
};

SnapshotMemos& snapshotMemos() {
    thread_local SnapshotMemos memos;
    return memos;
}

bool WholeTableReuse() {
    static const bool reuse = std::getenv("APS5_NO_WHOLE_TABLE_REUSE") == nullptr;
    return reuse;
}

struct TableTrace {
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint32_t records = 0;
    std::uint32_t keys = 0;
    std::uint32_t poison = 0;
    bool narrowed = false;

    bool operator==(const TableTrace& other) const = default;
};

void traceTable(const IrResourcePlan& plan, std::uint32_t table, const TableColumn& column, const TableTrace& trace) {
    static std::mutex mutex;
    static std::map<std::pair<std::uint64_t, std::uint32_t>, TableTrace> seen;
    std::lock_guard lock(mutex);
    auto& last = seen[{plan.shaderHash, table}];
    if (last == trace) return;
    last = trace;
    std::fprintf(stderr, "[image-table] shader 0x%llx table %u (%s): base 0x%llx size 0x%llx stride 0x%x addend 0x%x offset 0x%x dwords %u, %u records, %u keys%s, %u structural poison\n", static_cast<unsigned long long>(plan.shaderHash), table, column.sampler ? "sampler" : "image", static_cast<unsigned long long>(trace.base), static_cast<unsigned long long>(trace.size), column.stride, column.addend, column.offset, column.dwordCount, trace.records, trace.keys, trace.narrowed ? " (narrowed)" : "", trace.poison);
}

class TableSnapshotter {
public:
    TableSnapshotter(const IrResourcePlan& plan, const SrtRuntime& runtime, SrtWalker& walker, const std::vector<std::uint8_t>& activeSources, ImageTableSnapshot& tables) : plan(plan), runtime(runtime), walker(walker), activeSources(activeSources), tables(tables) {}

    void Run() {
        tables = ImageTableSnapshot{};
        std::uint32_t count = 0;
        for (const auto& image : plan.info.images) {
            if (image.table != NoTable) count = std::max(count, image.table + 1u);
        }
        for (const auto& sampler : plan.info.samplers) {
            if (sampler.table != NoTable) count = std::max(count, sampler.table + 1u);
        }
        if (count == 0u) {
            return;
        }
        if (runtime.readMemory == nullptr) {
            throw std::runtime_error("image table snapshot requires runtime memory access");
        }
        const auto started = std::chrono::steady_clock::now();
        tables.shader = plan.shaderHash;
        if (!ReuseWhole(count)) {
            RunColumns(count);
            RememberWhole(count);
        }
        auto& counters = tableCounters();
        counters.snapshots.fetch_add(1, std::memory_order_relaxed);
        counters.tables.fetch_add(count, std::memory_order_relaxed);
        counters.nanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
        if (TableStrict()) {
            for (std::uint32_t table = 0; table < tables.tables.size(); table++) {
                const auto& column = tables.tables[table];
                const auto poisoned = std::find_if(column.codes.begin(), column.codes.end(), [](std::uint32_t code) { return (code & ImageTableAbi::PoisonFlag) != 0u && (code & ~ImageTableAbi::PoisonFlag) != static_cast<std::uint32_t>(PoisonReason::OutsideDomain); });
                if (column.fault != 0u || poisoned != column.codes.end()) {
                    const auto reason = static_cast<PoisonReason>((column.fault != 0u ? column.fault : *poisoned) & ~ImageTableAbi::PoisonFlag);
                    throw std::runtime_error("image table " + std::to_string(table) + " at 0x" + hexText(column.base) + " holds a record that cannot be read (" + std::string(ImageTableAbi::PoisonReasonName(reason)) + "), and APS5_IMAGE_TABLE_STRICT is set");
                }
            }
        }
    }

private:
    static constexpr std::uint64_t PageBytes = 4096u;

    void RunColumns(std::uint32_t count) {
        tables.tables.resize(count);
        for (const auto& image : plan.info.images) {
            if (image.table != NoTable) Snapshot(image.source, image.table);
        }
        for (const auto& sampler : plan.info.samplers) {
            if (sampler.table != NoTable) Snapshot(sampler.source, sampler.table);
        }
    }

    // The table sources in the order RunColumns takes them, each once.
    std::vector<std::uint32_t> TableSources() const {
        std::vector<std::uint32_t> sources;
        const auto add = [&](std::uint32_t source) {
            if (std::find(sources.begin(), sources.end(), source) == sources.end()) sources.push_back(source);
        };
        for (const auto& image : plan.info.images) {
            if (image.table != NoTable) add(image.source);
        }
        for (const auto& sampler : plan.info.samplers) {
            if (sampler.table != NoTable) add(sampler.source);
        }
        return sources;
    }

    // A column's table as Column() evaluates it (the same reads): base, size, records, keys and
    // outside; false for a fault (not a buffer).
    bool TableInputs(const TableColumn& column, ColumnRecords& records) {
        if (column.address) {
            DescriptorValue pointer;
            walker.EvaluateDescriptorSource(plan, column.heapSource, runtime, pointer);
            records.base = (static_cast<std::uint64_t>(pointer.dwords[1]) << 32u) | pointer.dwords[0];
            records.records = ImageTableAbi::ScalarAddressRecords(column.stride);
        } else {
            const auto heap = EvaluateBuffer(column.heapSource);
            if (heap.Type() != 0u) {
                records.fault = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::NotBuffer));
                return false;
            }
            records.base = heap.Base48();
            records.size = heap.GetSize();
            records.records = ImageTableAbi::ScalarBufferRecords(column.addend, column.stride, column.offset, records.size);
        }
        records.records = std::min(records.records, ImageTableAbi::KeyRecords(column.maxKey, column.stride));
        records.keys = std::min(records.records, ImageTableAbi::MaxKeys);
        records.outside = records.keys < records.records;
        return true;
    }

    // The whole previous snapshot of the plan when every column would be a pure skip of the same
    // memo over the same table: the tables' inputs are evaluated (their reads recorded as the full
    // path records them), every memo's skip proof is made (its pages replayed), and the output is
    // taken whole, hash included.
    bool ReuseWhole(std::uint32_t count) {
        if (!WholeTableReuse() || !TableMemo() || runtime.tableMemory == nullptr) return false;
        auto& memos = snapshotMemos();
        const auto found = std::find_if(memos.entries.begin(), memos.entries.end(), [&](const SnapshotMemo& entry) { return entry.plan == &plan && entry.shaderHash == plan.shaderHash && entry.output.tables.size() == count; });
        if (found == memos.entries.end()) return false;
        auto& memo = *found;
        for (const auto& entry : memo.columns) {
            if (entry.memo->version != entry.version || entry.memo->generation == 0) return false;
            const auto& column = *plan.descriptorSources.at(entry.source).tableColumn;
            if (column.keyDomain.has_value() || !entry.memo->SameColumn(column)) return false;
            if (entry.source < activeSources.size() && activeSources[entry.source] == 0u) return false;
            ColumnRecords inputs;
            if (!TableInputs(column, inputs) || inputs.base != entry.base || inputs.size != entry.size || inputs.records != entry.records || inputs.keys != entry.keys || inputs.outside != entry.outside) return false;
        }
        const bool profiled = MaterializeProfiled();
        const auto started = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (const auto& entry : memo.columns) {
            if (!SkipReads(*entry.memo)) return false;
        }
        auto& counters = tableCounters();
        if (profiled) counters.proofNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
        tables = memo.output;
        memo.used = ++memos.clock;
        auto& columnClock = columnMemos();
        for (const auto& entry : memo.columns) entry.memo->used = ++columnClock.clock;
        const auto columnsReused = static_cast<std::uint64_t>(memo.columns.size());
        counters.memoHits.fetch_add(columnsReused, std::memory_order_relaxed);
        counters.memoSkipped.fetch_add(columnsReused, std::memory_order_relaxed);
        counters.outputsReused.fetch_add(columnsReused, std::memory_order_relaxed);
        counters.wholeReused.fetch_add(1, std::memory_order_relaxed);
        for (const auto& table : tables.tables) counters.keys.fetch_add(table.keys, std::memory_order_relaxed);
        if (TableMemoVerified()) VerifyWhole(count);
        return true;
    }

    // APS5_VERIFY_TABLE_MEMO=1: the full path again (into its own snapshot), compared with the
    // reused one, and the carried hash with one computed from it.
    void VerifyWhole(std::uint32_t count) {
        ImageTableSnapshot fresh;
        fresh.shader = plan.shaderHash;
        TableSnapshotter full(plan, runtime, walker, activeSources, fresh);
        full.RunColumns(count);
        auto& counters = tableCounters();
        counters.wholeVerified.fetch_add(1, std::memory_order_relaxed);
        const bool same = fresh == tables;
        const bool hashed = ResultKey::TablesHash(fresh) == tables.hash;
        if (same && hashed) return;
        counters.wholeMismatches.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<std::uint32_t> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 20u) std::fprintf(stderr, "[image-table] verify: shader 0x%llx's whole reused snapshot differs from the full path (%s)\n", static_cast<unsigned long long>(plan.shaderHash), !same ? "the snapshot" : "the hash");
    }

    // After the full path: a snapshot whose every column is a pure skip is kept for ReuseWhole,
    // with its hash (computed once here).
    void RememberWhole(std::uint32_t count) {
        if (!WholeTableReuse() || !TableMemo() || runtime.tableMemory == nullptr) return;
        SnapshotMemo memo;
        for (const auto source : TableSources()) {
            const auto found = columns.find(source);
            if (found == columns.end()) return;
            const auto& records = found->second;
            // A key-domain column (narrowed or not) is never reused whole (ReuseWhole).
            if (records.pure == nullptr || records.fault != 0u || records.narrowed || plan.descriptorSources.at(source).tableColumn->keyDomain.has_value()) return;
            memo.columns.push_back({source, records.pure, records.pure->version, records.base, records.size, records.records, records.keys, records.outside});
        }
        if (memo.columns.empty()) return;
        tables.hash = ResultKey::TablesHash(tables);
        memo.plan = &plan;
        memo.shaderHash = plan.shaderHash;
        memo.output = tables;
        auto& memos = snapshotMemos();
        memo.used = ++memos.clock;
        const auto same = std::find_if(memos.entries.begin(), memos.entries.end(), [&](const SnapshotMemo& entry) { return entry.plan == &plan && entry.shaderHash == plan.shaderHash && entry.output.tables.size() == count; });
        if (same != memos.entries.end()) *same = std::move(memo);
        else if (memos.entries.size() < SnapshotMemoEntries) memos.entries.push_back(std::move(memo));
        else *std::min_element(memos.entries.begin(), memos.entries.end(), [](const SnapshotMemo& a, const SnapshotMemo& b) { return a.used < b.used; }) = std::move(memo);
    }

    bool Accessible(std::uint64_t address, std::uint64_t bytes) const {
        return runtime.accessible == nullptr || runtime.accessible(runtime.userContext, address, bytes);
    }

    bool PageAccessible(std::uint64_t address, std::uint64_t begin, std::uint64_t end) {
        const auto page = address & ~(PageBytes - 1u);
        if (page == lastAccessPage) return lastAccessible;
        if (const auto found = pages.find(page); found != pages.end()) {
            lastAccessPage = page;
            lastAccessible = found->second;
            return found->second;
        }
        const auto first = std::max(page, begin);
        const auto last = std::min(page + PageBytes, end);
        const bool accessible = first < last && Accessible(first, last - first);
        pages.emplace(page, accessible);
        lastAccessPage = page;
        lastAccessible = accessible;
        return accessible;
    }

    bool WholePage(std::uint64_t address) {
        const auto page = address & ~(PageBytes - 1u);
        const auto [found, inserted] = wholePages.try_emplace(page, false);
        if (inserted) found->second = Accessible(page, PageBytes);
        return found->second;
    }

    bool AddressAccessible(std::uint64_t address) {
        return WholePage(address) || Accessible(address, sizeof(std::uint32_t));
    }

    std::uint32_t Read(std::uint64_t address) const {
        std::uint32_t word = 0;
        if (!runtime.readMemory(runtime.userContext, address, &word)) {
            throw std::runtime_error("image table: failed to read guest memory at 0x" + hexText(address));
        }
        return word;
    }

    ShaderBufferResource EvaluateBuffer(std::uint32_t source) const {
        DescriptorValue value;
        walker.EvaluateDescriptorSource(plan, source, runtime, value);
        return decodeBufferDescriptor(value);
    }

    void Narrow(const TableColumn& column, ColumnRecords& records) {
        const auto& domain = *column.keyDomain;
        const auto skip = [&] {
            tableCounters().narrowSkipped.fetch_add(1, std::memory_order_relaxed);
        };
        const auto keyBuffer = EvaluateBuffer(domain.source);
        if (keyBuffer.Type() != 0u) {
            skip();
            return;
        }
        const auto base = keyBuffer.Base48();
        const auto size = keyBuffer.GetSize();
        const std::uint64_t first = domain.offset & ~3u;
        const std::uint64_t positions = size >= first + 4u ? (size - first - 4u) / domain.stride + 1u : 0u;
        if (positions > TableKeyScanLimit() || first > 0xffffffffull) {
            skip();
            return;
        }
        if (positions != 0u && runtime.pendingWrite != nullptr && runtime.pendingWrite(runtime.userContext, base + first, size - first)) {
            skip();
            return;
        }
        std::vector<std::uint32_t> values{0u};
        for (std::uint64_t index = 0; index < positions; index++) {
            const auto address = base + first + index * domain.stride;
            if (!PageAccessible(address, base, base + size)) {
                skip();
                return;
            }
            values.push_back(Read(address));
        }
        records.states.assign(records.keys, RecordState::OutsideDomain);
        records.outside = false;
        for (const auto value : values) {
            const std::uint32_t relative = value * column.stride;
            const bool aligned = relative % column.stride == 0u;
            const auto record = relative / column.stride;
            if (aligned && record < records.keys) {
                records.states[record] = RecordState::Read;
                continue;
            }
            if (column.address || ImageTableAbi::ScalarBufferDword(relative + column.addend, column.offset, records.size).has_value()) records.outside = true;
        }
        records.narrowed = true;
        if (positions != 0u) tables.ranges.emplace_back(base, size);
        tableCounters().narrowed.fetch_add(1, std::memory_order_relaxed);
    }

    // Words appended in bulk by a reused column (Reuse) enter the index on the next lookup, in
    // order and keeping the first index of a value, as one lookup each would have.
    std::uint32_t WordsIndex(const DescriptorValue& value) {
        for (; indexed < tables.words.size(); ++indexed) wordsIndex.try_emplace({tables.words[indexed].dwords, tables.words[indexed].dwordCount}, static_cast<std::uint32_t>(indexed));
        const auto [found, inserted] = wordsIndex.try_emplace({value.dwords, value.dwordCount}, static_cast<std::uint32_t>(tables.words.size()));
        if (inserted) {
            tables.words.push_back(value);
            ++indexed;
            ++widths.at(value.dwordCount);
        }
        return found->second;
    }

    ColumnPlan PlanColumn(const TableColumn& column, const ColumnRecords& records) {
        ColumnPlan plan;
        plan.states = records.states;
        plan.first.assign(records.keys, 0u);
        plan.mask.assign(records.keys, 0u);
        const auto end = records.base + records.size;
        std::uint64_t checked = std::numeric_limits<std::uint64_t>::max();
        for (std::uint32_t record = 0; record < records.keys; record++) {
            if (plan.states[record] != RecordState::Read) continue;
            plan.first[record] = static_cast<std::uint32_t>(plan.words);
            const auto offset = column.addend + record * column.stride;
            for (std::uint32_t dword = 0; dword < column.dwordCount; dword++) {
                std::optional<std::uint64_t> address;
                if (column.address) {
                    address = ImageTableAbi::ScalarAddressDword(records.base, offset, column.offset + dword * 4u);
                    if (address.has_value() && (*address & 3u) != 0u) address.reset();
                } else if (const auto position = ImageTableAbi::ScalarBufferDword(offset, column.offset + dword * 4u, records.size); position.has_value()) {
                    address = records.base + *position;
                } else {
                    continue;
                }
                bool accessible = false;
                if (address.has_value()) {
                    const auto page = *address & ~(PageBytes - 1u);
                    if (column.address) {
                        const bool whole = WholePage(*address);
                        accessible = whole || Accessible(*address, sizeof(std::uint32_t));
                        if (!whole) plan.replayable = false;
                        if (page != checked) plan.checks.emplace_back(*address, whole);
                    } else {
                        accessible = PageAccessible(*address, records.base, end);
                        if (page != checked) plan.checks.emplace_back(*address, accessible);
                    }
                    checked = page;
                }
                if (!accessible) {
                    plan.states[record] = RecordState::Unmapped;
                    break;
                }
                plan.mask[record] |= static_cast<std::uint8_t>(1u << dword);
                if (!plan.runs.empty() && *address > plan.runs.back().first && *address - plan.runs.back().first == plan.runs.back().second * sizeof(std::uint32_t)) {
                    ++plan.runs.back().second;
                } else {
                    plan.runs.emplace_back(*address, 1u);
                }
                ++plan.words;
                if (column.address) {
                    if (!plan.readRanges.empty() && *address - plan.readRanges.back().first == plan.readRanges.back().second) {
                        plan.readRanges.back().second += sizeof(std::uint32_t);
                    } else {
                        plan.readRanges.emplace_back(*address, sizeof(std::uint32_t));
                    }
                }
                plan.low = std::min(plan.low, *address);
                plan.high = std::max(plan.high, *address + sizeof(std::uint32_t));
            }
        }
        std::sort(plan.readRanges.begin(), plan.readRanges.end());
        std::size_t ranges = 0;
        for (const auto& range : plan.readRanges) {
            if (ranges != 0u && range.first - plan.readRanges[ranges - 1u].first <= plan.readRanges[ranges - 1u].second) {
                auto& previous = plan.readRanges[ranges - 1u];
                previous.second = std::max(previous.second, range.first - previous.first + range.second);
            } else {
                plan.readRanges[ranges++] = range;
            }
        }
        plan.readRanges.resize(ranges);
        return plan;
    }

    // Asks the plan's accessibility questions again, in its order: equal answers mean the walk
    // would take the same path.
    bool Replay(const ColumnPlan& plan, const TableColumn& column, const ColumnRecords& records) {
        if (!plan.replayable) return false;
        const auto end = records.base + records.size;
        for (const auto& [address, answer] : plan.checks) {
            if ((column.address ? WholePage(address) : PageAccessible(address, records.base, end)) != answer) return false;
        }
        return true;
    }

    void ReadRuns(const ColumnPlan& plan, std::vector<std::uint32_t>& words) const {
        words.resize(static_cast<std::size_t>(plan.words));
        std::size_t at = 0;
        for (const auto& [address, count] : plan.runs) {
            if (runtime.readWords != nullptr) {
                if (!runtime.readWords(runtime.userContext, address, words.data() + at, count)) {
                    throw std::runtime_error("image table: failed to read guest memory at 0x" + hexText(address));
                }
            } else {
                for (std::uint64_t word = 0; word < count; word++) words[at + word] = Read(address + word * sizeof(std::uint32_t));
            }
            at += static_cast<std::size_t>(count);
        }
    }

    DescriptorValue RecordValue(const TableColumn& column, const ColumnPlan& plan, const std::vector<std::uint32_t>& words, std::uint32_t record) const {
        DescriptorValue value;
        value.dwordCount = column.sampler ? 4u : 8u;
        auto at = plan.first[record];
        for (std::uint32_t dword = 0; dword < column.dwordCount; dword++) {
            if ((plan.mask[record] & (1u << dword)) != 0u) value.dwords[dword] = words[at++];
        }
        return value;
    }

    // The per-record lookups of the walk, with the column's distinct descriptors recorded for a memo.
    void Assemble(const TableColumn& column, const ColumnPlan& plan, const std::vector<std::uint32_t>& words, ColumnRecords& records, std::vector<DescriptorValue>& distinct, std::vector<std::uint32_t>& local) {
        local.assign(records.keys, 0u);
        std::vector<std::uint32_t> seen;
        for (std::uint32_t record = 0; record < records.keys; record++) {
            if (plan.states[record] != RecordState::Read) continue;
            const auto value = RecordValue(column, plan, words, record);
            const auto code = WordsIndex(value);
            records.words[record] = code;
            if (code >= seen.size()) seen.resize(tables.words.size(), 0u);
            if (seen[code] == 0u) {
                distinct.push_back(value);
                seen[code] = static_cast<std::uint32_t>(distinct.size());
            }
            local[record] = seen[code] - 1u;
        }
    }

    // The lookups of an unchanged column: one per distinct descriptor in first-occurrence order
    // gives the codes and the words order the per-record lookups would. When no word of this width
    // is in the snapshot yet none of them can be found, so they are appended in that order.
    void Reuse(const ColumnMemo& memo, ColumnRecords& records) {
        const auto width = memo.sampler ? 4u : 8u;
        std::vector<std::uint32_t> codes(memo.distinct.size());
        if (widths.at(width) == 0u) {
            const auto first = static_cast<std::uint32_t>(tables.words.size());
            tables.words.insert(tables.words.end(), memo.distinct.begin(), memo.distinct.end());
            widths.at(width) += static_cast<std::uint32_t>(memo.distinct.size());
            for (std::uint32_t index = 0; index < codes.size(); index++) codes[index] = first + index;
        } else {
            for (std::uint32_t index = 0; index < codes.size(); index++) codes[index] = WordsIndex(memo.distinct[index]);
        }
        for (std::uint32_t record = 0; record < records.keys; record++) {
            if (memo.plan.states[record] == RecordState::Read) records.words[record] = codes[memo.local[record]];
        }
    }

    // The page-aligned spans [begin, end) the plan's runs read, sorted and merged; empty when the
    // plan cannot be skipped (an accessibility answer was false, or more than TableSkipPages pages).
    static std::vector<std::pair<std::uint64_t, std::uint64_t>> SkipSpans(const ColumnPlan& plan) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
        if (!plan.replayable || plan.runs.empty()) return spans;
        for (const auto& [address, answer] : plan.checks) {
            if (!answer) return spans;
        }
        spans.reserve(plan.runs.size());
        for (const auto& [address, count] : plan.runs) {
            const auto bytes = count * sizeof(std::uint32_t);
            if (address > std::numeric_limits<std::uint64_t>::max() - bytes - PageBytes) return {};
            spans.emplace_back(address & ~(PageBytes - 1u), (address + bytes + PageBytes - 1u) & ~(PageBytes - 1u));
        }
        std::sort(spans.begin(), spans.end());
        std::size_t merged = 0;
        std::uint64_t pages = 0;
        for (const auto& span : spans) {
            if (merged != 0u && span.first <= spans[merged - 1u].second) {
                spans[merged - 1u].second = std::max(spans[merged - 1u].second, span.second);
            } else {
                spans[merged++] = span;
            }
        }
        spans.resize(merged);
        for (const auto& [begin, end] : spans) pages += (end - begin) / PageBytes;
        if (pages > TableSkipPages) spans.clear();
        return spans;
    }

    // One generation for every span (the newest of their collects; 0 when any is untracked).
    std::uint64_t CollectSpans(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& spans) const {
        std::uint64_t generation = 0;
        for (const auto& [begin, end] : spans) {
            const auto collected = runtime.tableMemory->collect(runtime.userContext, begin, end - begin);
            if (collected == 0) return 0;
            generation = std::max(generation, collected);
        }
        return generation;
    }

    // Takes the images of an unchanged column's pages after a hit: read now (after `generation` was
    // collected), they must give the very words the hit read, which the memo's lookups describe.
    void Arm(ColumnMemo& memo, const std::vector<std::uint32_t>& words, std::vector<std::pair<std::uint64_t, std::uint64_t>> spans, std::uint64_t generation) const {
        memo.Disarm();
        if (generation == 0 || spans.empty()) return;
        std::vector<std::shared_ptr<TablePageImage>> images;
        for (const auto& [begin, end] : spans) {
            for (auto base = begin; base < end; base += PageBytes) {
                auto& image = images.emplace_back(std::make_shared<TablePageImage>());
                image->base = base;
                if (!runtime.tableMemory->readPage(runtime.userContext, base, image->words.data())) return;
            }
        }
        std::size_t at = 0;
        std::size_t page = 0;
        for (const auto& [address, count] : memo.plan.runs) {
            for (std::uint64_t index = 0; index < count; ++index, ++at) {
                const auto dword = address + index * sizeof(std::uint32_t);
                const auto base = dword & ~(PageBytes - 1u);
                if (images[page]->base != base) {
                    const auto found = std::lower_bound(images.begin(), images.end(), base, [](const std::shared_ptr<TablePageImage>& image, std::uint64_t value) { return image->base < value; });
                    if (found == images.end() || (*found)->base != base) return;
                    page = static_cast<std::size_t>(found - images.begin());
                }
                const auto slot = static_cast<std::size_t>((dword - base) / sizeof(std::uint32_t));
                if (at >= words.size() || images[page]->words[slot] != words[at]) return;
                images[page]->mask[slot / 64u] |= std::uint64_t{1} << (slot % 64u);
            }
        }
        memo.spans = std::move(spans);
        memo.images.assign(images.begin(), images.end());
        memo.allRead = std::all_of(memo.plan.states.begin(), memo.plan.states.end(), [](RecordState state) { return state == RecordState::Read; });
        memo.generation = generation;
        tableCounters().memoArmed.fetch_add(1, std::memory_order_relaxed);
    }

    // An armed memo's column without its reads: every span unchanged since the generation (after
    // what a read would flush, with no recorded work writing it), its pages recorded as the full
    // path's reads would have recorded them. False disarms the memo (the full path follows).
    bool SkipReads(ColumnMemo& memo) const {
        const auto& memory = *runtime.tableMemory;
        for (const auto& [begin, end] : memo.spans) {
            if (!memory.unchanged(runtime.userContext, begin, end - begin, memo.generation)) {
                memo.Disarm();
                return false;
            }
        }
        for (const auto& image : memo.images) {
            if (!memory.replayPage(runtime.userContext, image->base, image->words.data(), image->mask.data(), image)) {
                memo.Disarm();
                return false;
            }
        }
        return true;
    }

    // APS5_VERIFY_TABLE_MEMO=1: the word-by-word walk again (its reads hit the dwords already
    // fetched), compared with what the planned walk produced.
    void Verify(const TableColumn& column, const ColumnRecords& records, const ColumnPlan& plan, const std::vector<RecordState>& statesIn) {
        auto& counters = tableCounters();
        counters.verified.fetch_add(1, std::memory_order_relaxed);
        std::vector<RecordState> states = statesIn;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> readRanges;
        std::vector<std::uint64_t> reads;
        std::uint64_t low = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t high = 0;
        const auto end = records.base + records.size;
        std::uint64_t mismatches = 0;
        std::string first;
        const auto mismatch = [&](const std::string& what) {
            if (mismatches++ == 0u) first = what;
        };
        for (std::uint32_t record = 0; record < records.keys; record++) {
            if (states[record] != RecordState::Read) continue;
            DescriptorValue value;
            value.dwordCount = column.sampler ? 4u : 8u;
            const auto offset = column.addend + record * column.stride;
            for (std::uint32_t dword = 0; dword < column.dwordCount; dword++) {
                std::optional<std::uint64_t> address;
                if (column.address) {
                    address = ImageTableAbi::ScalarAddressDword(records.base, offset, column.offset + dword * 4u);
                    if (address.has_value() && (*address & 3u) != 0u) address.reset();
                } else if (const auto position = ImageTableAbi::ScalarBufferDword(offset, column.offset + dword * 4u, records.size); position.has_value()) {
                    address = records.base + *position;
                } else {
                    continue;
                }
                if (!address.has_value() || !(column.address ? AddressAccessible(*address) : PageAccessible(*address, records.base, end))) {
                    states[record] = RecordState::Unmapped;
                    break;
                }
                value.dwords[dword] = Read(*address);
                reads.push_back(*address);
                if (column.address) {
                    if (!readRanges.empty() && *address - readRanges.back().first == readRanges.back().second) {
                        readRanges.back().second += sizeof(std::uint32_t);
                    } else {
                        readRanges.emplace_back(*address, sizeof(std::uint32_t));
                    }
                }
                low = std::min(low, *address);
                high = std::max(high, *address + sizeof(std::uint32_t));
            }
            if (states[record] == RecordState::Read && records.states[record] == RecordState::Read && !(tables.words.at(records.words[record]) == value)) mismatch("record " + std::to_string(record) + " descriptor");
        }
        if (states != records.states) mismatch("record states");
        std::sort(readRanges.begin(), readRanges.end());
        std::size_t ranges = 0;
        for (const auto& range : readRanges) {
            if (ranges != 0u && range.first - readRanges[ranges - 1u].first <= readRanges[ranges - 1u].second) {
                auto& previous = readRanges[ranges - 1u];
                previous.second = std::max(previous.second, range.first - previous.first + range.second);
            } else {
                readRanges[ranges++] = range;
            }
        }
        readRanges.resize(ranges);
        if (readRanges != records.readRanges) mismatch("read ranges");
        if (low != plan.low || high != plan.high) mismatch("bounds");
        std::size_t at = 0;
        bool sameReads = true;
        for (const auto& [address, count] : plan.runs) {
            for (std::uint64_t word = 0; word < count && sameReads; word++, at++) sameReads = at < reads.size() && reads[at] == address + word * sizeof(std::uint32_t);
        }
        if (!sameReads || at != reads.size()) mismatch("read order");
        if (mismatches != 0u) {
            counters.verifyMismatches.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<std::uint32_t> reports{0};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 20u) std::fprintf(stderr, "[image-table] verify: table at 0x%llx (%s, %u keys) differs from the word-by-word walk in %llu places, first: %s\n", static_cast<unsigned long long>(records.base), column.sampler ? "sampler" : "image", records.keys, static_cast<unsigned long long>(mismatches), first.c_str());
        }
    }

    void WalkRecords(const TableColumn& column, ColumnRecords& records) {
        if (column.dwordCount > 8u) throw std::runtime_error("image table column reads more than 8 dwords per record");
        auto& counters = tableCounters();
        auto& memos = columnMemos();
        ColumnMemo* memo = nullptr;
        if (TableMemo()) {
            for (auto& entry : memos.entries) {
                if (entry.Matches(column, records)) {
                    memo = &entry;
                    break;
                }
            }
        }
        // APS5_NO_TABLE_SKIP=1 (ShaderMemory gives no SrtTableMemory): every column reads.
        const bool skips = TableMemo() && runtime.tableMemory != nullptr;
        const bool profiled = MaterializeProfiled();
        const auto since = [](std::chrono::steady_clock::time_point started) { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()); };
        auto started = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool skipped = skips && memo != nullptr && memo->generation != 0 && SkipReads(*memo);
        if (profiled && skips && memo != nullptr && memo->generation != 0) {
            counters.proofNanoseconds.fetch_add(since(started), std::memory_order_relaxed);
            started = std::chrono::steady_clock::now();
        }
        const auto width = column.sampler ? 4u : 8u;
        if (skipped) {
            auto statesIn = std::move(records.states);
            records.states = memo->plan.states;
            records.readRanges = memo->plan.readRanges;
            if (column.address && memo->plan.low < memo->plan.high) tables.ranges.emplace_back(memo->plan.low, memo->plan.high - memo->plan.low);
            // APS5_NO_TABLE_OUTPUT_REUSE=1: the lookups of every skipped column (Reuse).
            static const bool outputReuse = std::getenv("APS5_NO_TABLE_OUTPUT_REUSE") == nullptr;
            if (outputReuse && widths.at(width) == 0u && memo->codeBase == tables.words.size() && memo->codes.size() == records.keys) {
                // What Reuse's bulk append gives at the same position: the descriptors appended in
                // first-occurrence order, each record's code its local index past codeBase.
                tables.words.insert(tables.words.end(), memo->distinct.begin(), memo->distinct.end());
                widths.at(width) += static_cast<std::uint32_t>(memo->distinct.size());
                records.words = memo->codes;
                records.allRead = memo->allRead;
                records.pure = memo;
                counters.outputsReused.fetch_add(1, std::memory_order_relaxed);
            } else {
                ++memo->version;
                const bool bulk = widths.at(width) == 0u;
                const auto first = tables.words.size();
                Reuse(*memo, records);
                records.allRead = memo->allRead;
                if (bulk) {
                    memo->codes = records.words;
                    memo->codeBase = first;
                }
            }
            if (profiled) counters.outputNanoseconds.fetch_add(since(started), std::memory_order_relaxed);
            counters.memoHits.fetch_add(1, std::memory_order_relaxed);
            counters.memoSkipped.fetch_add(1, std::memory_order_relaxed);
            memo->used = ++memos.clock;
            if (TableMemoVerified()) Verify(column, records, memo->plan, statesIn);
            return;
        }
        // A memo that can be armed collects its generation before the plan's accessibility checks
        // fetch the column's pages, so what this capture reads is no older than the generation.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> armSpans;
        std::uint64_t armGeneration = 0;
        if (skips && memo != nullptr && memo->valued) {
            armSpans = SkipSpans(memo->plan);
            if (!armSpans.empty()) armGeneration = CollectSpans(armSpans);
        }
        ColumnPlan fresh;
        const bool replayed = memo != nullptr && Replay(memo->plan, column, records);
        if (!replayed) {
            fresh = PlanColumn(column, records);
            counters.memoPlanned.fetch_add(1, std::memory_order_relaxed);
        }
        const auto& plan = replayed ? memo->plan : fresh;
        std::vector<std::uint32_t> words;
        ReadRuns(plan, words);
        auto statesIn = std::move(records.states);
        records.states = plan.states;
        records.readRanges = plan.readRanges;
        if (column.address && plan.low < plan.high) tables.ranges.emplace_back(plan.low, plan.high - plan.low);
        if (replayed && memo->valued && memo->words == words) {
            const bool bulk = widths.at(width) == 0u;
            const auto first = tables.words.size();
            Reuse(*memo, records);
            counters.memoHits.fetch_add(1, std::memory_order_relaxed);
            if (armGeneration != 0) Arm(*memo, words, std::move(armSpans), armGeneration);
            // After Arm (which starts from a disarmed memo): the codes a later skip takes whole.
            if (bulk) {
                memo->codes = records.words;
                memo->codeBase = first;
            }
        } else {
            if (memo != nullptr) memo->Disarm();
            if (replayed) counters.memoChanged.fetch_add(1, std::memory_order_relaxed);
            std::vector<DescriptorValue> distinct;
            std::vector<std::uint32_t> local;
            Assemble(column, plan, words, records, distinct, local);
            if (TableMemo()) {
                if (memo == nullptr) {
                    if (memos.entries.size() < ColumnMemoEntries) {
                        memo = &memos.entries.emplace_back();
                    } else {
                        memo = &*std::min_element(memos.entries.begin(), memos.entries.end(), [](const ColumnMemo& a, const ColumnMemo& b) { return a.used < b.used; });
                    }
                    memo->address = column.address;
                    memo->sampler = column.sampler;
                    memo->dwordCount = column.dwordCount;
                    memo->stride = column.stride;
                    memo->addend = column.addend;
                    memo->offset = column.offset;
                    memo->keys = records.keys;
                    memo->base = records.base;
                    memo->size = records.size;
                    memo->statesIn = statesIn;
                    memo->Disarm();
                }
                if (!replayed) memo->plan = std::move(fresh);
                memo->valued = memo->plan.replayable;
                memo->words = std::move(words);
                memo->distinct = std::move(distinct);
                memo->local = std::move(local);
            }
        }
        if (memo != nullptr) {
            memo->used = ++memos.clock;
            ++memo->version;
        }
        if (profiled) counters.walkNanoseconds.fetch_add(since(started), std::memory_order_relaxed);
        if (TableMemoVerified()) Verify(column, records, memo != nullptr ? memo->plan : fresh, statesIn);
    }

    ColumnRecords& Column(std::uint32_t source) {
        if (const auto found = columns.find(source); found != columns.end()) {
            return found->second;
        }
        auto& records = columns[source];
        const auto& column = *plan.descriptorSources.at(source).tableColumn;
        const auto setupStarted = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool active = source >= activeSources.size() || activeSources[source] != 0u;
        if (!active) {
            records.fault = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::Inactive));
            return records;
        }
        if (!TableInputs(column, records)) return records;
        if (!column.address && records.size != 0u) tables.ranges.emplace_back(records.base, records.size);
        records.states.assign(records.keys, RecordState::Read);
        if (column.keyDomain.has_value() && records.keys != 0u) {
            Narrow(column, records);
        }
        records.words.assign(records.keys, 0u);
        if (MaterializeProfiled()) tableCounters().setupNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - setupStarted).count()), std::memory_order_relaxed);
        if (records.keys != 0u) WalkRecords(column, records);
        return records;
    }

    void Snapshot(std::uint32_t source, std::uint32_t table) {
        const auto& column = *plan.descriptorSources.at(source).tableColumn;
        const auto& records = Column(source);
        auto& out = tables.tables.at(table);
        out.base = records.base;
        out.size = records.size;
        out.records = records.records;
        out.keys = records.keys;
        out.fault = records.fault;
        out.outside = records.outside;
        const auto codesStarted = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        out.readRanges = records.readRanges;
        TableTrace trace{records.base, records.size, records.records, records.keys, 0u, records.narrowed};
        auto& counters = tableCounters();
        if (records.fault != 0u) counters.poison[records.fault & ~ImageTableAbi::PoisonFlag].fetch_add(1, std::memory_order_relaxed);
        // Every record read (a skipped column whose plan has no other state): the codes are the words.
        const bool whole = records.allRead && records.words.size() == records.keys;
        if (whole) out.codes = records.words;
        else out.codes.assign(records.keys, 0u);
        for (std::uint32_t key = whole ? records.keys : 0u; key < records.keys; key++) {
            switch (records.states[key]) {
                case RecordState::Read:
                    out.codes[key] = records.words[key];
                    continue;
                case RecordState::OutsideDomain:
                    out.codes[key] = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::OutsideDomain));
                    break;
                case RecordState::Unmapped:
                    out.codes[key] = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::Unmapped));
                    counters.poison[static_cast<std::size_t>(PoisonReason::Unmapped)].fetch_add(1, std::memory_order_relaxed);
                    break;
            }
            trace.poison++;
        }
        counters.keys.fetch_add(records.keys, std::memory_order_relaxed);
        if (MaterializeProfiled()) counters.codesNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - codesStarted).count()), std::memory_order_relaxed);
        if (TableTraced()) traceTable(plan, table, column, trace);
    }

    const IrResourcePlan& plan;
    const SrtRuntime& runtime;
    SrtWalker& walker;
    const std::vector<std::uint8_t>& activeSources;
    ImageTableSnapshot& tables;
    std::map<std::uint32_t, ColumnRecords> columns;
    struct WordsKeyHash {
        std::size_t operator()(const std::pair<std::array<std::uint32_t, 8>, std::uint32_t>& key) const noexcept {
            std::uint64_t hash = 14695981039346656037ull ^ key.second;
            for (const auto word : key.first) hash = (hash ^ word) * 1099511628211ull;
            return static_cast<std::size_t>(hash);
        }
    };
    std::unordered_map<std::pair<std::array<std::uint32_t, 8>, std::uint32_t>, std::uint32_t, WordsKeyHash> wordsIndex;
    std::size_t indexed = 0;
    std::array<std::uint32_t, 9> widths{};
    std::uint64_t lastAccessPage = std::numeric_limits<std::uint64_t>::max();
    bool lastAccessible = false;
    std::unordered_map<std::uint64_t, bool> pages;
    std::unordered_map<std::uint64_t, bool> wholePages;
};

void materializeSnapshot(const IrResourcePlan& plan, const SrtRuntime& runtime, SrtWalker& walker, ResourceSnapshot& snapshot, std::vector<std::uint8_t>& activeSources) {
    snapshot = ResourceSnapshot{};
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        const auto words = plan.uniformFill.fill.words;
        if (words == 0u || words > plan.uniformFill.values.size()) {
            throw std::runtime_error("uniform fill plan has an invalid word count");
        }
        std::array<std::uint32_t, 4> stored{};
        walker.EvaluateUniformValues(plan, std::span(plan.uniformFill.values).first(words), runtime, std::span(stored).first(words));
        for (std::uint32_t i = 1; i < words; i++) {
            if (stored[i] != stored[0]) {
                throw std::runtime_error("uniform fill values diverge at runtime");
            }
        }
        snapshot.uniformFill = plan.uniformFill.fill;
        snapshot.uniformFill.value = stored[0];
    }
    if (runtime.userData.size() < plan.userDataCount) {
        throw std::runtime_error("runtime user data is smaller than the shader user data count");
    }
    snapshot.userData.assign(runtime.userData.begin(), runtime.userData.begin() + plan.userDataCount);

    std::vector<DescriptorValue> values;
    walker.EvaluateRuntimeSources(plan, plan.materializationSources, runtime, values, snapshot.flattenedSrt, plan.cleanFlatSlots, activeSources, &snapshot.srtPoison, &snapshot.nullRootReads);

    std::size_t cursor = 0;
    if (values.size() < plan.info.buffers.size()) {
        throw std::runtime_error("materialization sources are missing buffer descriptors");
    }
    snapshot.buffers.assign(values.begin(), values.begin() + plan.info.buffers.size());
    cursor += plan.info.buffers.size();

    snapshot.images.resize(plan.info.images.size());
    for (std::uint32_t i = 0; i < plan.info.images.size(); i++) {
        const auto& image = plan.info.images[i];
        if (image.source >= plan.descriptorSources.size()) {
            throw std::runtime_error("image resource references an unknown descriptor source");
        }
        if (image.table != NoTable) {
            snapshot.images[i].dwordCount = 8u;
            continue;
        }
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing image descriptors");
        }
        auto descriptor = values[cursor];
        cursor++;
        if (descriptor.dwordCount != 8u) {
            throw std::runtime_error("image descriptor has an invalid width");
        }
        const auto activity = plan.descriptorSources.size() + i;
        if (activity < activeSources.size() && activeSources[activity] == 0u) {
            descriptor = DescriptorValue{};
            descriptor.dwordCount = 8u;
        }
        if (!validImageDescriptor(descriptor, image.r128) && !nullImageDescriptor(descriptor)) {
            throw std::runtime_error("runtime image descriptor is invalid");
        }
        snapshot.images[i] = descriptor;
    }

    snapshot.samplers.resize(plan.info.samplers.size());
    for (std::uint32_t i = 0; i < plan.info.samplers.size(); i++) {
        if (plan.info.samplers[i].table != NoTable) {
            snapshot.samplers[i].dwordCount = 4u;
            continue;
        }
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing sampler descriptors");
        }
        snapshot.samplers[i] = values[cursor];
        cursor++;
    }
}

std::uint32_t colorCompareReference(IrBufferFormat format) {
    switch (format) {
    case IrBufferFormat::Format16_16UNorm:
    case IrBufferFormat::Format11_11_10UNorm: case IrBufferFormat::Format10_11_11UNorm: case IrBufferFormat::Format2_10_10_10UNorm:
    case IrBufferFormat::Format10_10_10_2UNorm: case IrBufferFormat::Format16_16_16_16UNorm:
        return EmulatedCompare::ReferenceUnorm;
    case IrBufferFormat::Format8UNorm: case IrBufferFormat::Format8_8UNorm: case IrBufferFormat::Format8_8_8_8UNorm:
    case IrBufferFormat::Format8Srgb: case IrBufferFormat::Format8_8Srgb: case IrBufferFormat::Format8_8_8_8Srgb:
        return EmulatedCompare::ReferenceZero;
    case IrBufferFormat::Format8SNorm: case IrBufferFormat::Format16SNorm: case IrBufferFormat::Format8_8SNorm: case IrBufferFormat::Format16_16SNorm:
    case IrBufferFormat::Format11_11_10SNorm: case IrBufferFormat::Format10_11_11SNorm: case IrBufferFormat::Format2_10_10_10SNorm:
    case IrBufferFormat::Format10_10_10_2SNorm: case IrBufferFormat::Format8_8_8_8SNorm: case IrBufferFormat::Format16_16_16_16SNorm:
        return EmulatedCompare::ReferenceSnorm;
    case IrBufferFormat::Format16Float: case IrBufferFormat::Format16_16Float: case IrBufferFormat::Format11_11_10Float:
    case IrBufferFormat::Format10_11_11Float: case IrBufferFormat::Format32_32Float: case IrBufferFormat::Format16_16_16_16Float:
        return EmulatedCompare::ReferenceFloat;
    default:
        throw std::runtime_error("comparison sampling of a color texture is implemented only for float, unorm and snorm formats (format " + std::to_string(static_cast<std::uint32_t>(format)) + ")");
    }
}

std::uint32_t emulatedCompareState(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint32_t index) {
    const auto& image = info.images[index];
    const auto& descriptor = snapshot.images[index];
    if (!image.depthCompare || descriptor.dwordCount != 8u || nullImageDescriptor(descriptor)) return 0u;
    const auto format = rawImageFormat(descriptor);
    if (format == IrBufferFormat::Format32Float || format == IrBufferFormat::Format16UNorm || IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3])) return 0u;
    if (image.table != NoTable || (image.emulatedCompare & EmulatedCompare::Unsupported) != 0u) throw std::runtime_error("unsupported color comparison image instructions");
    if ((image.emulatedCompare & EmulatedCompare::RequiresSingleLevel) != 0u && ((descriptor.dwords[3] >> 12u) & 0xfu) != ((descriptor.dwords[3] >> 16u) & 0xfu)) throw std::runtime_error("color comparison requires a single mip level");
    const auto reference = colorCompareReference(format);
    const auto type = rawImageType(descriptor);
    if (type != ImageType::Color2D && type != ImageType::Color2DArray && type != ImageType::Cube) throw std::runtime_error("comparison sampling of a color texture is implemented only for 2D, 2D array and cube views");
    bool redChannel = false;
    bool compareChannel = false;
    for (std::uint32_t channel = 0u; channel < 4u; ++channel) {
        const auto select = (descriptorImageSwizzle(descriptor) >> (channel * 3u)) & 0x7u;
        if (select == 2u || select == 3u) throw std::runtime_error("comparison sampling of a color texture is implemented only when every view channel is a constant or a color channel");
        redChannel = redChannel || select == 4u;
        compareChannel = compareChannel || select >= 4u;
    }
    if (compareChannel && !redChannel) throw std::runtime_error("comparison sampling of a color texture is implemented only when a view channel selects red");
    std::optional<std::uint32_t> samplerState;
    for (const auto& pair : info.sampledPairs) {
        if (pair.image != index) continue;
        if (pair.sampler >= info.samplers.size() || info.samplers[pair.sampler].table != NoTable) throw std::runtime_error("comparison sampling of a color texture through a sampler table is not implemented");
        if (pair.sampler >= snapshot.samplers.size() || snapshot.samplers[pair.sampler].dwordCount != 4u) throw std::runtime_error("comparison sampling of a color texture has no sampler descriptor");
        const auto& words = snapshot.samplers[pair.sampler].dwords;
        const auto clampX = words[0] & 0x7u;
        const auto clampY = (words[0] >> 3u) & 0x7u;
        const auto function = (words[0] >> 12u) & 0x7u;
        const bool unnormalized = ((words[0] >> 15u) & 0x1u) != 0u;
        if (((words[0] >> 29u) & 0x3u) != 0u) throw std::runtime_error("comparison sampling of a color texture through a min or max reduction sampler is not implemented");
        const auto magFilter = (words[2] >> 20u) & 0x3u;
        const auto minFilter = (words[2] >> 22u) & 0x3u;
        const bool constantZero = reference == EmulatedCompare::ReferenceZero;
        if (type == ImageType::Cube && magFilter == 1u && ((words[0] >> 28u) & 1u) == 0u && !constantZero) {
            throw std::runtime_error("bilinear cube comparison requires DISABLE_CUBE_WRAP to avoid seamless face filtering");
        }
        const auto addressMode = [](std::uint32_t clamp) {
            if (clamp == 0u) return EmulatedCompare::AddressWrap;
            if (clamp == 2u) return EmulatedCompare::AddressEdge;
            if (clamp == 6u) return EmulatedCompare::AddressBorder;
            throw std::runtime_error("comparison sampling of a color texture is implemented only with wrap, clamp-to-edge or clamp-to-border addressing");
        };
        const auto addressX = addressMode(clampX);
        const auto addressY = addressMode(clampY);
        const auto borderType = (words[3] >> 30u) & 0x3u;
        const bool border = addressX == EmulatedCompare::AddressBorder || addressY == EmulatedCompare::AddressBorder;
        if (border && borderType == 3u) throw std::runtime_error("comparison sampling of a color texture with a border color table is not implemented");
        if (!constantZero && (magFilter != minFilter || magFilter > 1u)) throw std::runtime_error("comparison sampling of a color texture is implemented only with equal point or bilinear minification and magnification filters");
        if (unnormalized) throw std::runtime_error("comparison sampling of a color texture does not implement unnormalized coordinates");
        const auto state = EmulatedCompare::Enabled | (function << EmulatedCompare::FunctionShift) | (magFilter == 1u ? EmulatedCompare::Linear : 0u)
            | (addressX << EmulatedCompare::ClampXShift) | (addressY << EmulatedCompare::ClampYShift) | (border && borderType == 2u ? EmulatedCompare::BorderWhite : 0u);
        if (samplerState.has_value() && *samplerState != state) throw std::runtime_error("comparison sampling of a color texture through samplers that disagree is not implemented");
        samplerState = state;
    }
    if (!samplerState.has_value()) throw std::runtime_error("comparison sampling of a color texture has no paired sampler");
    const bool singleLevel = ((descriptor.dwords[3] >> 12u) & 0xfu) == ((descriptor.dwords[3] >> 16u) & 0xfu);
    return *samplerState | (reference << EmulatedCompare::ReferenceShift) | (singleLevel ? EmulatedCompare::SingleLevel : 0u);
}

void materializeTables(const IrResourcePlan& plan, ResourceSnapshot& snapshot) {
    for (std::uint32_t i = 0u; i < plan.info.buffers.size(); ++i) {
        const auto decoded = decodeBufferDescriptor(snapshot.buffers.at(i));
        if (decoded.Type() != 0u) throw std::runtime_error("buffer descriptor uses an unsupported type");
        if (plan.stage != IrShaderStage::Compute && decoded.AddTid()) throw std::runtime_error("buffer ADD_TID is only valid for compute shaders");
    }
    if (snapshot.flattenedSrt.size() != plan.srtReads.size()) throw std::runtime_error("runtime SRT size differs from the static interface");
    for (std::uint32_t i = 0u; i < plan.info.images.size(); ++i) {
        const auto& image = plan.info.images[i];
        if (image.table != NoTable) continue;
        static_cast<void>(ResourceMaterializer::RuntimeImageMode(image, snapshot.images.at(i), plan.info.runtimeImageModes.at(i)));
    }
}

void materializeSrtGuards(const IrResourcePlan& plan, ResourceSnapshot& snapshot) {
    const auto& guarded = plan.guardedSrtSlots;
    const auto flags = snapshot.flattenedSrt.size();
    snapshot.flattenedSrt.resize(flags + guarded.size(), 0u);
    for (std::uint32_t record = 0u; record < snapshot.srtPoison.size(); ++record) {
        const auto& poison = snapshot.srtPoison[record];
        const auto found = std::lower_bound(guarded.begin(), guarded.end(), poison.slot);
        if (found == guarded.end() || *found != poison.slot) throw std::runtime_error("SRT read at flat offset " + std::to_string(poison.slot) + " reads inaccessible memory but has no guard");
        snapshot.flattenedSrt[flags + static_cast<std::size_t>(found - guarded.begin())] = record + 1u;
        snapshot.flattenedSrt.insert(snapshot.flattenedSrt.end(), {poison.pc, static_cast<std::uint32_t>(poison.address), static_cast<std::uint32_t>(poison.address >> 32u)});
    }
}

struct DecodeMemoKey {
    std::uint64_t artifact = 0;
    std::uint32_t resource = 0;
    std::array<std::uint32_t, 8> words{};

    bool operator==(const DecodeMemoKey& other) const = default;
};

struct DecodeMemoHash {
    std::size_t operator()(const DecodeMemoKey& key) const {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        const auto mix = [&](std::uint64_t value) {
            hash ^= value;
            hash *= 0x100000001b3ull;
        };
        mix(key.artifact);
        mix(key.resource);
        for (const auto word : key.words) mix(word);
        return static_cast<std::size_t>(hash);
    }
};

struct DecodeMemo {
    std::mutex mutex;
    std::unordered_map<DecodeMemoKey, std::uint32_t, DecodeMemoHash> entries;
};

DecodeMemo& decodeMemo() {
    static DecodeMemo memo;
    return memo;
}

constexpr std::uint32_t DecodeMemoEntries = 1u << 20u;
constexpr std::uint32_t DecodeFailed = 0x80000000u;

std::uint32_t decodeTableEntry(const ImageResource& image, std::span<const ImageResource> modes, const DescriptorValue& words, std::uint64_t memoKey, std::uint32_t resource) {
    DecodeMemoKey key{memoKey, resource, words.dwords};
    if (memoKey != 0u) {
        auto& memo = decodeMemo();
        std::lock_guard lock(memo.mutex);
        if (const auto found = memo.entries.find(key); found != memo.entries.end()) return found->second;
    }
    const auto match = ResourceMaterializer::TryRuntimeImageMode(image, words, modes);
    const auto value = match.mode.has_value() ? *match.mode : DecodeFailed | static_cast<std::uint32_t>(match.reason);
    if (memoKey != 0u) {
        auto& memo = decodeMemo();
        std::lock_guard lock(memo.mutex);
        if (memo.entries.size() >= DecodeMemoEntries) memo.entries.clear();
        memo.entries.emplace(key, value);
    }
    return value;
}

}

ResolvedImageTables ResourceMaterializer::ResolveImageTables(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint64_t memoKey) {
    namespace Abi = ImageTableAbi;
    ResolvedImageTables out;
    std::uint32_t count = 0;
    for (const auto& image : info.images) {
        if (image.table != NoTable) count = std::max(count, image.table + 1u);
    }
    for (const auto& sampler : info.samplers) {
        if (sampler.table != NoTable) count = std::max(count, sampler.table + 1u);
    }
    if (count == 0u) return out;
    const auto& tables = snapshot.tables;
    if (tables.tables.size() != count) throw std::runtime_error("image table snapshot disagrees with the static tables");
    if (info.runtimeImageModes.size() != info.images.size()) throw std::runtime_error("prepared runtime image modes are missing");
    out.map.assign(Abi::HeaderWords + Abi::TableWords * count, 0u);
    out.map[Abi::HeaderVersion] = Abi::Version;
    out.poison.push_back({{}, 0u, NoTable, PoisonReason::OutsideSnapshot});
    out.poison.push_back({{}, 0u, NoTable, PoisonReason::OutsideDomain});
    std::map<std::tuple<std::uint32_t, PoisonReason, std::array<std::uint32_t, 8>, std::uint32_t>, std::uint32_t> poisonIndex;
    std::set<std::uint32_t> expected;
    const auto poison = [&](std::uint32_t resource, PoisonReason reason, const DescriptorValue* words) {
        const auto value = words != nullptr ? *words : DescriptorValue{};
        const auto [found, inserted] = poisonIndex.try_emplace({resource, reason, value.dwords, value.dwordCount}, static_cast<std::uint32_t>(out.poison.size()));
        if (inserted) out.poison.push_back({value.dwords, value.dwordCount, resource, reason});
        expected.insert(found->second);
        return Abi::PoisonCode(found->second);
    };
    std::map<std::tuple<std::array<std::uint32_t, 8>, RdnaImageDimension, bool, DescriptorBindingKind>, std::uint32_t> elementIndex;
    std::vector<std::uint32_t> directSamplers(info.images.size(), 0u);
    std::vector<bool> reducingSampler(info.images.size(), false);
    for (const auto& pair : info.sampledPairs) {
        const auto& sampler = info.samplers.at(pair.sampler);
        if (sampler.table != NoTable) continue;
        directSamplers.at(pair.image) |= 1u << pair.sampler;
        const auto& words = snapshot.samplers.at(pair.sampler).dwords;
        if (reducesBetweenTexels(words[0], words[2])) reducingSampler.at(pair.image) = true;
    }
    const auto imageRecord = [&](std::uint32_t resource, const DescriptorValue& words) {
        const auto& image = info.images[resource];
        const auto& modes = info.runtimeImageModes[resource];
        std::array<std::uint32_t, Abi::ImageRecordWords> record{0u, words.dwords[3], 0u};
        const auto decoded = decodeTableEntry(image, modes, words, memoKey, resource);
        if ((decoded & DecodeFailed) != 0u) {
            record[Abi::ImageRecordFlags] = poison(resource, static_cast<PoisonReason>(decoded & ~DecodeFailed), &words);
        } else {
            const auto& mode = modes.at(decoded);
            if (mode.dimension == RdnaImageDimension::Dim2DMsaa || mode.dimension == RdnaImageDimension::Dim2DMsaaArray) {
                record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::Multisampled, &words);
            } else if (requiresPointSampler(mode) && reducingSampler[resource]) {
                record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::Reduction, &words);
            } else {
                auto element = words.dwords;
                if (mode.conversionFormat != IrBufferFormat::Invalid || mode.depthBits) element[3] = (element[3] & ~0xfffu) | ShaderImageIdentitySwizzle;
                const auto kind = DescriptorBindingForImage(mode);
                const auto [found, inserted] = elementIndex.try_emplace({element, mode.dimension, mode.depthCompare, kind}, static_cast<std::uint32_t>(out.elements.size()));
                if (inserted && out.elements.size() >= Abi::ImageElementBudget) {
                    elementIndex.erase(found);
                    record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::Budget, &words);
                } else {
                    if (inserted) out.elements.push_back({element, mode.dimension, mode.depthCompare, kind, 0u});
                    out.elements[found->second].samplers |= directSamplers[resource];
                    record[Abi::ImageRecordFlags] = decoded << Abi::ModeShift;
                    record[Abi::ImageRecordElement] = found->second;
                }
            }
        }
        const auto offset = static_cast<std::uint32_t>(out.map.size());
        out.map.insert(out.map.end(), record.begin(), record.end());
        return offset;
    };
    const auto samplerRecord = [&](std::uint32_t resource, const DescriptorValue& words) {
        std::array<std::uint32_t, Abi::SamplerRecordWords> record{0u, 0u};
        if ((words.dwords[0] & (1u << 15u)) != 0u) {
            record[Abi::SamplerRecordFlags] = poison(resource, PoisonReason::Unnormalized, &words);
        } else if (((words.dwords[0] >> 29u) & 3u) != 0u) {
            record[Abi::SamplerRecordFlags] = poison(resource, PoisonReason::Reduction, &words);
        } else if (out.samplers.size() + Abi::SamplerEntryElements > Abi::SamplerElementBudget) {
            record[Abi::SamplerRecordFlags] = poison(resource, PoisonReason::Budget, &words);
        } else {
            record[Abi::SamplerRecordElement] = static_cast<std::uint32_t>(out.samplers.size());
            for (std::uint32_t variant = 0; variant < 2u; variant++) {
                const bool compare = variant == 1u;
                for (std::uint32_t point = 0; point < 2u; point++) {
                    ImageTableSamplerElement element;
                    std::copy_n(words.dwords.begin(), 4u, element.words.begin());
                    element.compare = compare;
                    element.point = point != 0u;
                    out.samplers.push_back(element);
                }
            }
        }
        const auto offset = static_cast<std::uint32_t>(out.map.size());
        out.map.insert(out.map.end(), record.begin(), record.end());
        return offset;
    };
    std::vector<bool> outside(count, false);
    const auto fill = [&](std::uint32_t resource, std::uint32_t table, bool sampler) {
        const auto& column = tables.tables.at(table);
        const auto header = Abi::TableHeader(table);
        out.map[header + Abi::TableKeyCount] = column.keys;
        out.map[header + Abi::TableSizeLow] = static_cast<std::uint32_t>(column.size);
        out.map[header + Abi::TableSizeHigh] = static_cast<std::uint32_t>(column.size >> 32u);
        out.map[header + Abi::TableBaseLow] = static_cast<std::uint32_t>(column.base);
        out.map[header + Abi::TableBaseHigh] = static_cast<std::uint32_t>(column.base >> 32u);
        if (column.codes.size() != column.keys) throw std::runtime_error("image table snapshot has a truncated map");
        if (column.fault != 0u) {
            out.map[header + Abi::TableFault] = poison(resource, static_cast<PoisonReason>(column.fault & ~Abi::PoisonFlag), nullptr);
        } else if (!sampler && (info.images[resource].tableOperation == TableOperation::Unsupported || info.runtimeImageModes[resource].empty())) {
            out.map[header + Abi::TableFault] = poison(resource, PoisonReason::UnsupportedOperation, nullptr);
        }
        outside[table] = column.outside;
        const auto mapStart = static_cast<std::uint32_t>(out.map.size());
        out.map[header + Abi::TableMapStart] = mapStart;
        out.map.resize(out.map.size() + column.keys, 0u);
        std::unordered_map<std::uint32_t, std::uint32_t> records;
        for (std::uint32_t key = 0; key < column.keys; key++) {
            const auto code = column.codes[key];
            std::uint32_t mapped = Abi::NullCode;
            if ((code & Abi::PoisonFlag) != 0u) {
                const auto reason = static_cast<PoisonReason>(code & ~Abi::PoisonFlag);
                mapped = reason == PoisonReason::OutsideDomain ? Abi::PoisonCode(Abi::OutsideDomainPoison) : poison(resource, reason, nullptr);
            } else if (out.map[header + Abi::TableFault] == 0u) {
                const auto& words = tables.words.at(code);
                if (sampler || !nullImageDescriptor(words)) {
                    const auto [found, inserted] = records.try_emplace(code, 0u);
                    if (inserted) found->second = sampler ? samplerRecord(resource, words) : imageRecord(resource, words);
                    mapped = found->second;
                }
            }
            out.map[mapStart + key] = mapped;
        }
        if (sampler && out.map[header + Abi::TableFault] == 0u) {
            DescriptorValue zero;
            zero.dwordCount = 4u;
            out.map[header + Abi::TableOutsideCode] = samplerRecord(resource, zero);
        }
    };
    for (std::uint32_t resource = 0; resource < info.images.size(); resource++) {
        if (info.images[resource].table != NoTable) fill(resource, info.images[resource].table, false);
    }
    for (std::uint32_t resource = 0; resource < info.samplers.size(); resource++) {
        if (info.samplers[resource].table != NoTable) fill(resource, info.samplers[resource].table, true);
    }
    out.map[Abi::HeaderElementCount] = static_cast<std::uint32_t>(out.elements.size());
    out.map[Abi::HeaderSamplerCount] = static_cast<std::uint32_t>(out.samplers.size());
    out.map[Abi::HeaderPoisonCount] = static_cast<std::uint32_t>(out.poison.size());
    out.map[Abi::HeaderTableCount] = count;
    out.map[Abi::HeaderElementFaults] = static_cast<std::uint32_t>(out.map.size());
    out.map.resize(out.map.size() + out.elements.size(), 0u);
    out.map[Abi::HeaderSamplerFaults] = static_cast<std::uint32_t>(out.map.size());
    out.map.resize(out.map.size() + out.samplers.size() / Abi::SamplerEntryElements, 0u);
    out.faults = static_cast<std::uint32_t>(expected.size() + static_cast<std::size_t>(std::count(outside.begin(), outside.end(), true)));
    return out;
}

std::vector<ImageResource> ResourceMaterializer::RuntimeImageModes(const ImageResource& image) {
    const bool table = image.table != NoTable;
    if (table && image.tableOperation == TableOperation::Unsupported) return {};
    std::vector<ImageResource> modes;
    const bool storage = image.resourceClass == ImageResourceClass::Storage;
    const auto append = [&](IrTextureNumericClass numeric, IrBufferFormat conversion, IrBufferFormat packed, bool depth, bool unorm16) {
        auto mode = image;
        mode.numericClass = numeric;
        mode.conversionFormat = conversion;
        mode.packedFormat = packed;
        mode.depthBits = depth;
        mode.depthUnorm16 = unorm16;
        mode.cube = false;
        mode.mipCount = mode.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u;
        mode.shaderSwizzle = ShaderImageIdentitySwizzle;
        if (conversion == IrBufferFormat::Format11_11_10UNorm || conversion == IrBufferFormat::Format10_11_11Float) mode.shaderSwizzle = 0x2acu;
        modes.push_back(mode);
        if (image.dimension == RdnaImageDimension::Dim2D && image.fmaskCompatible && !depth && packed == IrBufferFormat::Invalid && image.byElements == 0u) {
            auto volume = mode;
            volume.dimension = RdnaImageDimension::Dim3D;
            modes.push_back(volume);
        }
        if (image.dimension == RdnaImageDimension::Dim3D && image.flatVolumeCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            for (const auto dimension : {RdnaImageDimension::Dim2D, RdnaImageDimension::Dim2DArray}) {
                auto plane = mode;
                plane.dimension = dimension;
                modes.push_back(plane);
            }
        }
        if (image.dimension == RdnaImageDimension::Dim2D && storage && image.written && !image.read && !image.atomic && image.mipMode != ImageMipMode::DynamicStorage && !depth && packed == IrBufferFormat::Invalid && image.byElements == 0u) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim2D && image.flatLineCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim1D && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto plane = mode;
            plane.dimension = RdnaImageDimension::Dim2D;
            modes.push_back(plane);
        }
        if (image.dimension == RdnaImageDimension::Dim1DArray || image.dimension == RdnaImageDimension::Dim2DArray || image.dimension == RdnaImageDimension::Dim2DMsaaArray) {
            auto plain = mode;
            plain.dimension = image.dimension == RdnaImageDimension::Dim1DArray ? RdnaImageDimension::Dim1D : image.dimension == RdnaImageDimension::Dim2DArray ? RdnaImageDimension::Dim2D : RdnaImageDimension::Dim2DMsaa;
            modes.push_back(plain);
        }
        if (image.dimension == RdnaImageDimension::Dim2DArray) {
            mode.cube = true;
            modes.push_back(mode);
        }
    };
    if (image.atomic) {
        append(IrTextureNumericClass::Uint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
    } else if (image.packed) {
        for (std::uint32_t value = 1u; value <= 77u; ++value) {
            const auto format = static_cast<IrBufferFormat>(value);
            const auto info = GetFormatInfo(format);
            if (info.packedBitfield || info.componentCount == 0u || (storage && info.byteSize == 12u)) continue;
            if (image.byElements != 0u && info.byteSize * 8u * image.byElements > 32u) continue;
            const auto numeric = storage && (format == IrBufferFormat::Format32SInt || format == IrBufferFormat::Format32_32SInt || format == IrBufferFormat::Format32_32_32_32SInt) ? IrTextureNumericClass::Uint : SampledTextureNumericClass(format);
            bool supported = numeric != IrTextureNumericClass::Unsupported && (!storage || numeric != IrTextureNumericClass::Sint);
            for (std::uint32_t component = 0u; component < info.componentCount; ++component) {
                const auto bits = info.componentBits[component];
                const bool exactRead = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint || (info.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (info.type == SpirvFormatComponentType::Float && bits == 32u);
                const bool exactWrite = info.type == SpirvFormatComponentType::Uint || (bits == 32u && (info.type == SpirvFormatComponentType::Sint || info.type == SpirvFormatComponentType::Float));
                supported &= storage ? exactWrite : exactRead;
            }
            if (supported) append(numeric, IrBufferFormat::Invalid, format, false, false);
        }
    } else {
        if ((image.emulatedCompare & EmulatedCompare::NativeOffsetUnsupported) == 0u) append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
        if (!image.depthCompare) {
            append(IrTextureNumericClass::Uint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
            if (!storage) append(IrTextureNumericClass::Sint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
            append(IrTextureNumericClass::Uint, IrBufferFormat::Format11_11_10UInt, IrBufferFormat::Invalid, false, false);
            if (image.srgbDecodeCompatible && image.depthBitsCompatible) {
                append(IrTextureNumericClass::Uint, IrBufferFormat::Format11_11_10UNorm, IrBufferFormat::Invalid, false, false);
                append(IrTextureNumericClass::Uint, IrBufferFormat::Format10_11_11Float, IrBufferFormat::Invalid, false, false);
            }
            if (!storage) {
                if (image.depthBitsCompatible) {
                    append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, true, false);
                    append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, true, true);
                }
                if (image.fmaskCompatible && !table) append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Fmask8_S2_F1, false, false);
            }
        }
    }
    if (storage && image.depthBitsCompatible && !image.packed && !image.atomic64 && ((image.written && !image.read) || image.atomic)) {
        constexpr std::array formats{IrBufferFormat::Format32SInt, IrBufferFormat::Format32_32SInt, IrBufferFormat::Format32_32_32_32SInt, IrBufferFormat::Format16SInt, IrBufferFormat::Format8_8SInt, IrBufferFormat::Format16_16SInt, IrBufferFormat::Format8_8_8_8SInt, IrBufferFormat::Format16_16_16_16SInt};
        for (const auto format : formats) {
            if (image.atomic && format != IrBufferFormat::Format32SInt) continue;
            append(IrTextureNumericClass::Uint, format, IrBufferFormat::Invalid, false, false);
        }
    }
    if (image.depthCompare && !table && (image.emulatedCompare & EmulatedCompare::Unsupported) == 0u && (image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray)) {
        auto mode = image;
        mode.numericClass = IrTextureNumericClass::Float;
        mode.depthCompare = false;
        mode.cube = false;
        mode.emulatedCompare |= EmulatedCompare::Enabled;
        mode.conversionFormat = IrBufferFormat::Invalid;
        mode.packedFormat = IrBufferFormat::Invalid;
        mode.shaderSwizzle = ShaderImageIdentitySwizzle;
        modes.push_back(mode);
        if (image.dimension == RdnaImageDimension::Dim2DArray) {
            auto plain = mode;
            plain.dimension = RdnaImageDimension::Dim2D;
            modes.push_back(plain);
            mode.cube = true;
            modes.push_back(mode);
        }
    }
    if (image.srgbDecodeFormats != 0u && image.srgbDecodeCompatible && !image.depthCompare && !storage && !image.packed) {
        const auto count = modes.size();
        for (std::size_t index = 0; index < count; ++index) {
            const auto& base = modes[index];
            if (base.numericClass != IrTextureNumericClass::Float || base.conversionFormat != IrBufferFormat::Invalid || base.packedFormat != IrBufferFormat::Invalid || base.depthBits) continue;
            auto mode = base;
            mode.srgbDecode = true;
            modes.push_back(mode);
        }
    }
    if (image.constantSwizzleCompatible && !storage && !image.depthCompare && !image.packed && !table) {
        for (const auto numeric : {IrTextureNumericClass::Float, IrTextureNumericClass::Uint, IrTextureNumericClass::Sint}) {
            auto mode = image;
            mode.numericClass = numeric;
            mode.conversionFormat = IrBufferFormat::Invalid;
            mode.packedFormat = IrBufferFormat::Invalid;
            mode.depthBits = false;
            mode.depthUnorm16 = false;
            mode.cube = false;
            mode.mipCount = 1u;
            mode.shaderSwizzle = ShaderImageIdentitySwizzle;
            mode.constantSwizzle = true;
            modes.push_back(mode);
        }
    }
    if (modes.empty() && !table) throw std::runtime_error("image instruction has no supported runtime modes");
    if (image.foldTexelOffsetsOnly && std::ranges::any_of(modes, [](const ImageResource& mode) { return UnnormalizedSampleMismatch(mode) == nullptr; })) {
        std::erase_if(modes, [](const ImageResource& mode) { return UnnormalizedSampleMismatch(mode) != nullptr; });
    }
    return modes;
}

const char* ResourceMaterializer::UnnormalizedSampleMismatch(const ImageResource& image) {
    if (image.table != NoTable) return "samples an image selected at run time";
    if (image.constantSwizzle) return nullptr;
    if ((image.dimension != RdnaImageDimension::Dim1D && image.dimension != RdnaImageDimension::Dim2D) || image.cube) return "samples a 1D-array, 2D-array, 3D, cube or multisampled image";
    if (image.depthCompare) return "is used with depth comparison";
    if (image.conversionFormat != IrBufferFormat::Invalid || image.packed) return "samples an image that needs a format conversion or packed access";
    return nullptr;
}

RuntimeImageModeMatch ResourceMaterializer::TryRuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes) {
    RuntimeImageModeMatch result;
    const auto reject = [&](PoisonReason reason, std::string detail) {
        result.reason = reason;
        result.detail = std::move(detail);
        return result;
    };
    if (descriptor.dwordCount != 8u) throw std::runtime_error("runtime image descriptor must contain eight dwords");
    if (modes.empty()) throw std::runtime_error("prepared runtime image modes are missing");
    if (nullImageDescriptor(descriptor)) {
        result.mode = 0u;
        return result;
    }
    const auto inspected = inspectImageDescriptor(descriptor, image, image.srgbDecodeFormats);
    if (inspected.reason.has_value()) return reject(*inspected.reason, inspected.detail);
    const auto& decoded = inspected.decoded;
    const auto format = rawImageFormat(descriptor);
    const bool emulated = image.depthCompare && format != IrBufferFormat::Format32Float && format != IrBufferFormat::Format16UNorm && !IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3]);
    if (!emulated && (image.emulatedCompare & EmulatedCompare::NativeOffsetUnsupported) != 0u) return reject(PoisonReason::UnsupportedOperation, "native comparison with a nonconstant texel offset requires VK_KHR_maintenance8 and shaderImageGatherExtended");
    if (image.packed && decoded.packedFormat != IrBufferFormat::Invalid) {
        const auto packedFormat = GetFormatInfo(decoded.packedFormat);
        if (packedFormat.packedBitfield) return reject(PoisonReason::InvalidFormat, "runtime packed image accesses a bitfield format");
        const bool storage = image.resourceClass == ImageResourceClass::Storage;
        if (storage && packedFormat.byteSize == 12u) return reject(PoisonReason::InvalidFormat, "runtime packed image uses a format the hardware does not write");
        for (std::uint32_t component = 0u; component < packedFormat.componentCount; ++component) {
            const auto bits = packedFormat.componentBits[component];
            const bool exact = storage ? packedFormat.type == SpirvFormatComponentType::Uint || (bits == 32u && (packedFormat.type == SpirvFormatComponentType::Sint || packedFormat.type == SpirvFormatComponentType::Float)) : packedFormat.type == SpirvFormatComponentType::Uint || packedFormat.type == SpirvFormatComponentType::Sint || (packedFormat.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (packedFormat.type == SpirvFormatComponentType::Float && bits == 32u);
            if (!exact) return reject(PoisonReason::InvalidFormat, storage ? "runtime packed image bits are not reproducible through the view" : "runtime packed image bits are not recoverable from the view");
        }
    }
    if (decoded.mipCount > (image.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u)) return reject(PoisonReason::InvalidFormat, "runtime storage image mip capacity exceeded");
    if ((descriptorImageSwizzle(descriptor) & 06666u) == 0u && !decoded.fmask && !decoded.depthBits && decoded.conversionFormat == IrBufferFormat::Invalid && !decoded.srgbDecode) {
        for (std::uint32_t index = 0u; index < modes.size(); ++index) {
            if (modes[index].constantSwizzle && modes[index].numericClass == decoded.numericClass) {
                result.mode = index;
                return result;
            }
        }
    }
    for (std::uint32_t index = 0u; index < modes.size(); ++index) {
        const auto& mode = modes[index];
        if (mode.constantSwizzle) continue;
        if (((mode.emulatedCompare & EmulatedCompare::Enabled) != 0u) != emulated) continue;
        if (decoded.fmask) {
            if (mode.packedFormat == IrBufferFormat::Fmask8_S2_F1) {
                result.mode = index;
                return result;
            }
            continue;
        }
        if (mode.numericClass == decoded.numericClass && mode.dimension == decoded.dimension && mode.conversionFormat == decoded.conversionFormat && mode.packedFormat == decoded.packedFormat && mode.cube == decoded.cube && mode.depthBits == decoded.depthBits && mode.depthUnorm16 == decoded.depthUnorm16 && mode.srgbDecode == decoded.srgbDecode) {
            result.mode = index;
            return result;
        }
    }
    if (image.dimension == RdnaImageDimension::Dim1D && decoded.dimension != RdnaImageDimension::Dim1D) return reject(PoisonReason::Dimension, "image address has too few coordinate components");
    if (image.foldTexelOffsetsOnly) return reject(PoisonReason::UnsupportedOperation, "image sample with a texel offset that is not a constant takes only images an unnormalized S# can sample without VK_KHR_maintenance8 and shaderImageGatherExtended");
    const auto describe = [](IrTextureNumericClass numeric, RdnaImageDimension dimension, IrBufferFormat conversion, IrBufferFormat packed, bool cube, bool depthBits, bool depthUnorm16, bool srgb) {
        return "{class " + std::to_string(static_cast<int>(numeric)) + ", dim " + std::to_string(static_cast<int>(dimension)) + ", conversion " + std::to_string(static_cast<int>(conversion)) + ", packed " + std::to_string(static_cast<int>(packed)) + ", cube " + std::to_string(cube) + ", depthBits " + std::to_string(depthBits) + ", unorm16 " + std::to_string(depthUnorm16) + ", srgb " + std::to_string(srgb) + "}";
    };
    std::string message = "image descriptor is incompatible with the static runtime image interface: descriptor format " + std::to_string(static_cast<int>(format)) + " decodes to " + describe(decoded.numericClass, decoded.dimension, decoded.conversionFormat, decoded.packedFormat, decoded.cube, decoded.depthBits, decoded.depthUnorm16, decoded.srgbDecode) + (emulated ? " (emulated compare)" : "") + "; modes:";
    for (const auto& mode : modes) message += " " + describe(mode.numericClass, mode.dimension, mode.conversionFormat, mode.packedFormat, mode.cube, mode.depthBits, mode.depthUnorm16, mode.srgbDecode);
    char words[96];
    std::snprintf(words, sizeof(words), "; T# %08x %08x %08x %08x %08x %08x %08x %08x", descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]);
    const auto reason = emulated ? PoisonReason::ColorCompare
        : decoded.fmask ? PoisonReason::Fmask
        : decoded.depthBits ? PoisonReason::DepthBits16
        : decoded.srgbDecode ? PoisonReason::SrgbUnsupported
        : decoded.conversionFormat != IrBufferFormat::Invalid ? PoisonReason::Conversion
        : decoded.dimension == RdnaImageDimension::Dim2DMsaa || decoded.dimension == RdnaImageDimension::Dim2DMsaaArray ? PoisonReason::Multisampled
        : std::none_of(modes.begin(), modes.end(), [&](const ImageResource& mode) { return mode.dimension == decoded.dimension && mode.cube == decoded.cube; }) ? PoisonReason::Dimension
        : PoisonReason::NumericClass;
    return reject(reason, message + words);
}

std::uint32_t ResourceMaterializer::RuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes) {
    const auto match = TryRuntimeImageMode(image, descriptor, modes);
    if (!match.mode.has_value()) throw std::runtime_error(match.detail);
    return *match.mode;
}

std::uint32_t ResourceMaterializer::EmulatedCompareState(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint32_t index) {
    return emulatedCompareState(info, snapshot, index);
}

void ResourceMaterializer::ApplyStaticInterface(IrProgram& program, bool nativeSampleOffsets) const {
    auto& resources = program.Resources();
    if (!resources.resourceTrackingComplete || !resources.srtPlanComplete) throw std::runtime_error("static resource interface requires a completed resource plan");
    auto images = resources.info.images;
    if (!nativeSampleOffsets) {
        for (const auto& block : program.Blocks()) {
            for (const auto* inst : block->Instructions()) {
                if (inst->Opcode() != IrOpcode::ImageSampleRaw) continue;
                const auto& memory = resources.memoryInfo.at(inst->Flags<MemoryFlags>().index);
                if ((memory.imageSampleFlags & RdnaImageSampleFlagOffset) == 0u) continue;
                const auto* address = inst->Argument(2)->Resolve();
                const auto component = GetRdnaImageAddressComponentLayout(memory.imageSampleFlags, 0u);
                const auto argument = component.bitOffset / 32u;
                if (component.bitWidth == 32u && argument < address->ArgumentCount() && address->Argument(argument)->Resolve()->HasImmediate()) continue;
                if ((memory.imageSampleFlags & RdnaImageSampleFlagCompare) != 0u) {
                    images.at(memory.resource).emulatedCompare |= EmulatedCompare::NativeOffsetUnsupported;
                } else if (resources.info.samplers.at(memory.sampler).table == NoTable) {
                    images.at(memory.resource).foldTexelOffsetsOnly = true;
                    resources.info.samplers.at(memory.sampler).foldTexelOffsetsOnly = true;
                }
            }
        }
    }
    bool tables = false;
    for (auto& image : images) {
        image.srgbDecodeFormats = resources.srgbDecodeFormats;
        image.numericClass = image.atomic ? IrTextureNumericClass::Uint : IrTextureNumericClass::Float;
        image.mipCount = image.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u;
        if (image.table == NoTable) continue;
        tables = true;
        const auto modes = RuntimeImageModes(image);
        if (modes.empty() || modes.size() > ImageTableAbi::MaxModes) image.tableOperation = TableOperation::Unsupported;
    }
    for (const auto& pair : resources.info.sampledPairs) {
        if (pair.image >= images.size() || pair.sampler >= resources.info.samplers.size()) throw std::runtime_error("static sampled pair is out of range");
        auto& sampler = resources.info.samplers[pair.sampler];
        sampler.depthCompare = sampler.depthCompare || images[pair.image].depthCompare;
        tables = tables || sampler.table != NoTable;
    }
    resources.info.images = std::move(images);
    resources.guardedSrtSlots = Detail::ComputeGuardedFlatSlots(resources);
    resources.srtGuardOffset = static_cast<std::uint32_t>(resources.srtReads.size());
    if (tables || !resources.guardedSrtSlots.empty()) resources.info.usesFaultBuffer = true;
    PrepareImageModes(resources.info);
}

void ResourceMaterializer::PrepareImageModes(ShaderInfo& info) {
    info.runtimeImageModes.clear();
    info.runtimeImageModes.reserve(info.images.size());
    for (const auto& image : info.images) info.runtimeImageModes.push_back(RuntimeImageModes(image));
}

namespace {

void ownPlanValues(IrResourcePlan& plan) {
    std::vector<IrValue**> roots;
    for (auto& source : plan.descriptorSources) {
        for (auto& dword : source.dwords) roots.push_back(&dword);
    }
    for (auto& read : plan.srtReads) roots.push_back(&read.value);
    for (auto& block : plan.controlFlow) roots.push_back(&block.condition);
    for (auto& value : plan.uniformFill.values) roots.push_back(&value);

    std::unordered_map<const IrValue*, IrValue*> clones;
    std::vector<const IrValue*> order;
    std::vector<const IrValue*> pending;
    for (const auto* root : roots) {
        if (*root != nullptr) pending.push_back(*root);
    }
    while (!pending.empty()) {
        const auto* value = pending.back();
        pending.pop_back();
        if (!clones.emplace(value, nullptr).second) continue;
        order.push_back(value);
        for (const auto* argument : value->Arguments()) {
            if (argument != nullptr) pending.push_back(argument);
        }
    }
    for (const auto* value : order) {
        auto clone = std::make_unique<IrValue>(value->Opcode(), value->Type(), value->Id());
        clone->SetFlags(value->Flags<std::uint64_t>());
        if (value->HasImmediate()) clone->SetImmediateU64(value->ImmediateU64());
        clone->SetRegister(value->Register());
        clones[value] = clone.get();
        plan.valueStorage.push_back(std::move(clone));
    }
    std::unordered_map<const IrBlock*, IrBlock*> blocks;
    const auto blockFor = [&](const IrBlock* block) {
        auto& clone = blocks[block];
        if (clone == nullptr) {
            plan.blockStorage.push_back(std::make_unique<IrBlock>(block->Id()));
            clone = plan.blockStorage.back().get();
        }
        return clone;
    };
    for (const auto* value : order) {
        auto* clone = clones.at(value);
        for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
            const auto* argument = value->Argument(index);
            auto* mapped = argument == nullptr ? nullptr : clones.at(argument);
            if (value->IsPhi()) {
                clone->AddPhiOperand(blockFor(value->PhiBlock(index)), mapped);
            } else {
                clone->AddArgument(mapped);
            }
        }
    }
    for (auto* root : roots) {
        if (*root != nullptr) *root = clones.at(*root);
    }
}

}

IrResourcePlan ResourceMaterializer::ExtractPlan(const IrProgram& program) const {
    const IrResourcePlan& source = program.Resources();
    if (!source.resourceTrackingComplete || !source.srtPlanComplete) {
        throw std::runtime_error("ResourceMaterializer::ExtractPlan requires a completed resource and SRT plan");
    }
    IrResourcePlan plan;
    plan.stage = source.stage;
    plan.shaderHash = source.shaderHash;
    plan.userDataBase = source.userDataBase;
    plan.userDataCount = source.userDataCount;
    plan.srgbDecodeFormats = source.srgbDecodeFormats;
    plan.memoryInfo = source.memoryInfo;
    plan.descriptorSources = source.descriptorSources;
    plan.controlFlow = source.controlFlow;
    plan.srtReads = source.srtReads;
    plan.cleanFlatSlots = source.cleanFlatSlots;
    plan.requiresSpecializationMemory = source.requiresSpecializationMemory;
    plan.srtPlanComplete = source.srtPlanComplete;
    plan.resourceTrackingComplete = source.resourceTrackingComplete;
    plan.info = source.info;
    PrepareImageModes(plan.info);
    plan.uniformFill = source.uniformFill;
    ownPlanValues(plan);
    const auto addSource = [&plan](std::uint32_t index) {
        if (index >= plan.descriptorSources.size()) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan resource references an unknown descriptor source");
        }
        plan.materializationSources.push_back(index);
    };
    for (const auto& buffer : plan.info.buffers) addSource(buffer.source);
    for (const auto& image : plan.info.images) {
        if (image.source >= plan.descriptorSources.size()) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan image references an unknown descriptor source");
        }
        if (image.table != NoTable) {
            plan.requiresSpecializationMemory = true;
        } else {
            addSource(image.source);
        }
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.table != NoTable) {
            plan.requiresSpecializationMemory = true;
        } else {
            addSource(sampler.source);
        }
    }
    plan.pureFlatSlots = Detail::ComputePureFlatSlots(plan);
    plan.guardedSrtSlots = Detail::ComputeGuardedFlatSlots(plan);
    return plan;
}

void ResourceMaterializer::Materialize(const IrResourcePlan& program, const SrtRuntime& runtime, ResourceSnapshot& snapshot) const {
    const IrResourcePlan& plan = program;
    if (!plan.resourceTrackingComplete) {
        throw std::runtime_error("ResourceMaterializer::Materialize requires a completed resource plan");
    }
    if (plan.requiresSpecializationMemory && runtime.readMemory == nullptr) {
        throw std::runtime_error("ResourceMaterializer::Materialize requires runtime memory access for image tables");
    }
    SrtWalker walker;
    ResourceSnapshot nextSnapshot;
    std::vector<std::uint8_t> activeSources;
    try {
        materializeSnapshot(plan, runtime, walker, nextSnapshot, activeSources);
        TableSnapshotter(plan, runtime, walker, activeSources, nextSnapshot.tables).Run();
    } catch (...) {
        reportTables();
        throw;
    }
    const auto started = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    materializeTables(plan, nextSnapshot);
    materializeSrtGuards(plan, nextSnapshot);
    if (MaterializeProfiled()) specializationNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    snapshot = std::move(nextSnapshot);
    reportTables();
}

std::uint64_t ResourceMaterializer::SpecializationNanoseconds() {
    return specializationNanoseconds.load(std::memory_order_relaxed);
}

}
