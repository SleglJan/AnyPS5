#ifndef CORE_SHADER_RECOMPILER_IMAGETABLEABI_HPP
#define CORE_SHADER_RECOMPILER_IMAGETABLEABI_HPP

#include <cstdint>
#include <string_view>

namespace ShaderRecompiler::ImageTableAbi {

inline constexpr std::uint32_t Version = 6;

inline constexpr std::uint32_t HeaderVersion = 0;
inline constexpr std::uint32_t HeaderDeviceLow = 1;
inline constexpr std::uint32_t HeaderDeviceHigh = 2;
inline constexpr std::uint32_t HeaderWords = 3;

inline constexpr std::uint32_t MaxModes = 256;

enum class PoisonReason : std::uint32_t {
    NotImage,
    InvalidFormat,
    Dimension,
    Multisampled,
    Fmask,
    ColorCompare,
    DepthBits16,
    Conversion,
    SrgbUnsupported,
    NumericClass,
    Reduction,
    Unnormalized,
    DriverRejected,
    UnsupportedOperation,
    Count
};

constexpr std::string_view PoisonReasonName(PoisonReason reason) {
    switch (reason) {
        case PoisonReason::NotImage: return "not an image";
        case PoisonReason::InvalidFormat: return "invalid format";
        case PoisonReason::Dimension: return "dimension";
        case PoisonReason::Multisampled: return "multisampled";
        case PoisonReason::Fmask: return "FMASK";
        case PoisonReason::ColorCompare: return "color texture under comparison";
        case PoisonReason::DepthBits16: return "depth bits with 16-bit results";
        case PoisonReason::Conversion: return "format conversion";
        case PoisonReason::SrgbUnsupported: return "sRGB decode";
        case PoisonReason::NumericClass: return "numeric class";
        case PoisonReason::Reduction: return "min or max reduction";
        case PoisonReason::Unnormalized: return "unnormalized coordinates";
        case PoisonReason::DriverRejected: return "rejected by the driver";
        case PoisonReason::UnsupportedOperation: return "unsupported operation";
        case PoisonReason::Count: break;
    }
    return "unknown";
}

}

#endif
