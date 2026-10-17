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

enum class RejectionReason : std::uint32_t {
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

constexpr std::string_view RejectionReasonName(RejectionReason reason) {
    switch (reason) {
        case RejectionReason::NotImage: return "not an image";
        case RejectionReason::InvalidFormat: return "invalid format";
        case RejectionReason::Dimension: return "dimension";
        case RejectionReason::Multisampled: return "multisampled";
        case RejectionReason::Fmask: return "FMASK";
        case RejectionReason::ColorCompare: return "color texture under comparison";
        case RejectionReason::DepthBits16: return "depth bits with 16-bit results";
        case RejectionReason::Conversion: return "format conversion";
        case RejectionReason::SrgbUnsupported: return "sRGB decode";
        case RejectionReason::NumericClass: return "numeric class";
        case RejectionReason::Reduction: return "min or max reduction";
        case RejectionReason::Unnormalized: return "unnormalized coordinates";
        case RejectionReason::DriverRejected: return "rejected by the driver";
        case RejectionReason::UnsupportedOperation: return "unsupported operation";
        case RejectionReason::Count: break;
    }
    return "unknown";
}

}

#endif
