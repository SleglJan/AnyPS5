#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_RESOURCEMATERIALIZER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_RESOURCEMATERIALIZER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"
#include "DeviceImageTableAbi.hpp"
#include "ImageTableAbi.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ShaderRecompiler {

struct ImageTableElement {
    std::array<std::uint32_t, 8> words{};
    RdnaImageDimension dimension = RdnaImageDimension::Unknown;
    bool depthCompare = false;
    DescriptorBindingKind kind = DescriptorBindingKind::Buffers;
    std::vector<std::uint32_t> samplers;
    bool decodedCopy = false;
};

struct ImageTableSamplerElement {
    std::array<std::uint32_t, 4> words{};
    bool compare = false;
    bool point = false;
};

struct ImageTablePoisonRecord {
    std::array<std::uint32_t, 8> words{};
    std::uint32_t dwordCount = 0;
    std::uint32_t resource = 0;
    ImageTableAbi::PoisonReason reason = ImageTableAbi::PoisonReason::OutsideSnapshot;
};

struct ResolvedImageTables {
    std::vector<std::uint32_t> map;
    std::vector<ImageTableElement> elements;
    std::vector<ImageTableSamplerElement> samplers;
    std::vector<ImageTablePoisonRecord> poison;
    std::uint32_t faults = 0;
};

struct RuntimeImageModeMatch {
    std::optional<std::uint32_t> mode;
    ImageTableAbi::PoisonReason reason = ImageTableAbi::PoisonReason::InvalidFormat;
    std::string detail;
};

struct DeviceImageInterface {
    std::uint64_t id = 0;
    ImageResource image;
    std::vector<ImageResource> modes;
};

class ResourceMaterializer {
public:
    static std::uint32_t ColorCompareState(const ImageResource& image, const DescriptorValue& descriptor, std::span<const std::uint32_t, 4> sampler);
    void ApplyStaticInterface(IrProgram& program, bool nativeSampleOffsets = true) const;
    static std::vector<ImageResource> RuntimeImageModes(const ImageResource& image);
    static RuntimeImageModeMatch TryRuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes);
    static void PrepareImageModes(ShaderInfo& info);
    static const char* UnnormalizedSampleMismatch(const ImageResource& image);
    static const char* VolumePlaneBiasSamplerMismatch(const ImageResource& image, const ImageResource& mode, std::span<const std::uint32_t, 4> sampler);
    static bool BindsDecodedCopy(const ImageResource& mode, const DescriptorValue& descriptor);
    static ResolvedImageTables ResolveImageTables(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint64_t memoKey = 0);
    static std::uint64_t DeviceImageInterfaceId(const ImageResource& image, std::span<const ImageResource> modes);
    static void RegisterDeviceImageInterfaces(const ShaderInfo& info);
    static std::shared_ptr<const DeviceImageInterface> FindDeviceImageInterface(std::uint64_t id);
    static std::vector<DeviceImageTableAbi::Expectation> ExpectedDeviceImages(const IrResourcePlan& plan, const SrtRuntime& runtime, bool* complete = nullptr);
    static void EvaluateMemoryBuffers(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& stores, std::vector<DescriptorValue>& loads);
    static void EvaluateBufferSites(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& stores, std::vector<DescriptorValue>& loads);
    [[nodiscard]] IrResourcePlan ExtractPlan(const IrProgram& program) const;
    void Materialize(const IrResourcePlan& program, const SrtRuntime& runtime, ResourceSnapshot& snapshot) const;
    static bool ReadsResourcesAtRunTime(const IrResourcePlan& plan);
    static void RunTimeSnapshot(const IrResourcePlan& plan, const SrtRuntime& runtime, ResourceSnapshot& snapshot);
    static std::uint64_t SpecializationNanoseconds();

private:
    static void evaluateBufferSites(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& stores, std::vector<DescriptorValue>& loads, bool memoryOnly);
};

}

#endif
