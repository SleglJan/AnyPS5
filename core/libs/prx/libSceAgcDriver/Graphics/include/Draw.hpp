#include <string>
#include <optional>
#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAW_HPP

#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <memory>
#include <vector>

namespace AgcDriver::Graphics {

struct HostImport;

void Draw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders);
std::optional<std::string> KnownValidationFailure(const Context& context, std::span<const CompiledShader> shaders, const State& state);
std::uint64_t DrawRenderPassKey(const Context& context, const State& state, std::span<const VkImageView> targetViews);

struct DrawInputCopy {
    std::shared_ptr<Buffer> buffer;
    bool reused = false;
    std::uint32_t derived = 0;
    std::uint64_t generation = 0;
    std::uint64_t registryGeneration = 0;
};
DrawInputCopy CopyDrawInput(const Context& context, Recorder* recorder, std::uint64_t address, std::size_t bytes, std::size_t alignment, Recorder::SnapshotUse use);
DrawInputCopy CopyZeroPaddedDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t validBytes);
void KeepDrawInput(Recorder* recorder, std::uint64_t address, const DrawInputCopy& copy, Recorder::SnapshotUse use, std::uint32_t derived);
struct DrawInputSource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    std::shared_ptr<DeviceBuffer> gathered;
    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};
DrawInputSource InPlaceDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t alignment, bool cpuReads);
std::uint64_t DrawInputCopies();
struct InPlaceIndices {
    std::optional<std::uint32_t> highest;
    bool reused = false;
};
InPlaceIndices InPlaceIndexRange(Recorder& recorder, std::uint64_t address, std::size_t bytes, std::uint32_t indexSize, bool skipRestart);

std::array<std::uint32_t, 4> MeshIndexBufferDescriptor(const Pm4::DrawParameters& draw);

struct MeshArguments {
    std::uint32_t groups;
    std::uint32_t instances;
    std::uint32_t layers;
    std::uint32_t indexCount;
    std::uint32_t firstIndex;
};
static_assert(sizeof(MeshArguments) == ShaderRecompiler::MeshArgumentBytes);
struct MeshArgumentRules {
    std::uint32_t indexCount;
    std::uint32_t inputSize;
    std::uint32_t step;
    std::uint32_t primitivesPerGroup;
    std::uint32_t maxGroups;
    std::uint32_t maxInstances;
    std::uint32_t maxTotal;
};
MeshArgumentRules MeshArgumentRulesFor(const Context& context, const ShaderRecompiler::MeshConfiguration& mesh, std::uint32_t indexCount);
MeshArguments ResolveMeshArguments(const Pm4::DrawArguments& record, const MeshArgumentRules& rules);

// How an indirect draw's records were read, for the [draws] line (APS5_PROFILE_DRAW): Gpu is
// vkCmdDrawIndirect from the host import; every other value names why the CPU read them instead,
// either decided by the driver before the draw (its patched-SGPR fold rules, the shader path, the
// draw index, the vertex ranges, the device features, APS5_NO_GPU_INDIRECT_DRAW) or under the device
// lock in Draw (the memory state of the records). `readMs` is the CPU read's time (its sync);
// `rewritten` counts a GPU-side draw whose records were copied and patched with a constant.
enum class IndirectDrawPath : std::uint8_t { Gpu, NotFolded, FetchUnknown, NonVertexPath, IndxOffset, DrawIndex, VertexRange, FeatureGap, PendingImage, PendingLabelOrCopy, NotImported, Disabled, Count };
void CountIndirectDraw(IndirectDrawPath path, double readMs, bool rewritten = false);
const char* IndirectDrawPathName(IndirectDrawPath path);

// Draw packets that drew nothing, for the [draws] line: Nothing (an empty count, or no color writes
// and no pixel shader), Prechecked (State.hpp's DrawRejection refused it before the decode) and
// Thrown (the decode or the build threw); `us` is the packet's time in the driver.
enum class DrawSkip : std::uint8_t { Nothing, Prechecked, Thrown, Count };
void CountDrawSkip(DrawSkip kind, double us);

void RunColorMetadataPass(const Context& context, const ColorMetadataPass& pass);
bool KeysDecidedOnGpu(const Context& context, const ColorTarget& color);

}

#endif
