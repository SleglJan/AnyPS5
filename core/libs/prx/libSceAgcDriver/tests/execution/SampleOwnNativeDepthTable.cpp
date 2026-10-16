#include "MultisampleFeedbackFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <bit>

#ifdef _WIN32

namespace {

using namespace MultisampleFeedback;
using namespace AgcDriver::Graphics;

constexpr VkExtent2D Extent{16u, 16u};
constexpr std::uint32_t Samples = 4u;
constexpr const char* Refusal = "sampling a native multisampled depth target the draw attaches";

std::array<std::uint32_t, 8> DepthSampleWords(const DepthTarget& target) {
    const auto base = (target.address | target.pipeBankXor) >> 8u;
    const auto metadata = target.htileAddress >> 8u;
    const auto format = target.format == VK_FORMAT_D16_UNORM ? 7u : 22u;
    const auto count = static_cast<std::uint32_t>(std::countr_zero(target.samples));
    const auto width = target.extent.width - 1u, height = target.extent.height - 1u;
    return {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32u) & 255u) | (format << 20u) | ((width & 3u) << 30u), (width >> 2u) | (height << 14u), (14u << 28u) | (24u << 20u) | (count << 16u) | 0x204u, 0u, count << 4u, (1u << 10u) | (1u << 19u) | (1u << 21u) | (static_cast<std::uint32_t>(metadata & 255u) << 24u), static_cast<std::uint32_t>(metadata >> 8u)};
}

State DepthState(const DepthTarget& depth) {
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.depth = depth;
    state.depthTest = state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    state.renderExtent = Extent;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0.0f, 0.0f, static_cast<float>(Extent.width), static_cast<float>(Extent.height), 0.0f, 1.0f};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, Extent};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.samples = Samples;
    return state;
}

void Run(AgcDriver::VulkanDevice& device) {
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() {
            device.WaitIdle();
            ClearCachedTextures(device.Device());
            ClearDepthSurfaces(device.Device());
        }
    } cleanup{device};
    Triangle triangle;
    const auto depthBytes = MultisampledDepthBytes_nid_no_patch(Extent, 4u, Samples);
    NewPath::PrivatePages memory(static_cast<std::uint32_t>((depthBytes + NewPath::PageBytes - 1u) / NewPath::PageBytes));
    NewPath::PrivatePages metadata(1);
    NewPath::PrivatePages descriptorWords(1);
    DepthTarget depth{memory.Address(), 0u, Extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, metadata.Address(), false, Samples};
    depth.depthSwizzle = 24u;
    depth.iterate256 = depth.htilePipeAligned = true;
    const auto state = DepthState(depth);
    const auto descriptor = DepthSampleWords(depth);
    std::memcpy(descriptorWords.Data(), descriptor.data(), sizeof(descriptor));
    const std::array<std::uint32_t, 2> userData{NewPath::Low(descriptorWords.Address()), NewPath::High(descriptorWords.Address())};
    ExpectRefusedWithin(device, [&] { Draw(device, state, triangle, SamplingPixel(true, false), userData); }, Refusal, "a draw sampling its native multisampled depth target through the device image table");
}

}

#endif

int main() {
#ifdef _WIN32
    return MultisampleFeedback::Main("sample own native depth through the device table", Run);
#else
    std::printf("skipped, guest pages are Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
