#include "MultisampleFeedbackFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"

#ifdef _WIN32

namespace {

using namespace MultisampleFeedback;

constexpr std::uint32_t Side = 16;
constexpr std::uint32_t Samples = 4;
constexpr std::uint32_t SampleShift = 2;
constexpr std::uint32_t Float16x4Info = 0x0730u;
constexpr std::uint32_t MultisampledAttrib = (SampleShift << 12u) | (SampleShift << 15u);
constexpr std::uint32_t TiledAttrib3 = 0x4dc6c000u;
constexpr std::uint32_t Format16161616Float = 71;
constexpr std::uint32_t Type2DMsaa = 14;
constexpr std::uint32_t TileRenderTarget = 0x1b;
constexpr const char* Refusal = "sampling a multisampled color target the draw renders to is not implemented";

AgcDriver::Graphics::ColorTarget DecodeTarget(std::uint64_t address) {
    AgcDriver::Registers cx;
    cx[0x318] = static_cast<std::uint32_t>(address >> 8u);
    cx[0x31b] = 0;
    cx[0x31c] = Float16x4Info;
    cx[0x31d] = MultisampledAttrib;
    cx[0x3b0] = ((Side - 1u) << 14u) | (Side - 1u);
    cx[0x3b8] = TiledAttrib3;
    cx[0x390] = static_cast<std::uint32_t>(address >> 40u);
    const auto color = AgcDriver::Graphics::DecodeColorBuffer(cx, 0);
    AgcDriver::Graphics::Require(color.samples == Samples, "the target did not decode as a surface of 4 samples");
    return color;
}

AgcDriver::Graphics::State TargetState(const AgcDriver::Graphics::ColorTarget& color) {
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 0xf;
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.colors = {color};
    state.color = color;
    state.hasColorTarget = true;
    state.renderExtent = {Side, Side};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Side), static_cast<float>(Side), -static_cast<float>(Side), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Side, Side}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blends = {blend};
    state.blend = blend;
    state.blendConstants = {};
    state.samples = Samples;
    return state;
}

std::array<std::uint32_t, 8> TargetDescriptor(std::uint64_t address) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format16161616Float << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (SampleShift << 16u) | (TileRenderTarget << 20u) | (Type2DMsaa << 28u),
        0u,
        SampleShift << 4u,
        0u,
        0u,
    };
}

void Run(AgcDriver::VulkanDevice& device) {
    Triangle triangle;
    const auto& renderOnly = SamplingPixel(false, true);
    for (const bool fromMemory : {false, true}) {
        const std::string what = fromMemory ? "a draw sampling its multisampled color target through the device image table" : "a draw sampling its multisampled color target through a bound texture";
        NewPath::PrivatePages target(4);
        NewPath::PrivatePages descriptorWords(1);
        const auto color = DecodeTarget(target.Address());
        const auto state = TargetState(color);
        const auto descriptor = TargetDescriptor(target.Address());
        std::memcpy(descriptorWords.Data(), descriptor.data(), sizeof(descriptor));
        NewPath::PrivatePages otherTarget(4);
        const auto other = TargetDescriptor(otherTarget.Address());
        std::array<std::uint32_t, 8> otherUserData{};
        std::copy(other.begin(), other.end(), otherUserData.begin());
        const auto& pixel = SamplingPixel(fromMemory, true);
        std::vector<std::uint32_t> userData;
        if (fromMemory) {
            userData = {NewPath::Low(descriptorWords.Address()), NewPath::High(descriptorWords.Address())};
        } else {
            userData.assign(descriptor.begin(), descriptor.end());
        }
        ExpectRefusedWithin(device, [&] { Draw(device, state, triangle, pixel, userData); }, Refusal, what);
        try {
            Draw(device, state, triangle, renderOnly, otherUserData);
            device.WaitIdle();
        } catch (const std::exception& error) {
            Expect(false, what + ": a draw into the same target sampling another multisampled surface was refused: " + error.what());
        }
    }
}

}

#endif

int main() {
#ifdef _WIN32
    return MultisampleFeedback::Main("sample own multisample target", Run);
#else
    std::printf("skipped, guest pages are Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
