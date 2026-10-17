#include "StorageMipFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"

namespace {

using namespace StorageMips;
using AgcDriver::Graphics::ViewCreationCounts;
using AgcDriver::Graphics::ViewCreationTotals;
using AgcDriver::Graphics::ViewKind;

std::uint64_t Delta(const std::array<std::uint64_t, ViewCreationTotals::Kinds>& before, const std::array<std::uint64_t, ViewCreationTotals::Kinds>& after) {
    const auto kind = static_cast<std::size_t>(ViewKind::Storage);
    return after[kind] - before[kind];
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t index = 1; index <= 4; ++index) {
        const auto texture = NewTexture(index);
        const auto other = NewTexture(index + 4u);
        Surface(&texture);
        const View view{&texture, 0, Levels - 1u};
        const View otherView{&other, 1, Levels - 1u};
        Model model;
        const Dispatch dispatch(view, otherView, true, 0x10000u * index);
        const auto what = "storage mip views: the first use of four-level texture " + std::to_string(index);
        const auto compiled = CompileFor(device, dispatch, what);
        const auto before = ViewCreationCounts();
        const auto outcome = RunOne(device, compiled);
        const auto after = ViewCreationCounts();
        ExpectClean(outcome, what);
        dispatch.Apply(model);
        ExpectTexture(texture, model, what);
        Expect(Delta(before.onPool, after.onPool) >= Levels, what + ": " + std::to_string(Delta(before.onPool, after.onPool)) + " storage views were made on the view pool, expected one per level");
        Expect(Delta(before.onCaller, after.onCaller) == 0u, what + ": " + std::to_string(Delta(before.onCaller, after.onCaller)) + " storage views were made on the recording thread");
        Expect(Delta(before.waits, after.waits) == 0u, what + ": the recording thread waited " + std::to_string(Delta(before.waits, after.waits)) + " times for the views of the mip run");
    }
}

}

int main() {
    return Main("device table storage mip views", Run);
}
