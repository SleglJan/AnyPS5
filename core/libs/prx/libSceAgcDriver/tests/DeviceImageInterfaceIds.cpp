#include "Optimization/ResourceMaterializer.hpp"
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ShaderRecompiler;

ImageResource SampledImage() {
    ImageResource image;
    image.resourceClass = ImageResourceClass::Sampled;
    image.numericClass = IrTextureNumericClass::Float;
    image.dimension = RdnaImageDimension::Dim2D;
    image.read = true;
    image.table = 0u;
    image.deviceTable = true;
    return image;
}

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void ExpectDistinct(const char* field, const std::function<void(ImageResource&)>& change) {
    const auto base = SampledImage();
    auto changed = base;
    change(changed);
    Expect(base != changed, std::string(field) + ": the changed image equals the base image");
    const std::vector<ImageResource> modes{base};
    Expect(ResourceMaterializer::DeviceImageInterfaceId(base, modes) != ResourceMaterializer::DeviceImageInterfaceId(changed, modes), std::string("two image interfaces that differ in ") + field + " share an id");
    ShaderInfo info;
    info.images = {base, changed};
    info.runtimeImageModes = {modes, modes};
    ResourceMaterializer::RegisterDeviceImageInterfaces(info);
}

}

int main() {
    try {
        ExpectDistinct("cubeFaceCompatible", [](ImageResource& image) { image.cubeFaceCompatible = false; });
        ExpectDistinct("singleMipPlaneBiasCompatible", [](ImageResource& image) { image.singleMipPlaneBiasCompatible = true; });
        ExpectDistinct("flatVolumeLineCompatible", [](ImageResource& image) { image.flatVolumeLineCompatible = false; });
        ExpectDistinct("directRuntime", [](ImageResource& image) { image.directRuntime = true; });
    } catch (const std::exception& error) {
        std::cerr << "DeviceImageInterfaceIds: " << error.what() << '\n';
        return 1;
    }
    std::puts("DeviceImageInterfaceIds: image interfaces that differ in any field have different ids");
    return 0;
}
