#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "SceShaders.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint8_t FetchShaderType = 8;

alignas(256) constexpr std::array<std::uint32_t, 4> FetchCode{0xe00c2000u, 0x80010004u, 0xbf8c3f70u, 0xbe802000u};

struct alignas(8) Header {
    Shader shader{};
    ShaderUserData userData{};
};

Header Fetch;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

void Run() {
    Fetch.shader.file_header = 0x34333231u;
    Fetch.shader.version = 0x18u;
    Fetch.shader.header_size = sizeof(Header);
    Fetch.shader.shader_size = static_cast<std::uint32_t>(sizeof(FetchCode));
    Fetch.shader.code = FetchCode.data();
    Fetch.shader.type = FetchShaderType;
    Fetch.shader.user_data = &Fetch.userData;
    std::string log;
    {
        NewPath::StderrCapture capture;
        AgcDriverRegisterShader_nid_postfix(&Fetch.shader);
        AgcDriver::DriverDetail::AwaitRegistrationPreparation();
        log = capture.Finish();
    }
    std::cerr << log;
    Expect(log.find("unsupported registered shader type") == std::string::npos, "the registration of a fetch shader reported its binary type as unsupported");
    Expect(log.find("could not be prepared at registration") == std::string::npos, "the registration of a fetch shader left it to be prepared at use");
    const auto registry = AgcDriver::DriverDetail::Driver::Get().RegisteredShaders();
    Require(registry != nullptr, "the registration of a fetch shader published no shader registry");
    const auto snapshot = AgcDriver::DriverDetail::RegisteredHeader(*registry, &Fetch.shader);
    Require(snapshot != nullptr, "the registration of a fetch shader published no snapshot");
    AgcDriver::DriverDetail::ShaderPreparationTransaction transaction;
    const auto& state = transaction.Read(*snapshot);
    Expect(!state.deferred, "the registration of a fetch shader deferred a preparation to its use");
    Expect(state.entries.empty(), "the registration of a fetch shader prepared " + std::to_string(state.entries.size()) + " stage programs from it, expected none: the vertex program that calls it inlines its loads");
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Run();
        AgcDriverShutdown_nid_postfix();
        if (failures != 0) {
            std::cerr << failures << " registered fetch shader checks failed\n";
            return 1;
        }
        std::puts("registered fetch shader tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
