#include "prx/libc/include/general/VabiMacros.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

extern "C" {
std::int32_t APS5_VABI sceTextToSpeech2Initialize(const void* param);
std::int32_t APS5_VABI sceTextToSpeech2Open();
std::int32_t APS5_VABI sceTextToSpeech2Close();
int APS5_VABI sceTextToSpeech2Speak();
int APS5_VABI sceTextToSpeech2Terminate();
}

namespace {

constexpr std::int32_t Unsupported = static_cast<std::int32_t>(0x8002002D);

template <typename Call>
bool Throws(Call call) {
    try {
        call();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void Require(bool value, const char* message) {
    if (value) return;
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}

}

int main() {
    const std::uint32_t param[12]{0x2000000, 0x26c};
    Require(sceTextToSpeech2Initialize(param) == Unsupported, "TextToSpeech2: initialization did not report the unsupported operation");
    Require(sceTextToSpeech2Close() == Unsupported, "TextToSpeech2: close after failed initialization did not report the unsupported operation");
    Require(sceTextToSpeech2Open() == Unsupported, "TextToSpeech2: open after failed initialization did not report the unsupported operation");
    Require(Throws([] { sceTextToSpeech2Speak(); }), "TextToSpeech2: speak without an open speech engine did not throw");
    Require(Throws([] { sceTextToSpeech2Terminate(); }), "TextToSpeech2: terminate without an initialized speech engine did not throw");
    std::puts("TextToSpeech2 tests passed");
    return 0;
}
