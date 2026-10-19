#include "WrittenImageFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"

namespace {

using namespace WrittenImages;
using AgcDriver::DriverDetail::Driver;

constexpr std::uint32_t Repeats = 8;

void DispatchReader() {
    Commands commands;
    commands.Dispatch(ChainProgram(), ReaderUserData(0), Lanes, Height);
    commands.Submit(0);
    NewPath::WaitIdle();
}

void Run() {
    WriteDescriptor(0, 0);
    DispatchReader();
    ExpectSampled(0, [](std::uint32_t) { return 0u; }, "run-time result reuse: first dispatch");
    const auto first = Driver::RunTimeEvaluations();
    for (std::uint32_t repeat = 0; repeat < Repeats; ++repeat) DispatchReader();
    ExpectSampled(0, [](std::uint32_t) { return 0u; }, "run-time result reuse: repeated dispatches");
    const auto repeated = Driver::RunTimeEvaluations() - first;
    Expect(repeated == 0u, "run-time result reuse: " + std::to_string(repeated) + " evaluations for " + std::to_string(Repeats) + " dispatches whose user data and table words did not change");
    WriteDescriptor(0, 1);
    DispatchReader();
    const auto changed = Driver::RunTimeEvaluations() - first - repeated;
    Expect(changed == 1u, "run-time result reuse: " + std::to_string(changed) + " evaluations for the dispatch after its texture descriptor changed, expected 1");
    ExpectSampled(0, [](std::uint32_t) { return 1u; }, "run-time result reuse: dispatch after the texture descriptor changed");
    const auto settled = Driver::RunTimeEvaluations();
    DispatchReader();
    Expect(Driver::RunTimeEvaluations() == settled, "run-time result reuse: the changed descriptor's result was not reused on the next dispatch");
    ExpectSampled(0, [](std::uint32_t) { return 1u; }, "run-time result reuse: dispatch reusing the changed descriptor's result");
}

}

int main() {
    return Main("run time result reuse", Run);
}
