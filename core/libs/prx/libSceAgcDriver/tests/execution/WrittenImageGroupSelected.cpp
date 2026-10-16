#include "WrittenImageFixture.hpp"

namespace {

using namespace WrittenImages;

void Run() {
    for (std::uint32_t texture = 0; texture < Textures; ++texture) WriteDescriptor(texture, texture);
    Commands commands;
    commands.Dispatch(GroupProgram(), ReaderUserData(0), Lanes, Height);
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectSampled(0, [](std::uint32_t row) { return row; }, "written image: the first use of T#s each workgroup selects from a table the CPU wrote");
    Commands again;
    again.Dispatch(GroupProgram(), ReaderUserData(1), Lanes, Height);
    again.Submit(0);
    NewPath::WaitIdle();
    ExpectSampled(1, [](std::uint32_t row) { return row; }, "written image: a later submission's use of the T#s each workgroup selects");
}

}

int main() {
    return Main("written image group selected", Run);
}
