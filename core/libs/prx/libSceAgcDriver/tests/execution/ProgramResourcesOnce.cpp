#include "DrawSubmitFixture.hpp"
#include "WrittenImageFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"

namespace {

using namespace DrawSubmit;

constexpr std::uint32_t Dispatches = 8;
constexpr std::uint32_t RegionWords = 256;
constexpr std::uint32_t BufferStoreDword = 0x1cu;
constexpr std::uint32_t TargetWidth = 64;
constexpr std::uint32_t TargetHeight = 32;
constexpr std::uint32_t Draws = 16;
constexpr std::uint32_t BandRows = TargetHeight / Draws;
constexpr std::uint32_t BandVertices = 6;
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t PixelUserWords = 4;

TestGuest::Array<std::uint32_t, Dispatches * RegionWords> Stored;
TestGuest::Array<std::array<float, 4>, Draws * BandVertices> Vertices;
TestGuest::Array<std::uint32_t, TargetWidth * TargetHeight> Pixels;

using Counts = AgcDriver::Graphics::ProgramResourceCounts;

Counts Now() {
    return AgcDriver::Graphics::ProgramResourceCounters();
}

void ExpectBuiltOnce(const Counts& before, std::uint64_t uses, const std::string& what) {
    const auto after = Now();
    const auto built = after.programs - before.programs;
    const auto used = after.uses - before.uses;
    Expect(used >= uses, what + ": " + std::to_string(used) + " resource uses, expected at least " + std::to_string(uses));
    Expect(built == 1u, what + ": " + std::to_string(built) + " resource objects were built for one program over " + std::to_string(used) + " uses, expected one");
}

std::array<std::uint32_t, 4> Structured(const void* base, std::uint32_t stride, std::uint32_t records) {
    const auto address = AddressOf(base);
    return {Low(address), (High(address) & 0xffffu) | (stride << 16u), records, 0x01016facu};
}

constexpr std::array<std::uint32_t, 2> IndexedStore(std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource) {
    return {0xe0000000u | (BufferStoreDword << 18u) | 0x2000u, (NewPath::Isa::ZeroOffset << 24u) | ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

const Program& UserDataStoreProgram() {
    static const Program program = [] {
        using namespace WrittenImages::Ops;
        Program built;
        built << Vop2(VAddNcU32, 1, 4, 0)
              << IndexedStore(1, 0, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

std::uint32_t Seed(std::uint32_t dispatch) {
    return 0x1000u * (dispatch + 1u) + 0x35u * dispatch;
}

void Dispatched() {
    Stored.fill(DeviceTable::Untouched);
    const auto before = Now();
    Commands commands;
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        const auto resource = Structured(Stored.data() + dispatch * RegionWords, 4u, DeviceTable::Lanes);
        const std::array<std::uint32_t, 5> userData{resource[0], resource[1], resource[2], resource[3], Seed(dispatch)};
        commands.Dispatch(UserDataStoreProgram(), userData, DeviceTable::Lanes);
    }
    commands.Submit(0);
    NewPath::WaitIdle();
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(Stored.data()), decltype(Stored)::ByteSize);
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        std::size_t wrong = 0;
        std::string first;
        for (std::uint32_t word = 0; word < RegionWords; ++word) {
            const auto expected = word < DeviceTable::Lanes ? word + Seed(dispatch) : DeviceTable::Untouched;
            const auto actual = Stored[dispatch * RegionWords + word];
            if (actual == expected) continue;
            if (wrong++ == 0u) first = "word " + std::to_string(word) + " is " + Hex(actual) + ", expected " + Hex(expected);
        }
        Expect(wrong == 0u, "dispatches: dispatch " + std::to_string(dispatch) + " stores through the V# and seed of its own user data: " + std::to_string(wrong) + " words differ, first " + first);
    }
    ExpectBuiltOnce(before, Dispatches, "dispatches: one program dispatched " + std::to_string(Dispatches) + " times with other user data each time");
}

void FillBand(std::uint32_t band) {
    const auto top = 1.0f - 2.0f * static_cast<float>(band * BandRows) / static_cast<float>(TargetHeight);
    const auto bottom = 1.0f - 2.0f * static_cast<float>((band + 1u) * BandRows) / static_cast<float>(TargetHeight);
    const std::array<std::array<float, 2>, BandVertices> corners{{{-1.0f, top}, {1.0f, top}, {-1.0f, bottom}, {1.0f, top}, {1.0f, bottom}, {-1.0f, bottom}}};
    for (std::uint32_t vertex = 0; vertex < BandVertices; ++vertex) Vertices[band * BandVertices + vertex] = {corners[vertex][0], corners[vertex][1], 0.5f, 1.0f};
}

const Program& BandVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Indexed(BufferLoadDwordX4, 10, 5, 8)
              << WaitAll
              << Export(PositionTarget, 10, false)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& UserDataPixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Vop1(VMovB32, 6, 0)
              << Vop1(VMovB32, 7, 1)
              << Vop1(VMovB32, 8, 2)
              << Vop1(VMovB32, 9, 3)
              << Export(ColorTarget, 6, true)
              << EndProgram;
        return built;
    }();
    return program;
}

std::array<std::uint32_t, 4> BandColor(std::uint32_t band) {
    return {(0x10u + 7u * band) & 0xffu, (0x80u + 3u * band) & 0xffu, (0x20u + 11u * band) & 0xffu, (0xffu - 5u * band) & 0xffu};
}

std::array<std::uint32_t, VertexUserWords> BandUserData(std::uint32_t band) {
    const auto address = AddressOf(Vertices.data() + band * BandVertices);
    return {Low(address), (High(address) & 0xffffu) | (16u << 16u), BandVertices, 0x01016facu};
}

std::array<std::uint32_t, PixelUserWords> ColorUserData(std::uint32_t band) {
    const auto color = BandColor(band);
    std::array<std::uint32_t, PixelUserWords> words{};
    for (std::size_t component = 0; component < color.size(); ++component) words[component] = Bits(static_cast<float>(color[component]) / 255.0f);
    return words;
}

void Drawn() {
    for (std::uint32_t band = 0; band < Draws; ++band) FillBand(band);
    Pixels.fill(DeviceTable::Untouched);
    const auto before = Now();
    Commands commands;
    Baseline(commands, {Pixels.data(), TargetWidth, TargetHeight}, CenterAndPosition);
    Programs(commands, BandVertex(), VertexUserWords, UserDataPixel(), PixelUserWords);
    Instances(commands, 1u);
    for (std::uint32_t band = 0; band < Draws; ++band) {
        VertexUserData(commands, BandUserData(band));
        PixelUserData(commands, ColorUserData(band));
        DrawAuto(commands, BandVertices);
    }
    commands.Submit(0);
    NewPath::WaitIdle();
    const auto pixels = ReadBack(Pixels.data(), Pixels.size());
    for (std::uint32_t band = 0; band < Draws; ++band) {
        const auto color = BandColor(band);
        const auto expected = color[0] | (color[1] << 8u) | (color[2] << 16u) | (color[3] << 24u);
        std::size_t wrong = 0;
        std::string first;
        for (std::uint32_t y = band * BandRows; y < (band + 1u) * BandRows; ++y) {
            for (std::uint32_t x = 0; x < TargetWidth; ++x) {
                const auto pixel = pixels[y * TargetWidth + x];
                if (pixel == expected) continue;
                if (wrong++ == 0u) first = "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + Hex(pixel) + ", expected " + Hex(expected);
            }
        }
        Expect(wrong == 0u, "draws: draw " + std::to_string(band) + " colors its band with its own pixel user data: " + std::to_string(wrong) + " pixels differ, first " + first);
    }
    ExpectBuiltOnce(before, Draws, "draws: one vertex and pixel program pair drawn " + std::to_string(Draws) + " times with other user data each time");
}

void Run() {
    Cases({{"dispatches", Dispatched}, {"draws", Drawn}});
}

}

int main() {
    return DrawSubmit::Main("program resources once", Run);
}
