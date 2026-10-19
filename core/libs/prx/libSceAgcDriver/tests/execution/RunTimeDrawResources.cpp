#include "DrawSubmitFixture.hpp"
#include "WrittenImageFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

namespace {

using namespace DrawSubmit;
using AgcDriver::Graphics::TextureTileMode;

constexpr std::uint32_t TargetWidth = 64;
constexpr std::uint32_t TargetHeight = 32;
constexpr std::uint32_t Draws = 16;
constexpr std::uint32_t BandRows = TargetHeight / Draws;
constexpr std::uint32_t BandVertices = 6;
constexpr std::uint32_t WrittenTexture = Draws;
constexpr std::uint32_t TextureCount = Draws + 1;
constexpr std::uint32_t TextureSlotWords = TargetWidth * TargetHeight * 2;
constexpr std::uint32_t LinkWords = 64;
constexpr std::uint32_t SamplerWord = 4;
constexpr std::uint32_t OneOverTargetWidth = 0x3c800000u;
constexpr std::uint32_t OneOverTargetHeight = 0x3d000000u;
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t PixelUserWords = 4;
constexpr std::uint32_t ConstantPixelUserWords = 8;
constexpr std::uint32_t Format16_16_16_16Float = 71;
constexpr std::array<std::uint32_t, 3> Formats{DeviceTable::Format8_8_8_8UNorm, DeviceTable::Format32Float, Format16_16_16_16Float};

TestGuest::Array<std::uint32_t, TextureCount * TextureSlotWords> TextureMemory;
TestGuest::Array<std::uint32_t, TextureCount * LinkWords> Roots;
TestGuest::Array<std::uint32_t, TextureCount * LinkWords> Tables;
TestGuest::Array<std::uint32_t, Draws * LinkWords> Constants;
TestGuest::Array<std::array<float, 4>, Draws * BandVertices> Vertices;
TestGuest::Array<std::uint32_t, TargetWidth * TargetHeight> Pixels;

Header VertexHeader;
Header PixelHeader;
Header ConstantPixelHeader;

std::uint32_t FormatOf(std::uint32_t texture) {
    return Formats[texture % Formats.size()];
}

std::string FormatName(std::uint32_t format) {
    switch (format) {
    case DeviceTable::Format8_8_8_8UNorm: return "8_8_8_8 unorm";
    case DeviceTable::Format32Float: return "32 float";
    default: return "16_16_16_16 float";
    }
}

std::uint32_t Channel(std::uint32_t texture, std::uint32_t x, std::uint32_t y, std::uint32_t component) {
    return (0x07u + 23u * texture + 5u * x + 37u * y + 71u * component) & 0xffu;
}

std::uint32_t Blue(std::uint32_t resources) {
    return (0x30u + 13u * resources) & 0xffu;
}

std::uint32_t Alpha(std::uint32_t resources) {
    return (0xf0u - 11u * resources) & 0xffu;
}

float Unorm(std::uint32_t byte) {
    return static_cast<float>(byte) / 255.0f;
}

std::uint16_t HalfOf(float value) {
    const auto bits = Bits(value);
    if ((bits & 0x7fffffffu) == 0u) return 0u;
    const auto exponent = static_cast<std::uint32_t>(static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15);
    return static_cast<std::uint16_t>((exponent << 10u) | ((bits >> 13u) & 0x3ffu));
}

std::byte* TextureData(std::uint32_t texture) {
    return reinterpret_cast<std::byte*>(TextureMemory.data() + texture * TextureSlotWords);
}

void FillTexture(std::uint32_t texture) {
    const auto format = FormatOf(texture);
    const auto bytes = format == Format16_16_16_16Float ? 8u : 4u;
    const auto layout = AgcDriver::Graphics::ComputeMipLayout(TextureTileMode::kLinear, format, TargetWidth, TargetHeight, 1u).at(0);
    AgcDriver::Graphics::Require(layout.tiledOffset + layout.tiledSize <= TextureSlotWords * 4u, "run-time draw resources: a texture does not fit its slot");
    auto* base = TextureData(texture) + layout.tiledOffset;
    for (std::uint32_t y = 0; y < TargetHeight; ++y) {
        for (std::uint32_t x = 0; x < TargetWidth; ++x) {
            auto* texel = base + y * layout.pitchBytes + x * bytes;
            if (format == DeviceTable::Format8_8_8_8UNorm) {
                const auto word = Channel(texture, x, y, 0) | (Channel(texture, x, y, 1) << 8u) | (Channel(texture, x, y, 2) << 16u) | (Channel(texture, x, y, 3) << 24u);
                std::memcpy(texel, &word, sizeof(word));
            } else if (format == DeviceTable::Format32Float) {
                const auto value = Unorm(Channel(texture, x, y, 0));
                std::memcpy(texel, &value, sizeof(value));
            } else {
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto half = HalfOf(Unorm(Channel(texture, x, y, component)));
                    std::memcpy(texel + component * 2u, &half, sizeof(half));
                }
            }
        }
    }
}

std::array<std::uint32_t, 8> TextureDescriptor(std::uint32_t texture) {
    const auto address = AddressOf(TextureData(texture));
    return {
        Low(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (FormatOf(texture) << 20u) | (((TargetWidth - 1u) & 3u) << 30u),
        ((TargetWidth - 1u) >> 2u) | ((TargetHeight - 1u) << 14u) | (1u << 31u),
        0xfacu | (DeviceTable::Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::uint32_t* RootOf(std::uint32_t root) {
    return Roots.data() + root * LinkWords;
}

std::uint32_t* TableOf(std::uint32_t root) {
    return Tables.data() + root * LinkWords;
}

void Link(std::uint32_t root, const std::array<std::uint32_t, 8>& descriptor) {
    auto* words = RootOf(root);
    std::fill(words, words + LinkWords, 0u);
    const auto table = AddressOf(TableOf(root));
    words[0] = Low(table);
    words[1] = High(table);
    std::copy(DeviceTable::PointClamp.begin(), DeviceTable::PointClamp.end(), words + SamplerWord);
    std::fill(TableOf(root), TableOf(root) + LinkWords, 0u);
    std::copy(descriptor.begin(), descriptor.end(), TableOf(root));
}

void FillBand(std::uint32_t band) {
    const auto top = 1.0f - 2.0f * static_cast<float>(band * BandRows) / static_cast<float>(TargetHeight);
    const auto bottom = 1.0f - 2.0f * static_cast<float>((band + 1u) * BandRows) / static_cast<float>(TargetHeight);
    const std::array<std::array<float, 2>, BandVertices> corners{{{-1.0f, top}, {1.0f, top}, {-1.0f, bottom}, {1.0f, top}, {1.0f, bottom}, {-1.0f, bottom}}};
    for (std::uint32_t vertex = 0; vertex < BandVertices; ++vertex) Vertices[band * BandVertices + vertex] = {corners[vertex][0], corners[vertex][1], 0.5f, 1.0f};
}

void FillConstants(std::uint32_t resources) {
    auto* words = Constants.data() + resources * LinkWords;
    std::fill(words, words + LinkWords, 0u);
    words[0] = Bits(Unorm(Alpha(resources)));
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

void SampleTexel(Program& built) {
    using namespace DrawSubmit::Isa;
    built << Smem(SLoadDwordX8, 36, 20, 0)
          << Vop2(VMulF32, 4, Literal, 2) << OneOverTargetWidth
          << Vop2(VMulF32, 5, Literal, 3) << OneOverTargetHeight
          << WaitAll
          << Mimg(ImageSampleLz, 0xfu, false, 6, 4, 36, 32)
          << WaitAll;
}

const Program& SamplePixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Smem(SLoadDwordX2, 20, 0, 0)
              << Smem(SLoadDwordX4, 32, 0, SamplerWord * 4u)
              << WaitAll;
        SampleTexel(built);
        built << Vop1(VMovB32, 8, 2)
              << Vop1(VMovB32, 9, 3)
              << Export(ColorTarget, 6, true)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& ConstantSamplePixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Smem(SLoadDwordX2, 20, 0, 0)
              << Smem(SLoadDwordX4, 32, 0, SamplerWord * 4u)
              << Smem(SBufferLoadDword, 24, 4, 0)
              << WaitAll;
        SampleTexel(built);
        built << Vop1(VMovB32, 8, 2)
              << Vop1(VMovB32, 9, 24)
              << Export(ColorTarget, 6, true)
              << EndProgram;
        return built;
    }();
    return program;
}

std::array<std::uint32_t, VertexUserWords> BandUserData(std::uint32_t band) {
    const auto address = AddressOf(Vertices.data() + band * BandVertices);
    return {Low(address), (High(address) & 0xffffu) | (16u << 16u), BandVertices, 0x01016facu};
}

std::array<std::uint32_t, PixelUserWords> SampleUserData(std::uint32_t root, std::uint32_t resources) {
    const auto address = AddressOf(RootOf(root));
    return {Low(address), High(address), Bits(Unorm(Blue(resources))), Bits(Unorm(Alpha(resources)))};
}

std::array<std::uint32_t, ConstantPixelUserWords> ConstantUserData(std::uint32_t root, std::uint32_t resources) {
    const auto address = AddressOf(RootOf(root));
    const auto constants = DeviceTable::Buffer(Constants.data() + resources * LinkWords, 16u);
    return {Low(address), High(address), Bits(Unorm(Blue(resources))), 0u, constants[0], constants[1], constants[2], constants[3]};
}

std::uint32_t Expected(std::uint32_t texture, std::uint32_t resources, std::uint32_t x, std::uint32_t y) {
    const auto green = FormatOf(texture) == DeviceTable::Format32Float ? 0u : Channel(texture, x, y, 1);
    return Channel(texture, x, y, 0) | (green << 8u) | (Blue(resources) << 16u) | (Alpha(resources) << 24u);
}

struct Band {
    std::uint32_t texture;
    std::uint32_t resources;
};

void ExpectBands(const std::function<Band(std::uint32_t band)>& bands, const std::string& what) {
    const auto pixels = ReadBack(Pixels.data(), Pixels.size());
    for (std::uint32_t band = 0; band < Draws; ++band) {
        const auto [texture, resources] = bands(band);
        std::size_t wrong = 0;
        std::size_t untouched = 0;
        std::string first;
        for (std::uint32_t y = band * BandRows; y < (band + 1u) * BandRows; ++y) {
            for (std::uint32_t x = 0; x < TargetWidth; ++x) {
                const auto pixel = pixels[y * TargetWidth + x];
                const auto expected = Expected(texture, resources, x, y);
                if (pixel == expected) continue;
                untouched += pixel == DeviceTable::Untouched ? 1u : 0u;
                if (wrong++ == 0u) first = "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + Hex(pixel) + ", expected " + Hex(expected);
            }
        }
        Expect(wrong == 0u, what + ": draw " + std::to_string(band) + " (texture " + std::to_string(texture) + ", " + FormatName(FormatOf(texture)) + ", user data and buffers " + std::to_string(resources) + "): " + std::to_string(wrong) + " of " + std::to_string(BandRows * TargetWidth) + " pixels differ (" + std::to_string(untouched) + " untouched), first " + first);
    }
}

void Begin(Commands& commands, const Program& pixel, std::uint32_t pixelUserWords) {
    Baseline(commands, {Pixels.data(), TargetWidth, TargetHeight}, CenterAndPosition);
    Programs(commands, BandVertex(), VertexUserWords, pixel, pixelUserWords);
    Instances(commands, 1u);
}

void DrawBand(Commands& commands, std::uint32_t band, std::span<const std::uint32_t> pixelUserData) {
    VertexUserData(commands, BandUserData(band));
    PixelUserData(commands, pixelUserData);
    DrawAuto(commands, BandVertices);
}

std::uint32_t Shuffled(std::uint32_t band) {
    return (5u * band + 3u) % Draws;
}

void TexturesPerDraw() {
    Pixels.fill(DeviceTable::Untouched);
    const auto before = StageCompiles();
    Commands commands;
    Begin(commands, SamplePixel(), PixelUserWords);
    for (std::uint32_t band = 0; band < Draws; ++band) DrawBand(commands, band, SampleUserData(band, band));
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectBands([](std::uint32_t band) { return Band{band, band}; }, "textures per draw: 16 draws of one program pair in one submission, each with its own vertex buffer, T# chain in guest memory and user data");
    ExpectStageCompiles(before, 2u, "textures per draw: 16 draws of one vertex and pixel program pair");

    Pixels.fill(DeviceTable::Untouched);
    const auto again = StageCompiles();
    Commands shuffled;
    Begin(shuffled, SamplePixel(), PixelUserWords);
    for (std::uint32_t band = 0; band < Draws; ++band) DrawBand(shuffled, band, SampleUserData(Shuffled(band), Shuffled(band)));
    shuffled.Submit(0);
    NewPath::WaitIdle();
    ExpectBands([](std::uint32_t band) { return Band{Shuffled(band), Shuffled(band)}; }, "textures per draw, shuffled: the next submission draws each band with another draw's T# chain and user data");
    ExpectStageCompiles(again, 0u, "textures per draw, shuffled: the same program pair and state");
}

void ConstantBuffersPerDraw() {
    Pixels.fill(DeviceTable::Untouched);
    const auto before = StageCompiles();
    Commands commands;
    Begin(commands, ConstantSamplePixel(), ConstantPixelUserWords);
    for (std::uint32_t band = 0; band < Draws; ++band) DrawBand(commands, band, ConstantUserData(Shuffled(band), band));
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectBands([](std::uint32_t band) { return Band{Shuffled(band), band}; }, "constant buffers per draw: 16 draws whose pixel program reads alpha from its own constant buffer V# through a scalar buffer load");
    ExpectStageCompiles(before, 1u, "constant buffers per draw: a new pixel program with the vertex program drawn before");
}

std::array<std::uint32_t, 12> WriterUserData(std::uint32_t root, std::uint32_t texture) {
    std::array<std::uint32_t, 12> userData{};
    const auto table = DeviceTable::Buffer(TableOf(root), 32u);
    const auto words = TextureDescriptor(texture);
    std::copy(table.begin(), table.end(), userData.begin());
    std::copy(words.begin(), words.begin() + 8, userData.begin() + 4);
    return userData;
}

void WrittenDescriptor() {
    constexpr std::uint32_t root = WrittenTexture;
    Link(root, TextureDescriptor(0));
    Pixels.fill(DeviceTable::Untouched);
    const auto before = StageCompiles();
    Commands commands;
    commands.Dispatch(WrittenImages::WriterProgram(), WriterUserData(root, WrittenTexture), DeviceTable::Lanes, 1);
    commands.Barrier();
    Begin(commands, SamplePixel(), PixelUserWords);
    for (std::uint32_t band = 0; band < Draws; ++band) DrawBand(commands, band, SampleUserData(root, band));
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectBands([](std::uint32_t band) { return Band{WrittenTexture, band}; }, "written descriptor: draws sampling through a table whose T# a compute dispatch stored earlier in the same submission");
    ExpectStageCompiles(before, 0u, "written descriptor: the program pair drawn before");
    const auto stored = ReadBack(TableOf(root), 8u);
    const auto written = TextureDescriptor(WrittenTexture);
    Expect(std::equal(written.begin(), written.end(), stored.begin()), "written descriptor: the table in guest memory does not hold the T# the dispatch stored");

    Pixels.fill(DeviceTable::Untouched);
    Commands again;
    Begin(again, SamplePixel(), PixelUserWords);
    for (std::uint32_t band = 0; band < Draws; ++band) DrawBand(again, band, SampleUserData(root, Shuffled(band)));
    again.Submit(0);
    NewPath::WaitIdle();
    ExpectBands([](std::uint32_t band) { return Band{WrittenTexture, Shuffled(band)}; }, "written descriptor: the next submission's draws through the stored T#");
}

void WrittenDescriptorDeferred() {
    constexpr std::uint32_t root = WrittenTexture;
    Link(root, TextureDescriptor(0));
    Pixels.fill(DeviceTable::Untouched);
    const auto waitsBefore = AgcDriver::GuestMemory::PendingReadWaitCounts().reads;
    const auto deferredBefore = AgcDriver::DriverDetail::Driver::Get().DeferredDispatches();
    Commands commands;
    commands.Dispatch(WrittenImages::WriterProgram(), WriterUserData(root, WrittenTexture), DeviceTable::Lanes, 1);
    commands.Barrier();
    Begin(commands, SamplePixel(), PixelUserWords);
    for (std::uint32_t band = 0; band < Draws; ++band) DrawBand(commands, band, SampleUserData(root, band));
    commands.Submit(0);
    NewPath::WaitIdle();
    const auto waits = AgcDriver::GuestMemory::PendingReadWaitCounts().reads - waitsBefore;
    const auto deferred = AgcDriver::DriverDetail::Driver::Get().DeferredDispatches() - deferredBefore;
    Expect(waits == 0u, "written descriptor, deferred: the recording thread waited for the writer " + std::to_string(waits) + " times");
    Expect(deferred != 0u, "written descriptor, deferred: the first draw was not deferred");
    ExpectBands([](std::uint32_t band) { return Band{WrittenTexture, band}; }, "written descriptor, deferred: draws recorded after the dispatch that stores their T# completed");
}

void Run() {
    for (std::uint32_t texture = 0; texture < TextureCount; ++texture) FillTexture(texture);
    for (std::uint32_t draw = 0; draw < Draws; ++draw) {
        Link(draw, TextureDescriptor(draw));
        FillBand(draw);
        FillConstants(draw);
    }
    Register(VertexHeader, BandVertex(), VertexShaderType);
    Register(PixelHeader, SamplePixel(), PixelShaderType);
    Register(ConstantPixelHeader, ConstantSamplePixel(), PixelShaderType);
    Cases({
        {"textures per draw", TexturesPerDraw},
        {"constant buffers per draw", ConstantBuffersPerDraw},
        {"written descriptor", WrittenDescriptor},
        {"written descriptor, deferred", WrittenDescriptorDeferred},
    });
}

}

int main() {
    return DrawSubmit::Main("run time draw resources", Run);
}
