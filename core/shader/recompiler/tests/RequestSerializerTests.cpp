#include "ControlFlow/RequestSerializer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr char kHistoricalVersion21[] =
    "NVNQQRUAAAAFAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJAAAAAAAAAOQAAAAAAAAAAAAAAAAAAAAAAQEAAAEAAAAAAAAAAAAFAAAAAAAEAAAAAAAAABI0VngAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAALAAAAEAAAAEAAAAAAAAF4VjQSAQEBOTAAAAA=";
constexpr char kVersion21WithF16Tail[] =
    "NVNQQRUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAQA0DAAEB";

constexpr char kVersion22WithoutF16Tail[] =
    "NVNQQRYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAQA0DAAE=";

constexpr char kLayerVersion23[] =
    "NVNQQRcAAAAFAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gAB"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAJAAAAAAAAAOQAAAAAAAAAAAAAAAAAAAAAAQEDAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAkAAAANAAAAAQUKBQABAAAAAAAAAAAABQAAAAAABAAAAAAAAAABAwUHAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAACwAAABAAAABAAAAAAQYAAAAAAAAAAAAAAAAAADTvzasAAAA5"
    "AAAABAAAAAkAAAAAAXhWNBIBAQE5MAAAAAA=";

void Require(bool condition) {
    if (!condition) std::abort();
}

void InsertPixelDefaultsAfterLayer(std::vector<std::uint8_t>& bytes) {
    bytes.insert(bytes.begin() + 259u, {0x0fu, 0u, 0u, 0xffu, 0xffu, 0u, 0u, 0xffu, 0xffu, 0u, 0u, 0xffu, 0xffu, 0u, 0u, 0xffu, 0xffu, 0u, 0u, 0u, 0u, 0u, 0u});
}

std::vector<std::uint8_t> DecodeBase64(std::string_view text) {
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<std::uint8_t> bytes;
    std::uint32_t chunk = 0;
    std::size_t bits = 0;
    for (const char symbol : text) {
        if (symbol == '=') break;
        const auto value = alphabet.find(symbol);
        Require(value != std::string_view::npos);
        chunk = ((chunk << 6u) | static_cast<std::uint32_t>(value)) & 0xffffffu;
        bits += 6u;
        if (bits >= 8u) {
            bits -= 8u;
            bytes.push_back(static_cast<std::uint8_t>((chunk >> bits) & 0xffu));
        }
    }
    return bytes;
}

std::string EncodeBase64(const std::vector<std::uint8_t>& bytes) {
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string text;
    for (std::size_t index = 0; index < bytes.size(); index += 3u) {
        const std::size_t count = std::min<std::size_t>(3u, bytes.size() - index);
        std::uint32_t chunk = static_cast<std::uint32_t>(bytes[index]) << 16u;
        if (count > 1u) chunk |= static_cast<std::uint32_t>(bytes[index + 1u]) << 8u;
        if (count > 2u) chunk |= bytes[index + 2u];
        for (std::size_t symbol = 0; symbol < 4u; ++symbol) {
            text.push_back(symbol <= count ? alphabet[(chunk >> (18u - 6u * symbol)) & 0x3fu] : '=');
        }
    }
    return text;
}

}

int main() {
    const ShaderRecompiler::RequestSerializer serializer;

    const auto historical = serializer.Deserialize(kHistoricalVersion21);
    Require(!historical.request.target.float16DenormPreserve);

    const auto version21WithTail = serializer.Deserialize(kVersion21WithF16Tail);
    Require(version21WithTail.request.target.float16DenormPreserve);

    bool rejectedTruncatedVersion22 = false;
    try {
        static_cast<void>(serializer.Deserialize(kVersion22WithoutF16Tail));
    } catch (const std::runtime_error& error) {
        rejectedTruncatedVersion22 = std::string_view(error.what()) == "RequestSerializer: truncated data";
    }
    Require(rejectedTruncatedVersion22);

    struct CanonicalMain22 {
        bool pixel;
        bool tessellation;
        std::string_view encoded;
    };
    constexpr std::array<CanonicalMain22, 4> canonicalMain22{{
        {true, false,
            "NVNQQRYAAAAFAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gAB"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAJAAAAAAAAAOQAAAAAAAAAAAAAAAAAAAAAAQEDAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAACQAAAA0AAAABBQoFAAEAAAAAAAAAAAAFAAAAAAAEAAAAAAAAAAEDBQcAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAALAAAAEAAAAEAAAAABBgAAAAAAAAAAAAAAAAAANO/NqwAAADkA"
            "AAAEAAAACQAAAAABeFY0EgEBATkwAAAAAA=="
        },
        {true, true,
            "NVNQQRYAAAAFAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gAB"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAJAAAAAAAAAOQAAAAAAAAAAAAAAAAAAAAAAQEDAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAACQAAAA0AAAABBQoFAAEAAAAAAAAAAAAFAAAAAAAEAAAAAAAAAAEDBQcAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAALAAAAEAAAAEAAAAABBgAAAAAAAAAAAAAAAAEEAAAABAAAAAIA"
            "AAACAAAAAwAAAD8AAAAANO/NqwAAADkAAAAEAAAACQAAAAABeFY0EgEBATkwAAAAAA=="
        },
        {false, false,
            "NVNQQRYAAAABAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gAA"
            "AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQAAAA0AAAABBQoFAAEAAAAAAAAA"
            "AAAFAAAAAAAEAAAAAAAAAAEDBQcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAIAAAALAAAAEAAAAEAAAAABBgAAAAAAAAAAAAAAAAAANO/NqwAAADkAAAAEAAAACQAAAAABeFY0EgEBATkwAAAAAA=="
        },
        {false, true,
            "NVNQQRYAAAABAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gAA"
            "AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQAAAA0AAAABBQoFAAEAAAAAAAAA"
            "AAAFAAAAAAAEAAAAAAAAAAEDBQcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAIAAAALAAAAEAAAAEAAAAABBgAAAAAAAAAAAAAAAAEEAAAABAAAAAIAAAACAAAAAwAAAD8AAAAANO/NqwAAADkAAAAEAAAACQAA"
            "AAABeFY0EgEBATkwAAAAAA=="
        },
    }};
    for (const auto& fixture : canonicalMain22) {
        for (const bool preserve : {false, true}) {
            auto bytes = DecodeBase64(fixture.encoded);
            bytes.back() = static_cast<std::uint8_t>(preserve);
            const auto replay = serializer.Deserialize(EncodeBase64(bytes));
            const auto& request = replay.request;
            Require(request.context.pixel.has_value() == fixture.pixel && request.context.vertex.has_value());
            Require(request.context.vertex->fetchAttribReg == 9u && request.context.vertex->fetchBufferReg == 13u && request.context.vertex->fetchEmbedded && request.context.vertex->paClVsOutCntl == 0x00050a05u);
            Require(!fixture.pixel || (request.context.pixel->rasterPrimitive == ShaderRecompiler::RasterPrimitive::Rectangle && request.context.pixel->alphaToCoverage && request.context.pixel->stencilRefExportEnable && request.context.pixel->targetOutputMode[0] == 9u && request.context.pixel->targetExportMapping[0] == 0xe4u && !request.context.pixel->layerExported && !request.context.pixel->ditheredAlphaToCoverage && request.context.pixel->reversedBlend == 0u));
            Require(request.graphics.has_value() && request.graphics->firstUserSgpr == 6u && request.graphics->tessellation.has_value() == fixture.tessellation);
            Require(!fixture.tessellation || (request.graphics->tessellation->patchesPerGroup == 63u && request.graphics->tessellation->domain == 2u && request.graphics->tessellation->outputTopology == 3u));
            Require(request.graphics->draw.indexAddress == 0xabcdef3400u && request.graphics->draw.indexCount == 57u && request.graphics->draw.indexElementBytes == 4u && request.graphics->draw.instanceCount == 9u);
            Require(request.target.float16DenormPreserve == preserve);
            auto upgraded = bytes;
            upgraded[4] = 34u;
            if (fixture.pixel) upgraded.insert(upgraded.begin() + 258u, 0u);
            if (fixture.pixel) InsertPixelDefaultsAfterLayer(upgraded);
            if (fixture.tessellation) upgraded.insert(upgraded.end() - 35u, 5u, 0u);
            upgraded.insert(upgraded.end(), 2u * sizeof(std::uint64_t) + 1u, 0u);
            upgraded.push_back(1u);
            Require(serializer.Serialize(request) == EncodeBase64(upgraded));
            auto truncated = bytes;
            truncated.pop_back();
            bool rejected = false;
            try {
                static_cast<void>(serializer.Deserialize(EncodeBase64(truncated)));
            } catch (const std::runtime_error& error) {
                rejected = std::string_view(error.what()) == "RequestSerializer: truncated data";
            }
            Require(rejected);
            if (!fixture.pixel) {
                auto vertex23 = bytes;
                vertex23[4] = 23u;
                const auto vertexReplay = serializer.Deserialize(EncodeBase64(vertex23));
                Require(serializer.Serialize(vertexReplay.request) == EncodeBase64(upgraded));
                continue;
            }
            for (const bool layer : {false, true}) {
                auto layered = bytes;
                layered[4] = 23u;
                layered.insert(layered.begin() + 258u, static_cast<std::uint8_t>(layer));
                const auto layerReplay = serializer.Deserialize(EncodeBase64(layered));
                Require(layerReplay.request.context.pixel->layerExported == layer && !layerReplay.request.context.pixel->ditheredAlphaToCoverage && layerReplay.request.context.pixel->reversedBlend == 0u);
                Require(layerReplay.request.target.float16DenormPreserve == preserve && layerReplay.request.graphics->tessellation.has_value() == fixture.tessellation);
                Require(!fixture.tessellation || layerReplay.request.graphics->tessellation->patchesPerGroup == 63u);
                auto truncatedLayer = layered;
                truncatedLayer.pop_back();
                bool layerRejected = false;
                try {
                    static_cast<void>(serializer.Deserialize(EncodeBase64(truncatedLayer)));
                } catch (const std::runtime_error& error) {
                    layerRejected = std::string_view(error.what()) == "RequestSerializer: truncated data";
                }
                Require(layerRejected);
                layered[4] = 34u;
                InsertPixelDefaultsAfterLayer(layered);
                if (fixture.tessellation) layered.insert(layered.end() - 35u, 5u, 0u);
                layered.insert(layered.end(), 2u * sizeof(std::uint64_t) + 1u, 0u);
                layered.push_back(1u);
                Require(serializer.Serialize(layerReplay.request) == EncodeBase64(layered));
            }
        }
    }

    const auto layer23 = DecodeBase64(kLayerVersion23);
    Require(layer23.size() == 1451u);
    for (const bool layer : {false, true}) {
        for (const bool preserve : {false, true}) {
            auto bytes = layer23;
            bytes[258] = static_cast<std::uint8_t>(layer);
            bytes.back() = static_cast<std::uint8_t>(preserve);
            const auto encoded23 = EncodeBase64(bytes);
            const auto replay = serializer.Deserialize(encoded23);
            Require(replay.request.context.pixel.has_value() && replay.request.context.vertex.has_value());
            Require(replay.request.context.pixel->layerExported == layer);
            Require(!replay.request.context.pixel->ditheredAlphaToCoverage && replay.request.context.pixel->reversedBlend == 0u);
            Require(replay.request.context.pixel->rasterPrimitive == ShaderRecompiler::RasterPrimitive::Rectangle);
            Require(replay.request.context.vertex->paClVsOutCntl == 0x00050a05u);
            Require(replay.request.target.float16DenormPreserve == preserve);
            bytes[4] = 34u;
            InsertPixelDefaultsAfterLayer(bytes);
            bytes.insert(bytes.end(), 2u * sizeof(std::uint64_t) + 1u, 0u);
            bytes.push_back(1u);
            Require(serializer.Serialize(replay.request) == EncodeBase64(bytes));

            auto truncated23 = DecodeBase64(encoded23);
            truncated23.pop_back();
            bool rejected = false;
            try {
                static_cast<void>(serializer.Deserialize(EncodeBase64(truncated23)));
            } catch (const std::runtime_error& error) {
                rejected = std::string_view(error.what()) == "RequestSerializer: truncated data";
            }
            Require(rejected);
        }
    }

    for (const auto& fixture : canonicalMain22) {
        const auto replay = serializer.Deserialize(fixture.encoded);
        for (const bool dither : {false, true}) {
            auto modern = replay.request;
            if (fixture.tessellation) {
                modern.graphics->tessellation->primitiveGroupSize = 37u;
                modern.graphics->tessellation->breakAtInstance = true;
            }
            if (fixture.pixel) {
                modern.context.pixel->layerExported = dither;
                modern.context.pixel->ditheredAlphaToCoverage = dither;
                modern.context.pixel->frontFaceAllBits = !dither;
                modern.context.pixel->quadPixelMask = 0x5u;
                modern.context.pixel->quadSampleMask = {0x1u, 0x2u, 0x4u, 0x8u};
                modern.context.pixel->reversedBlend = 0x3u;
            }
            const auto encoded = serializer.Serialize(modern);
            const auto back = serializer.Deserialize(encoded);
            Require(back.request.graphics.has_value() && back.request.graphics->tessellation.has_value() == fixture.tessellation);
            Require(!fixture.tessellation || (back.request.graphics->tessellation->patchesPerGroup == 63u && back.request.graphics->tessellation->primitiveGroupSize == 37u && back.request.graphics->tessellation->breakAtInstance));
            Require(back.request.context.pixel.has_value() == fixture.pixel);
            if (fixture.pixel) {
                const auto& pixel = *back.request.context.pixel;
                Require(pixel.layerExported == dither && pixel.ditheredAlphaToCoverage == dither && pixel.frontFaceAllBits == !dither && pixel.quadPixelMask == 0x5u);
                Require(pixel.quadSampleMask[0] == 0x1u && pixel.quadSampleMask[1] == 0x2u && pixel.quadSampleMask[2] == 0x4u && pixel.quadSampleMask[3] == 0x8u && pixel.reversedBlend == 0x3u);
            }
            Require(serializer.Serialize(back.request) == encoded);
        }
    }

    for (const std::uint8_t colliding : {24u, 25u, 26u}) {
        auto bytes = layer23;
        bytes[4] = colliding;
        if (colliding >= 25u) bytes.insert(bytes.begin() + 259u, 0x5u);
        bytes.insert(bytes.end(), 2u * sizeof(std::uint64_t), 0u);
        bool refused = false;
        try {
            static_cast<void>(serializer.Deserialize(EncodeBase64(bytes)));
        } catch (const std::runtime_error& error) {
            refused = std::string_view(error.what()) == "request version " + std::to_string(colliding) + " has two layouts; re-record it";
        }
        Require(refused);
        const auto own = serializer.Deserialize(EncodeBase64(bytes), ShaderRecompiler::CollidingVersions::ReadAsOwn);
        Require(own.request.context.pixel.has_value() && own.request.context.pixel->quadPixelMask == (colliding >= 25u ? 0x5u : 0x0fu));
        Require(own.linkedLiterals.empty() && own.linkedWords.empty());
        const auto migrated = serializer.Serialize(own.request);
        Require(DecodeBase64(migrated)[4] == 34u && serializer.Serialize(serializer.Deserialize(migrated).request) == migrated);
    }

    for (const auto& fixture : canonicalMain22) {
        const auto base = DecodeBase64(fixture.encoded);
        const auto source = serializer.Deserialize(fixture.encoded);
        auto modern = source.request;
        if (fixture.tessellation) {
            modern.graphics->tessellation->primitiveGroupSize = 37u;
            modern.graphics->tessellation->breakAtInstance = true;
        }
        for (const std::uint32_t version : {27u, 28u, 29u, 30u, 31u}) {
            auto expected = modern;
            auto upstream = base;
            upstream[4] = static_cast<std::uint8_t>(version);
            if (fixture.pixel) {
                auto& pixel = *expected.context.pixel;
                pixel.layerExported = true;
                pixel.ditheredAlphaToCoverage = true;
                pixel.frontFaceAllBits = true;
                std::vector<std::uint8_t> fields{1u, 1u, 1u};
                if (version >= 28u) {
                    pixel.quadPixelMask = 0x5u;
                    fields.push_back(0x5u);
                }
                if (version >= 29u) {
                    pixel.quadSampleMask = {0x1u, 0x2u, 0x4u, 0x8u};
                    for (const std::uint8_t mask : {0x1u, 0x2u, 0x4u, 0x8u}) fields.insert(fields.end(), {mask, 0u, 0u, 0u});
                }
                if (version >= 30u) {
                    pixel.reversedBlend = 0x3u;
                    fields.insert(fields.end(), {0x3u, 0u, 0u, 0u});
                }
                upstream.insert(upstream.begin() + 258u, fields.begin(), fields.end());
            }
            if (fixture.tessellation) upstream.insert(upstream.end() - 35u, {37u, 0u, 0u, 0u, 1u});
            upstream.insert(upstream.end(), sizeof(std::uint64_t), 0u);
            const auto replay = serializer.Deserialize(EncodeBase64(upstream));
            Require(replay.request.context.pixel.has_value() == fixture.pixel && replay.request.graphics.has_value() && replay.request.graphics->tessellation.has_value() == fixture.tessellation);
            if (fixture.pixel) {
                const auto& pixel = *replay.request.context.pixel;
                Require(pixel.layerExported && pixel.ditheredAlphaToCoverage && pixel.frontFaceAllBits);
                Require(pixel.quadPixelMask == expected.context.pixel->quadPixelMask && pixel.quadSampleMask == expected.context.pixel->quadSampleMask && pixel.reversedBlend == expected.context.pixel->reversedBlend);
            }
            Require(!fixture.tessellation || (replay.request.graphics->tessellation->patchesPerGroup == 63u && replay.request.graphics->tessellation->primitiveGroupSize == 37u && replay.request.graphics->tessellation->breakAtInstance));
            Require(replay.linkedLiterals.empty() && replay.linkedWords.empty());
            Require(serializer.Serialize(replay.request) == serializer.Serialize(expected));
            auto truncated = upstream;
            truncated.pop_back();
            bool rejected = false;
            try {
                static_cast<void>(serializer.Deserialize(EncodeBase64(truncated)));
            } catch (const std::runtime_error& error) {
                rejected = std::string_view(error.what()) == "RequestSerializer: truncated data";
            }
            Require(rejected);
            auto fallbacks = upstream;
            fallbacks[fallbacks.size() - sizeof(std::uint64_t)] = 1u;
            fallbacks.insert(fallbacks.end(), {0x40u, 0u, 0u, 0u});
            bool fallbacksRejected = false;
            try {
                static_cast<void>(serializer.Deserialize(EncodeBase64(fallbacks)));
            } catch (const std::runtime_error& error) {
                fallbacksRejected = std::string_view(error.what()) == "unsupported recompile request: upstream request version " + std::to_string(version) + " carries linked call fallbacks";
            }
            Require(fallbacksRejected);
        }
    }

    ShaderRecompiler::RecompileRequest request{};
    for (const bool preserve : {false, true}) {
        request.target.float16DenormPreserve = preserve;
        const auto encoded = serializer.Serialize(request);
        const auto replay = serializer.Deserialize(encoded);
        Require(replay.request.target.float16DenormPreserve == preserve);
        auto bytes = DecodeBase64(encoded);
        Require(bytes.size() >= 8u && bytes[4] == 34u && bytes[5] == 0u && bytes[6] == 0u && bytes[7] == 0u);
        for (const std::uint8_t refusedVersion : {0u, 35u, 36u}) {
            auto refused = bytes;
            refused[4] = refusedVersion;
            bool versionRejected = false;
            try {
                static_cast<void>(serializer.Deserialize(EncodeBase64(refused)));
            } catch (const std::runtime_error& error) {
                versionRejected = std::string_view(error.what()) == "unsupported recompile request serialization version " + std::to_string(refusedVersion);
            }
            Require(versionRejected);
        }
        bytes.pop_back();
        bool rejected = false;
        try {
            static_cast<void>(serializer.Deserialize(EncodeBase64(bytes)));
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()) == "RequestSerializer: truncated data";
        }
        Require(rejected);
    }

    ShaderRecompiler::RecompileRequest split{};
    const std::array<std::uint32_t, 4> code{0xbf820001u, 0xbe8e210eu, 0xbf810000u, 0xbf9f0000u};
    split.shader.stage = ShaderRecompiler::ShaderStage::Compute;
    split.shader.code = code;
    split.context.waveSize = 32u;
    split.context.compute = ShaderRecompiler::ShaderComputeStageInfo{{32u, 1u, 1u}, 1152u, {true, false, false}, false, 1u, {}};
    const std::array<ShaderRecompiler::ContinuationAbi::BinSite, 2> sites{{{4u, 0x1fe0u}, {0x40u, 0x2400u}}};
    const std::array<std::uint32_t, 3> scalars{0u, 1u, 16u};
    const std::array<std::uint32_t, 2> vectors{0u, 143u};
    const std::array<std::uint32_t, 2> userDataScalars{0u, 1u};
    split.context.continuation = ShaderRecompiler::ContinuationAbi::Shape{ShaderRecompiler::ContinuationAbi::Role::Continuation, sites, scalars, vectors, 1152u, userDataScalars, 1u};
    const auto splitEncoded = serializer.Serialize(split);
    const auto splitBack = serializer.Deserialize(splitEncoded);
    Require(splitBack.request.context.continuation.has_value());
    const auto& shape = *splitBack.request.context.continuation;
    Require(shape.role == ShaderRecompiler::ContinuationAbi::Role::Continuation && shape.ldsDwords == 1152u);
    Require(std::ranges::equal(shape.sites, sites) && std::ranges::equal(shape.scalarRegisters, scalars) && std::ranges::equal(shape.vectorRegisters, vectors));
    Require(std::ranges::equal(shape.userDataScalars, userDataScalars) && shape.entrySite == 1u);
    Require(serializer.Serialize(splitBack.request) == splitEncoded);
    auto splitBytes = DecodeBase64(splitEncoded);
    splitBytes[splitBytes.size() - 4u - (8u + 2u * 4u) - 4u - (8u + 2u * 4u) - (8u + 3u * 4u) - (8u + 2u * 15u * 4u) - 4u] = 7u;
    bool roleRejected = false;
    try {
        static_cast<void>(serializer.Deserialize(EncodeBase64(splitBytes)));
    } catch (const std::runtime_error& error) {
        roleRejected = std::string_view(error.what()) == "RequestSerializer: invalid continuation role";
    }
    Require(roleRejected);

    return 0;
}
