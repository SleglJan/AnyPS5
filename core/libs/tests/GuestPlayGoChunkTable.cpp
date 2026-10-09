#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

extern "C" {
int APS5_VABI scePlayGoInitialize(const PlayGoInitParams*);
int APS5_VABI scePlayGoOpen(int*, const void*);
int APS5_VABI scePlayGoClose(int);
int APS5_VABI scePlayGoGetChunkId(int, std::uint16_t*, std::uint32_t, std::uint32_t*);
int APS5_VABI scePlayGoGetLocus(int, const std::uint16_t*, std::uint32_t, std::int8_t*);
}

namespace {

constexpr int BadChunkId = static_cast<int>(0x80B2000C);

template <typename TCall>
bool Throws(TCall call) {
    try {
        call();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void Put(std::vector<char>& data, std::size_t offset, std::uint32_t value, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i) data[offset + i] = static_cast<char>((value >> (8u * i)) & 0xffu);
}

std::vector<char> ChunkTable(std::uint32_t count) {
    std::vector<char> data(0x100 + count * 32u + 16u, '\0');
    std::memcpy(data.data(), "plgx", 4);
    Put(data, 0x04, 0x1000, 2);
    Put(data, 0x0a, count, 2);
    Put(data, 0x10, static_cast<std::uint32_t>(data.size()), 4);
    Put(data, 0xc0, 0x100, 4);
    Put(data, 0xc4, count * 32u, 4);
    return data;
}

void Write(const std::filesystem::path& path, const std::vector<char>& data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
}

bool Chunks(int handle, const std::vector<std::uint16_t>& expected) {
    std::uint32_t entries = 0;
    if (scePlayGoGetChunkId(handle, nullptr, 0, &entries) != 0 || entries != expected.size()) return false;
    std::vector<std::uint16_t> ids(expected.size(), 0xffff);
    return scePlayGoGetChunkId(handle, ids.data(), static_cast<std::uint32_t>(ids.size()), &entries) == 0 && entries == expected.size() && ids == expected;
}

bool TableMode(const std::filesystem::path& table, int handle) {
    std::int8_t locus = 0;
    const std::uint16_t last = 39;
    const std::uint16_t past = 40;
    std::vector<std::vector<char>> malformed;
    malformed.emplace_back(ChunkTable(40));
    malformed.back().resize(0x80);
    Put(malformed.back(), 0x10, 0x80, 4);
    malformed.emplace_back(ChunkTable(40));
    std::memcpy(malformed.back().data(), "plgo", 4);
    malformed.emplace_back(ChunkTable(40));
    Put(malformed.back(), 0x10, static_cast<std::uint32_t>(malformed.back().size() + 16u), 4);
    malformed.emplace_back(ChunkTable(40));
    Put(malformed.back(), 0x0a, 0, 2);
    Put(malformed.back(), 0xc4, 0, 4);
    malformed.emplace_back(ChunkTable(40));
    Put(malformed.back(), 0xc4, 39u * 32u, 4);
    malformed.emplace_back(ChunkTable(40));
    Put(malformed.back(), 0xc0, 0x120, 4);
    malformed.emplace_back(ChunkTable(40));
    Put(malformed.back(), 0xc0, 0x40, 4);
    for (const auto& data : malformed) {
        Write(table, data);
        if (!Throws([&] { scePlayGoGetLocus(handle, &last, 1, &locus); })) return false;
    }
    Write(table, ChunkTable(40));
    std::vector<std::uint16_t> expected(40);
    for (std::uint16_t id = 0; id < 40; ++id) expected[id] = id;
    return Chunks(handle, expected) && scePlayGoGetLocus(handle, &last, 1, &locus) == 0 && locus == 3 && scePlayGoGetLocus(handle, &past, 1, &locus) == BadChunkId;
}

bool XmlMode(const std::filesystem::path& table, const std::filesystem::path& app0, int handle) {
    Write(table, ChunkTable(40));
    const std::string xml = R"(<playgo default_chunk="2"><chunk id="5"/></playgo>)";
    Write(app0 / "playgo-chunkdefs.xml", std::vector<char>(xml.begin(), xml.end()));
    std::int8_t locus = 0;
    const std::uint16_t five = 5;
    const std::uint16_t thirtyNine = 39;
    return Chunks(handle, {0, 1, 2, 5}) && scePlayGoGetLocus(handle, &five, 1, &locus) == 0 && scePlayGoGetLocus(handle, &thirtyNine, 1, &locus) == BadChunkId;
}

}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const std::string mode = argv[1];
    const auto previous = std::filesystem::current_path();
    const auto root = std::filesystem::temp_directory_path() / ("anyps5-playgo-chunk-table-" + mode + "-" + std::to_string(std::random_device{}()));
    bool passed = false;
    try {
        std::filesystem::create_directories(root / "app0" / "sce_sys");
        std::filesystem::current_path(root);
        const auto table = root / "app0" / "sce_sys" / "playgo-chunk.dat";
        PlayGoInitParams init{};
        int handle = 0;
        if (scePlayGoInitialize(&init) == 0 && scePlayGoOpen(&handle, nullptr) == 0) {
            passed = mode == "table" ? TableMode(table, handle) : mode == "xml" && XmlMode(table, root / "app0", handle);
            passed = scePlayGoClose(handle) == 0 && passed;
        }
    } catch (const std::exception&) {
        passed = false;
    }
    std::filesystem::current_path(previous);
    std::filesystem::remove_all(root);
    return passed ? 0 : 1;
}
