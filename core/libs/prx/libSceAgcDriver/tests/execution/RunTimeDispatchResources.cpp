#include "WrittenImageFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "SceShaders.hpp"
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

using namespace WrittenImages;
using AgcDriver::Graphics::TextureTileMode;
using NewPath::High;
using NewPath::Low;

constexpr std::uint32_t Format16_16_16_16Float = 71;
constexpr std::uint32_t Format32_32_32_32Float = 77;
constexpr std::uint32_t Chains = 8;
constexpr std::uint32_t SlotWords = Width * Height * 4;
constexpr std::uint32_t LinkWords = 64;
constexpr std::uint32_t RewrittenRoot = Chains;
constexpr std::uint32_t StorageRoot = Chains + 1;
constexpr std::uint32_t Links = Chains + 2;
constexpr std::uint32_t ShuffledSlot = Chains;
constexpr std::uint32_t RewrittenSlot = 2 * Chains;
constexpr std::uint32_t StoredSlot = 3 * Chains;
constexpr std::uint32_t RegionWords = 1024;
constexpr std::uint32_t TableSeedBias = 0x10000u;
constexpr std::uint32_t FirstStorageSeed = 0x21u;
constexpr std::uint32_t SecondStorageSeed = 0x6au;
constexpr std::uint32_t BufferStoreDword = 0x1cu;
constexpr std::array<std::uint32_t, Chains> Formats{Format8_8_8_8UNorm, Format32Float, Format16_16_16_16Float, Format32_32_32_32Float, Format32Float, Format8_8_8_8UNorm, Format32_32_32_32Float, Format16_16_16_16Float};

struct Store {
    std::uint32_t stride;
    std::uint32_t records;
    std::uint32_t offsetWords;
    std::uint32_t seed;
};

constexpr std::array<Store, 8> Stores{{
    {4, 64, 0, 0x100},
    {8, 64, 1, 0x200},
    {12, 40, 3, 0x300},
    {20, 64, 2, 0x400},
    {64, 17, 0, 0x500},
    {16, 1, 5, 0x600},
    {4, 33, 7, 0x700},
    {28, 64, 1, 0x800},
}};

TestGuest::Array<std::uint32_t, Chains * SlotWords> Images;
TestGuest::Array<std::uint32_t, Links * LinkWords> Roots;
TestGuest::Array<std::uint32_t, Links * LinkWords> Tables;
TestGuest::Array<std::uint32_t, Width * Height * 4> Storage;
TestGuest::Array<std::uint32_t, Stores.size() * RegionWords> Stored;
TestGuest::Array<std::uint32_t, Stores.size() * LinkWords> BufferTables;
TestGuest::Array<std::uint32_t, 64> CatchAllTarget;

std::uint64_t AddressOf(const void* pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint64_t Hits() {
    return AgcDriver::DriverDetail::Driver::Get().DataIndependentDispatchHits();
}

void ExpectHits(std::uint64_t before, std::uint64_t expected, const std::string& what) {
    const auto hits = Hits() - before;
    Expect(hits == expected, what + ": " + std::to_string(hits) + " dispatches were served without a capture, expected all " + std::to_string(expected) + ", the first included");
}

std::uint32_t Byte(std::uint32_t chain, std::uint32_t x, std::uint32_t y, std::uint32_t component) {
    return (0x11u + 29u * chain + 3u * x + 41u * y + 67u * component) & 0xffu;
}

float Single(std::uint32_t chain, std::uint32_t x, std::uint32_t y) {
    return -0.5f * static_cast<float>(x + 64u * y + 1000u * chain);
}

std::uint16_t Half(std::uint32_t chain, std::uint32_t x, std::uint32_t y, std::uint32_t component) {
    return static_cast<std::uint16_t>((((x + component) & 1u) << 15u) | 0x3c00u | ((5u * x + 17u * y + 31u * component + 11u * chain) & 0x3ffu));
}

float HalfValue(std::uint16_t half) {
    const auto magnitude = 1.0f + static_cast<float>(half & 0x3ffu) / 1024.0f;
    return (half & 0x8000u) != 0u ? -magnitude : magnitude;
}

float Wide(std::uint32_t chain, std::uint32_t x, std::uint32_t y, std::uint32_t component) {
    return static_cast<float>(x + 64u * y) + 0.25f * static_cast<float>(component) + 512.0f * static_cast<float>(chain);
}

std::uint32_t TexelBytes(std::uint32_t format) {
    switch (format) {
    case Format16_16_16_16Float: return 8u;
    case Format32_32_32_32Float: return 16u;
    default: return 4u;
    }
}

std::string FormatName(std::uint32_t format) {
    switch (format) {
    case Format8_8_8_8UNorm: return "8_8_8_8 unorm";
    case Format32Float: return "32 float";
    case Format16_16_16_16Float: return "16_16_16_16 float";
    default: return "32_32_32_32 float";
    }
}

std::byte* SlotBytes(std::uint32_t chain) {
    return reinterpret_cast<std::byte*>(Images.data() + chain * SlotWords);
}

void FillImage(std::uint32_t chain) {
    const auto format = Formats[chain];
    const auto layout = AgcDriver::Graphics::ComputeMipLayout(TextureTileMode::kLinear, format, Width, Height, 1u).at(0);
    AgcDriver::Graphics::Require(layout.tiledOffset + layout.tiledSize <= SlotWords * 4u, "run-time dispatch resources: a texture does not fit its slot");
    auto* base = SlotBytes(chain) + layout.tiledOffset;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            auto* texel = base + y * layout.pitchBytes + x * TexelBytes(format);
            if (format == Format8_8_8_8UNorm) {
                const auto word = Byte(chain, x, y, 0) | (Byte(chain, x, y, 1) << 8u) | (Byte(chain, x, y, 2) << 16u) | (Byte(chain, x, y, 3) << 24u);
                std::memcpy(texel, &word, sizeof(word));
            } else if (format == Format32Float) {
                const auto value = Single(chain, x, y);
                std::memcpy(texel, &value, sizeof(value));
            } else if (format == Format16_16_16_16Float) {
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto half = Half(chain, x, y, component);
                    std::memcpy(texel + component * 2u, &half, sizeof(half));
                }
            } else {
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto value = Wide(chain, x, y, component);
                    std::memcpy(texel + component * 4u, &value, sizeof(value));
                }
            }
        }
    }
}

bool Matches(std::uint32_t chain, std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) {
    switch (Formats[chain]) {
    case Format8_8_8_8UNorm: return SameUnorm(word, Byte(chain, x, y, component));
    case Format32Float: return SameFloat(word, component == 0u ? Single(chain, x, y) : component == 3u ? 1.0f : 0.0f);
    case Format16_16_16_16Float: return SameFloat(word, HalfValue(Half(chain, x, y, component)));
    default: return SameFloat(word, Wide(chain, x, y, component));
    }
}

std::uint32_t* Link(TestGuest::Array<std::uint32_t, Links * LinkWords>& links, std::uint32_t index) {
    return links.data() + index * LinkWords;
}

std::array<std::uint32_t, 8> Descriptor(const void* texels, std::uint32_t format) {
    const auto words = Image(texels, format);
    return {words[0], words[1], words[2], words[3], 0u, 0u, 0u, 0u};
}

std::array<std::uint32_t, 8> ChainDescriptor(std::uint32_t chain) {
    return Descriptor(SlotBytes(chain), Formats[chain]);
}

void LinkChain(std::uint32_t index, const std::array<std::uint32_t, 8>& descriptor) {
    auto* root = Link(Roots, index);
    auto* table = Link(Tables, index);
    std::fill(root, root + LinkWords, 0u);
    std::fill(table, table + LinkWords, 0u);
    const auto address = AddressOf(table);
    root[0] = Low(address);
    root[1] = High(address);
    std::copy(PointClamp.begin(), PointClamp.end(), root + SamplerWord);
    std::copy(descriptor.begin(), descriptor.end(), table);
}

std::array<std::uint32_t, 8> ChainUserData(std::uint32_t index, std::uint32_t slot) {
    std::array<std::uint32_t, 8> userData{};
    const auto root = AddressOf(Link(Roots, index));
    userData[0] = Low(root);
    userData[1] = High(root);
    const auto output = Buffer(Results.data() + slot * DispatchWords, DispatchWords * 4u);
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    return userData;
}

void ExpectChain(std::uint32_t slot, const std::function<bool(std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word)>& matches, const std::string& what) {
    const auto* words = Results.data() + slot * DispatchWords;
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(words), DispatchWords * 4u);
    std::size_t wrong = 0;
    std::size_t untouched = 0;
    std::string first;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Lanes; ++x) {
            for (std::uint32_t component = 0; component < LaneWords; ++component) {
                const auto word = words[(y * Lanes + x) * LaneWords + component];
                if (matches(x, y, component, word)) continue;
                untouched += word == Untouched ? 1u : 0u;
                if (wrong++ == 0u) first = "texel (" + std::to_string(x) + ", " + std::to_string(y) + ") component " + std::to_string(component) + " is " + Hex(word);
            }
        }
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(DispatchWords) + " words differ (" + std::to_string(untouched) + " untouched), first " + first);
}

void ChainPerDispatch() {
    for (std::uint32_t chain = 0; chain < Chains; ++chain) {
        FillImage(chain);
        LinkChain(chain, ChainDescriptor(chain));
    }
    const auto before = Hits();
    Commands commands;
    for (std::uint32_t dispatch = 0; dispatch < Chains; ++dispatch) commands.Dispatch(ChainProgram(), ChainUserData(dispatch, dispatch), Lanes, Height);
    commands.Submit(0);
    NewPath::WaitIdle();
    for (std::uint32_t dispatch = 0; dispatch < Chains; ++dispatch) {
        ExpectChain(dispatch, [dispatch](std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) { return Matches(dispatch, x, y, component, word); }, "chain per dispatch: dispatch " + std::to_string(dispatch) + " of one submission samples a " + FormatName(Formats[dispatch]) + " texture through its own root and table in guest memory");
    }
    ExpectHits(before, Chains, "chain per dispatch: one program, a different T# chain and format per dispatch");

    const auto again = Hits();
    Commands shuffled;
    const auto chainOf = [](std::uint32_t dispatch) { return (3u * dispatch + 1u) % Chains; };
    for (std::uint32_t dispatch = 0; dispatch < Chains; ++dispatch) shuffled.Dispatch(ChainProgram(), ChainUserData(chainOf(dispatch), ShuffledSlot + dispatch), Lanes, Height);
    shuffled.Submit(0);
    NewPath::WaitIdle();
    for (std::uint32_t dispatch = 0; dispatch < Chains; ++dispatch) {
        const auto chain = chainOf(dispatch);
        ExpectChain(ShuffledSlot + dispatch, [chain](std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) { return Matches(chain, x, y, component, word); }, "chain per dispatch, shuffled: dispatch " + std::to_string(dispatch) + " of the next submission samples chain " + std::to_string(chain) + " (" + FormatName(Formats[chain]) + ")");
    }
    ExpectHits(again, Chains, "chain per dispatch, shuffled: the same chains in another order");
}

void RewrittenTable() {
    LinkChain(RewrittenRoot, ChainDescriptor(Chains - 1u));
    const auto table = AddressOf(Link(Tables, RewrittenRoot));
    const auto before = Hits();
    Commands commands;
    for (std::uint32_t dispatch = 0; dispatch < Chains; ++dispatch) {
        const auto words = ChainDescriptor(dispatch);
        if (dispatch != 0u) commands.Barrier();
        commands.WriteData(table, {words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]});
        commands.Barrier();
        commands.Dispatch(ChainProgram(), ChainUserData(RewrittenRoot, RewrittenSlot + dispatch), Lanes, Height);
    }
    commands.Submit(0);
    NewPath::WaitIdle();
    for (std::uint32_t dispatch = 0; dispatch < Chains; ++dispatch) {
        ExpectChain(RewrittenSlot + dispatch, [dispatch](std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) { return Matches(dispatch, x, y, component, word); }, "rewritten table: dispatch " + std::to_string(dispatch) + " samples the " + FormatName(Formats[dispatch]) + " T# a WRITE_DATA stored into the shared table just before it, in the same submission");
    }
    ExpectHits(before, Chains, "rewritten table: one root and table, its T# rewritten between the dispatches of a submission");
    std::array<std::uint32_t, 8> stored{};
    AgcDriver::GuestMemory::Read(table, std::as_writable_bytes(std::span(stored)));
    Expect(stored == ChainDescriptor(Chains - 1u), "rewritten table: the table in guest memory does not hold the last T# written into it");
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
        using namespace Ops;
        Program built;
        built << Vop2(VAddNcU32, 1, 4, 0)
              << IndexedStore(1, 0, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& TableStoreProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Smem(SLoadDwordX4, 8, 0, 0)
              << WaitAll
              << Vop2(VAddNcU32, 1, 2, 0)
              << IndexedStore(1, 0, 8)
              << EndProgram;
        return built;
    }();
    return program;
}

const std::uint32_t* StoreBase(std::uint32_t store) {
    return Stored.data() + store * RegionWords + Stores[store].offsetWords;
}

void ExpectStores(std::uint32_t seedBias, const std::string& what) {
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(Stored.data()), decltype(Stored)::ByteSize);
    for (std::uint32_t store = 0; store < Stores.size(); ++store) {
        const auto& value = Stores[store];
        const auto* region = Stored.data() + store * RegionWords;
        std::size_t wrong = 0;
        std::string first;
        for (std::uint32_t word = 0; word < RegionWords; ++word) {
            auto expected = Untouched;
            if (word >= value.offsetWords) {
                const auto offset = (word - value.offsetWords) * 4u;
                const auto index = offset / value.stride;
                if (offset % value.stride == 0u && index < std::min(value.records, Lanes)) expected = index + value.seed + seedBias;
            }
            if (region[word] == expected) continue;
            if (wrong++ == 0u) first = "word " + std::to_string(word) + " is " + Hex(region[word]) + ", expected " + Hex(expected);
        }
        Expect(wrong == 0u, what + ": dispatch " + std::to_string(store) + " (stride " + std::to_string(value.stride) + ", " + std::to_string(value.records) + " records, base word " + std::to_string(value.offsetWords) + "): " + std::to_string(wrong) + " words differ, first " + first);
    }
}

void UserDataBuffers() {
    Stored.fill(Untouched);
    const auto before = Hits();
    Commands commands;
    for (std::uint32_t store = 0; store < Stores.size(); ++store) {
        const auto& value = Stores[store];
        const auto resource = Structured(StoreBase(store), value.stride, value.records);
        const std::array<std::uint32_t, 5> userData{resource[0], resource[1], resource[2], resource[3], value.seed};
        commands.Dispatch(UserDataStoreProgram(), userData, Lanes);
    }
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectStores(0u, "V# from user data: an indexed store through the dispatch's own base, stride and record count");
    ExpectHits(before, Stores.size(), "V# from user data: one program, a different V# in each dispatch's user data");
}

void TableBuffers() {
    Stored.fill(Untouched);
    BufferTables.fill(0u);
    const auto before = Hits();
    Commands commands;
    for (std::uint32_t store = 0; store < Stores.size(); ++store) {
        const auto& value = Stores[store];
        const auto resource = Structured(StoreBase(store), value.stride, value.records);
        auto* table = BufferTables.data() + store * LinkWords;
        std::copy(resource.begin(), resource.end(), table);
        const auto address = AddressOf(table);
        const std::array<std::uint32_t, 3> userData{Low(address), High(address), value.seed + TableSeedBias};
        commands.Dispatch(TableStoreProgram(), userData, Lanes);
    }
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectStores(TableSeedBias, "V# from a table in guest memory: an indexed store through the V# its own table holds");
    ExpectHits(before, Stores.size(), "V# from a table in guest memory: one program, a different table pointer in each dispatch's user data");
}

const Program& StorageWriterProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 9)
              << Vop1(VCvtF32U32, 4, Vgpr(1))
              << Vop1(VCvtF32U32, 5, Vgpr(2))
              << Vop1(VCvtF32U32, 6, 8)
              << Vop2(VLshlRevB32, 7, InlineInteger(6), 2)
              << Vop2(VAddNcU32, 7, Vgpr(1), 7)
              << Vop2(VAddNcU32, 7, 8, 7)
              << Vop1(VCvtF32U32, 7, Vgpr(7))
              << Mimg(ImageStore, 0xfu, false, 4, 1, 0, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

std::array<std::uint32_t, 9> StorageUserData(std::uint32_t seed) {
    std::array<std::uint32_t, 9> userData{};
    const auto words = Descriptor(Storage.data(), Format32_32_32_32Float);
    std::copy(words.begin(), words.end(), userData.begin());
    userData[8] = seed;
    return userData;
}

float StoredComponent(std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t seed) {
    const std::array<std::uint32_t, 4> values{x, y, seed, x + 64u * y + seed};
    return static_cast<float>(values[component]);
}

void ExpectStorage(std::uint32_t seed, const std::string& what) {
    const auto address = AddressOf(Storage.data());
    NewPath::WaitIdle();
    AgcDriver::Graphics::StorageTexture::FlushPending(address, decltype(Storage)::ByteSize, nullptr, "run-time dispatch resources");
    NewPath::WaitIdle();
    AgcDriver::GuestMemory::FlushGpuWrites(address, decltype(Storage)::ByteSize);
    const auto layout = AgcDriver::Graphics::ComputeMipLayout(TextureTileMode::kLinear, Format32_32_32_32Float, Width, Height, 1u).at(0);
    const auto* bytes = reinterpret_cast<const std::byte*>(Storage.data()) + layout.tiledOffset;
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            for (std::uint32_t component = 0; component < 4u; ++component) {
                std::uint32_t word = 0;
                std::memcpy(&word, bytes + y * layout.pitchBytes + x * 16u + component * 4u, sizeof(word));
                if (SameFloat(word, StoredComponent(x, y, component, seed))) continue;
                if (wrong++ == 0u) first = "texel (" + std::to_string(x) + ", " + std::to_string(y) + ") component " + std::to_string(component) + " is " + Hex(word);
            }
        }
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " stored words in guest memory differ, first " + first);
}

void StoredThenSampled() {
    Storage.fill(0u);
    LinkChain(StorageRoot, Descriptor(Storage.data(), Format32_32_32_32Float));
    const auto alone = Hits();
    Commands first;
    first.Dispatch(StorageWriterProgram(), StorageUserData(FirstStorageSeed), Lanes, Height);
    first.Submit(0);
    NewPath::WaitIdle();
    ExpectHits(alone, 1u, "storage image: the writer dispatch alone, its T# in user data");
    ExpectStorage(FirstStorageSeed, "storage image: the writer dispatch alone");

    const auto before = Hits();
    Commands commands;
    commands.Dispatch(StorageWriterProgram(), StorageUserData(SecondStorageSeed), Lanes, Height);
    commands.Barrier();
    commands.Dispatch(ChainProgram(), ChainUserData(StorageRoot, StoredSlot), Lanes, Height);
    commands.Submit(0);
    NewPath::WaitIdle();
    ExpectChain(StoredSlot, [](std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) { return SameFloat(word, StoredComponent(x, y, component, SecondStorageSeed)); }, "storage image: the next dispatch of the submission samples the texels the writer stored, through a T# chain in guest memory");
    ExpectHits(before, 2u, "storage image: the writer and the reader of one submission");
    ExpectStorage(SecondStorageSeed, "storage image: guest memory after the writer and the reader");
}

const Program& CatchAllWriterProgram() {
    static const Program program = [] {
        using namespace Ops;
        constexpr std::uint32_t BufferLoadDwordX2 = 0x0du;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(0))
              << Mubuf(BufferLoadDwordX2, 2, 1, 0)
              << Vop1(VMovB32, 4, Literal) << 0x5b000005u
              << WaitAll
              << Global(GlobalStoreDword, 0, 2, 4, NullSgpr)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& AddressWriterProgram() {
    static const Program program = [] {
        using namespace Ops;
        constexpr std::uint32_t BufferStoreDwordX2 = 0x1du;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(0))
              << Vop1(VMovB32, 2, 4)
              << Vop1(VMovB32, 3, 5)
              << Mubuf(BufferStoreDwordX2, 2, 1, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

void RegisterChainProgram() {
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 7> registers{};
        ShaderSpecialRegs specials{};
    };
    alignas(8) static Header header{};
    const auto& program = ChainProgram();
    const auto address = program.Address();
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(program.size * sizeof(std::uint32_t));
    header.shader.code = const_cast<std::uint32_t*>(program.words.data());
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, Lanes}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, (8u << 1u) | (1u << 7u)}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

const Program& CatchAllValueWriterProgram() {
    static const Program program = [] {
        using namespace Ops;
        constexpr std::uint32_t BufferLoadDwordX2 = 0x0du;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(0))
              << Mubuf(BufferLoadDwordX2, 2, 1, 0)
              << Vop1(VMovB32, 4, 4)
              << WaitAll
              << Global(GlobalStoreDword, 0, 2, 4, NullSgpr)
              << EndProgram;
        return built;
    }();
    return program;
}

constexpr std::uint64_t ParkedHits = 3;

class ParkedSubmission {
public:
    void Dispatch(const Program& program, std::span<const std::uint32_t> userData, std::uint32_t threads, std::uint32_t groups = 1) {
        Registers(0x207, {threads, 1u, 1u});
        Registers(0x20c, {Low(program.Address() >> 8u), Low(program.Address() >> 40u)});
        Registers(0x213, {(static_cast<std::uint32_t>(userData.size()) << 1u) | (1u << 7u)});
        Registers(0x240, userData);
        words.insert(words.end(), {0xc0031500u, groups, 1u, 1u, 0x41u});
    }

    void Flip(std::uint32_t handle) {
        words.insert(words.end(), {AgcDriver::FlipPacketHeader, handle, 0xfffffffeu, 1u, 0u, 0u});
    }

    void Submit(std::uint32_t queue) {
        Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
        AgcDriver::Submit(&packet, queue);
        words.clear();
    }

private:
    void Registers(std::uint32_t first, std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    }

    void Registers(std::uint32_t first, std::initializer_list<std::uint32_t> values) { Registers(first, std::span<const std::uint32_t>(values.begin(), values.size())); }

    std::vector<std::uint32_t> words;
};

class ParkedReader {
public:
    ParkedReader(std::uint32_t chain, std::uint32_t slot) : chain(chain), slot(slot) {
        RegisterChainProgram();
        FillImage(chain);
        LinkChain(chain, ChainDescriptor(chain));
        root = Link(Roots, chain);
        tableLow = root[0];
        CatchAllTarget.fill(0u);
        target = AddressOf(root);
        const auto resource = Buffer(CatchAllTarget.data(), 8u);
        writerData = {resource[0], resource[1], resource[2], resource[3], tableLow};
        addressData = {resource[0], resource[1], resource[2], resource[3], Low(target), High(target)};
        ParkedSubmission warm;
        Append(warm);
        warm.Submit(0);
        NewPath::WaitIdle();
        root[0] = 0u;
        waitsBefore = AgcDriver::GuestMemory::PendingReadWaitCounts().reads;
        deferredBefore = Deferred();
        hitsBefore = Hits();
    }

    void Submit(std::uint32_t flip = 0) const {
        ParkedSubmission submission;
        Append(submission);
        if (flip != 0u) submission.Flip(flip);
        submission.Submit(0);
    }

    static std::uint64_t Deferred() {
        return AgcDriver::DriverDetail::Driver::Get().DeferredDispatches();
    }

    std::uint64_t DeferredSince() const {
        return Deferred() - deferredBefore;
    }

    std::uint64_t HitsSince() const {
        return Hits() - hitsBefore;
    }

    bool AwaitParked() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (DeferredSince() == 0u && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        return DeferredSince() != 0u;
    }

    void ExpectRecorded(const std::string& what) const {
        const auto waits = AgcDriver::GuestMemory::PendingReadWaitCounts().reads - waitsBefore;
        Expect(waits == 0u, what + ": the recording thread waited for the writer " + std::to_string(waits) + " times");
        Expect(DeferredSince() != 0u, what + ": the reader was not deferred");
        AgcDriver::GuestMemory::FlushGpuWrites(target, 4u);
        Expect(root[0] == tableLow, what + ": the writer did not store the table address: " + Hex(root[0]));
        const auto sampled = chain;
        ExpectChain(slot, [sampled](std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) { return Matches(sampled, x, y, component, word); }, what + ": the deferred reader samples through the written table address");
    }

private:
    void Append(ParkedSubmission& submission) const {
        submission.Dispatch(AddressWriterProgram(), addressData, 1);
        submission.Dispatch(CatchAllValueWriterProgram(), writerData, Lanes);
        submission.Dispatch(ChainProgram(), ChainUserData(chain, slot), Lanes, Height);
    }

    std::uint32_t chain;
    std::uint32_t slot;
    std::uint32_t* root = nullptr;
    std::uint32_t tableLow = 0;
    std::uint64_t target = 0;
    std::array<std::uint32_t, 5> writerData{};
    std::array<std::uint32_t, 6> addressData{};
    std::uint64_t waitsBefore = 0;
    std::uint64_t deferredBefore = 0;
    std::uint64_t hitsBefore = 0;
};

void PrePassUnderCatchAllDeferred() {
    ParkedReader reader(1, StoredSlot + 2u);
    reader.Submit();
    NewPath::WaitIdle();
    const auto recorded = reader.HitsSince();
    Expect(recorded == ParkedHits, "pre-pass under a catch-all writer: the idle wait returned with " + std::to_string(recorded) + " of " + std::to_string(ParkedHits) + " served dispatches recorded");
    reader.ExpectRecorded("pre-pass under a catch-all writer");
}

constexpr std::uint32_t ParkedOutputHandle = 0x74bu;

class ParkedFlip final : public AgcDriver::IFlipRequest {
public:
    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override {
        hitsAtFlip.store(Hits(), std::memory_order_release);
        deferredAtFlip.store(ParkedReader::Deferred(), std::memory_order_release);
        ready.fetch_add(1, std::memory_order_acq_rel);
    }

    void Fail(std::exception_ptr) noexcept override { failed.fetch_add(1, std::memory_order_acq_rel); }

    std::atomic<std::uint64_t> hitsAtFlip{0};
    std::atomic<std::uint64_t> deferredAtFlip{0};
    std::atomic<int> ready{0};
    std::atomic<int> failed{0};
};

class ParkedOutput final : public AgcDriver::IVideoOutput {
public:
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo&) override {
        auto flip = std::make_shared<ParkedFlip>();
        std::lock_guard lock(mutex);
        flips.push_back(flip);
        return flip;
    }

    void Fail(std::exception_ptr) noexcept override {}

    std::shared_ptr<ParkedFlip> Only() {
        std::lock_guard lock(mutex);
        return flips.size() == 1u ? flips.front() : nullptr;
    }

    std::size_t Reserved() {
        std::lock_guard lock(mutex);
        return flips.size();
    }

private:
    std::mutex mutex;
    std::vector<std::shared_ptr<ParkedFlip>> flips;
};

void FlipBehindParkedReader() {
    auto output = std::make_shared<ParkedOutput>();
    AgcDriverRegisterVideoOutput_nid_postfix(ParkedOutputHandle, output);
    ParkedReader reader(1, StoredSlot + 3u);
    const auto deferredBefore = ParkedReader::Deferred();
    reader.Submit(ParkedOutputHandle);
    NewPath::WaitIdle();
    const auto hits = Hits();
    AgcDriverUnregisterVideoOutput_nid_postfix(ParkedOutputHandle, output);
    const auto flip = output->Only();
    Expect(flip != nullptr, "flip behind a parked reader: " + std::to_string(output->Reserved()) + " flips reserved, expected 1");
    if (flip != nullptr) {
        Expect(flip->ready.load() == 1 && flip->failed.load() == 0, "flip behind a parked reader: the flip was presented " + std::to_string(flip->ready.load()) + " times and failed " + std::to_string(flip->failed.load()) + " times");
        Expect(flip->deferredAtFlip.load() > deferredBefore, "flip behind a parked reader: the flip ran before the reader was deferred");
        Expect(flip->hitsAtFlip.load() == hits, "flip behind a parked reader: the flip ran before the deferred reader recorded");
    }
    reader.ExpectRecorded("flip behind a parked reader");
}

void SuspendOfParkedQueue() {
    ParkedReader reader(1, StoredSlot + 4u);
    reader.Submit();
    AgcDriverSuspendPoint_nid_postfix();
    NewPath::WaitIdle();
    const auto recorded = reader.HitsSince();
    Expect(recorded == ParkedHits, "suspend of a parked queue: the suspend completed with " + std::to_string(recorded) + " of " + std::to_string(ParkedHits) + " served dispatches recorded");
    reader.ExpectRecorded("suspend of a parked queue");
    ParkedReader after(1, StoredSlot + 5u);
    after.Submit();
    NewPath::WaitIdle();
    after.ExpectRecorded("suspend of a parked queue: the next submission");
}

void ShutdownWithParkedQueue() {
    ParkedReader reader(1, StoredSlot + 6u);
    reader.Submit();
    Expect(reader.AwaitParked(), "shutdown with a parked queue: the reader was not deferred");
}

void FailureWhileParked() {
    auto output = std::make_shared<ParkedOutput>();
    AgcDriverRegisterVideoOutput_nid_postfix(ParkedOutputHandle, output);
    ParkedReader reader(1, StoredSlot + 7u);
    reader.Submit(ParkedOutputHandle);
    Expect(reader.AwaitParked(), "failure while parked: the reader was not deferred");
    NewPath::Locked([] { AgcDriverReportFailure_nid_postfix(std::make_exception_ptr(std::runtime_error("device failure while a queue is parked"))); });
    bool threw = false;
    try {
        NewPath::WaitIdle();
    } catch (const std::exception&) {
        threw = true;
    }
    Expect(threw, "failure while parked: the idle wait returned without the failure");
    const auto settled = [&] {
        const auto flip = output->Only();
        return flip != nullptr && flip->failed.load() + flip->ready.load() != 0;
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!settled() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto flip = output->Only();
    Expect(flip != nullptr, "failure while parked: " + std::to_string(output->Reserved()) + " flips reserved, expected 1");
    if (flip != nullptr) {
        Expect(flip->failed.load() == 1 && flip->ready.load() == 0, "failure while parked: the parked submission's flip failed " + std::to_string(flip->failed.load()) + " times and was presented " + std::to_string(flip->ready.load()) + " times");
    }
    AgcDriverUnregisterVideoOutput_nid_postfix(ParkedOutputHandle, output);
}

void PrePassAfterCatchAll() {
    constexpr std::uint32_t chain = 0;
    constexpr std::uint32_t slot = StoredSlot + 1u;
    RegisterChainProgram();
    FillImage(chain);
    LinkChain(chain, ChainDescriptor(chain));
    CatchAllTarget.fill(0u);
    const auto target = AddressOf(CatchAllTarget.data() + 8);
    const auto resource = Buffer(CatchAllTarget.data(), 8u);
    const std::array<std::uint32_t, 4> writerData{resource[0], resource[1], resource[2], resource[3]};
    const std::array<std::uint32_t, 6> addressData{resource[0], resource[1], resource[2], resource[3], Low(target), High(target)};
    Commands warm;
    warm.Dispatch(AddressWriterProgram(), addressData, 1);
    warm.Dispatch(CatchAllWriterProgram(), writerData, Lanes);
    warm.Dispatch(ChainProgram(), ChainUserData(chain, slot), Lanes, Height);
    warm.Submit(0);
    NewPath::WaitIdle();
    CatchAllTarget[8] = 0u;
    Commands commands;
    commands.Dispatch(AddressWriterProgram(), addressData, 1);
    commands.Dispatch(CatchAllWriterProgram(), writerData, Lanes);
    commands.Dispatch(ChainProgram(), ChainUserData(chain, slot), Lanes, Height);
    commands.Submit(0);
    NewPath::WaitIdle();
    Expect(CatchAllTarget[8] == 0x5b000005u, "pre-pass after a catch-all writer: the unbounded store did not land: " + Hex(CatchAllTarget[8]));
    ExpectChain(slot, [](std::uint32_t x, std::uint32_t y, std::uint32_t component, std::uint32_t word) { return Matches(chain, x, y, component, word); }, "pre-pass after a catch-all writer: the reader samples its chain");
}

void Run() {
    const std::array<std::pair<const char*, void (*)()>, 10> cases{{
        {"chain per dispatch", ChainPerDispatch},
        {"rewritten table", RewrittenTable},
        {"V# from user data", UserDataBuffers},
        {"V# from a table in guest memory", TableBuffers},
        {"storage image", StoredThenSampled},
        {"pre-pass after a catch-all writer", PrePassAfterCatchAll},
        {"pre-pass under a catch-all writer is deferred", PrePassUnderCatchAllDeferred},
        {"flip behind a parked reader", FlipBehindParkedReader},
        {"suspend of a parked queue", SuspendOfParkedQueue},
        {"shutdown with a parked queue", ShutdownWithParkedQueue},
    }};
    for (const auto& [name, run] : cases) {
        try {
            run();
        } catch (const std::exception& error) {
            Expect(false, std::string(name) + ": exception: " + error.what());
        }
    }
}

}

int FailureWhileParkedMain() {
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        FillTextures();
        FailureWhileParked();
    } catch (const std::exception& error) {
        Expect(false, std::string("failure while parked: exception: ") + error.what());
    }
    bool threw = false;
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception&) {
        threw = true;
    }
    Expect(threw, "failure while parked: shutdown did not report the failure");
    return Finish("run time dispatch resources, failure while parked");
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--failure-while-parked") return FailureWhileParkedMain();
    return Main("run time dispatch resources", Run);
}
