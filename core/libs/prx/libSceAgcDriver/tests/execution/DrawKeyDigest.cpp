#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "Recompiler.hpp"
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::CompiledShader;
using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::ShaderResources;
using ShaderRecompiler::DescriptorBinding;
using ShaderRecompiler::DescriptorImageShape;
using ShaderRecompiler::DescriptorKind;
using ShaderRecompiler::DescriptorRole;
using ShaderRecompiler::RecompileResult;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t TableElements = 64;

RecompileResult Program(std::uint32_t tableElements = TableElements) {
    RecompileResult result;
    result.variantId = 0x1234'5678'9abc'def0ull;
    DescriptorBinding buffers{DescriptorKind::StorageBuffer, DescriptorRole::GuestBuffers, 0, 0, 2, {}};
    buffers.guestDescriptor = {0x1000u, 0x0002'0000u, 0x100u, 0x2002'2facu, 0x8000u, 0x0001'0000u, 0x200u, 0x2002'2facu};
    buffers.bufferWritten = {true, false};
    DescriptorBinding table{DescriptorKind::SampledImage, DescriptorRole::ImageTable, 0, 1, tableElements, {}};
    for (std::uint32_t element = 0; element < tableElements; ++element) {
        for (std::uint32_t word = 0; word < 8; ++word) table.guestDescriptor.push_back(element * 8u + word + 1u);
        table.imageShapes.push_back(DescriptorImageShape::Image2D);
        table.imageSamplers.push_back(element % 3u);
        table.imageWritten.push_back(element % 5u == 0);
        table.imageDepthCompare.push_back(false);
        table.imageAtomic.push_back(false);
    }
    result.bindings = {buffers, table};
    return result;
}

std::vector<std::uint32_t> Key(const RecompileResult& result, ShaderStage stage) {
    return ShaderResources::ContentKey(CompiledShader{stage, &result, 0}, true, true);
}

ShaderResources::ContentDigest Digest(const RecompileResult& result, ShaderStage stage) {
    return ShaderResources::ContentKeyDigest(CompiledShader{stage, &result, 0}, true);
}

// The digest keeps the content key's first words and its length, and differs exactly when the key does.
void DigestFollowsTheContentKey() {
    const auto base = Program();
    const auto baseKey = Key(base, ShaderStage::Fragment);
    const auto baseDigest = Digest(base, ShaderStage::Fragment);
    Require(baseDigest[0] == ShaderResources::ContentDigestMarker && baseDigest[0] != baseKey[0], "the digest does not start with its marker");
    for (std::size_t word = 1; word < 5; ++word) Require(baseDigest[word] == baseKey[word], "the digest lost word " + std::to_string(word) + " of the content key");
    Require(baseDigest[5] == baseKey.size(), "the digest does not carry the content key's length");
    Require(Digest(base, ShaderStage::Fragment) == baseDigest, "the same content gave two digests");

    const std::vector<std::pair<std::string, std::function<void(RecompileResult&)>>> changes{
        {"a written buffer's base", [](RecompileResult& r) { r.bindings[0].guestDescriptor[0] ^= 0x40u; }},
        {"a read-only buffer's base (moved, not keyed)", [](RecompileResult& r) { r.bindings[0].guestDescriptor[4] ^= 0x40u; }},
        {"a read-only buffer's format word", [](RecompileResult& r) { r.bindings[0].guestDescriptor[7] ^= 1u; }},
        {"the buffers' write set", [](RecompileResult& r) { r.bindings[0].bufferWritten[1] = true; }},
        {"the last table element's last word", [](RecompileResult& r) { r.bindings[1].guestDescriptor.back() ^= 1u; }},
        {"a table element's sampler", [](RecompileResult& r) { r.bindings[1].imageSamplers[17] = 9; }},
        {"a table element's shape", [](RecompileResult& r) { r.bindings[1].imageShapes[33] = DescriptorImageShape::Image2DArray; }},
        {"a table element's written bit", [](RecompileResult& r) { r.bindings[1].imageWritten[63] = !r.bindings[1].imageWritten[63]; }},
        {"a table element's depth compare", [](RecompileResult& r) { r.bindings[1].imageDepthCompare[1] = true; }},
        {"the variant", [](RecompileResult& r) { ++r.variantId; }},
        {"one more table element", [](RecompileResult& r) {
            auto& table = r.bindings[1];
            ++table.count;
            table.guestDescriptor.insert(table.guestDescriptor.end(), 8, 0u);
            table.imageShapes.push_back(DescriptorImageShape::Image2D);
            table.imageSamplers.push_back(0);
        }},
        {"an empty write set bit appended", [](RecompileResult& r) { r.bindings[1].imageAtomic.push_back(false); }},
    };
    for (const auto& [name, change] : changes) {
        auto changed = base;
        change(changed);
        const bool keyChanged = Key(changed, ShaderStage::Fragment) != baseKey;
        const bool digestChanged = Digest(changed, ShaderStage::Fragment) != baseDigest;
        Require(keyChanged == digestChanged, "changing " + name + (keyChanged ? " changed the content key but not the digest" : " changed the digest but not the content key"));
    }
    Require(Digest(base, ShaderStage::Vertex) != baseDigest, "the stage is not part of the digest");
}

// Every descriptor and sampler word of a table reaches the digest, whatever the table's length (the
// digest takes runs of words in stripes of eight).
void EveryTableWordCounts() {
    for (const std::uint32_t elements : {1u, 3u, 61u}) {
        const auto base = Program(elements);
        const auto baseDigest = Digest(base, ShaderStage::Fragment);
        for (std::size_t word = 0; word < base.bindings[1].guestDescriptor.size(); ++word) {
            auto changed = base;
            changed.bindings[1].guestDescriptor[word] ^= 0x10u;
            Require(Digest(changed, ShaderStage::Fragment) != baseDigest, "table word " + std::to_string(word) + " of " + std::to_string(elements) + " elements did not reach the digest");
        }
        for (std::size_t element = 0; element < elements; ++element) {
            auto changed = base;
            changed.bindings[1].imageSamplers[element] ^= 0x10u;
            Require(Digest(changed, ShaderStage::Fragment) != baseDigest, "sampler " + std::to_string(element) + " of " + std::to_string(elements) + " elements did not reach the digest");
        }
    }
}

// Random single-word and single-bit changes over the whole table: no two content keys share a digest.
void NoDigestSharedByTwoKeys() {
    const auto base = Program();
    std::mt19937_64 random(0x5eed);
    std::map<ShaderResources::ContentDigest, std::vector<std::uint32_t>> seen;
    for (int round = 0; round < 20000; ++round) {
        auto changed = base;
        auto& words = changed.bindings[1].guestDescriptor;
        const auto at = random() % words.size();
        words[at] ^= 1u << (random() % 32u);
        if (round % 2 == 1) words[random() % words.size()] = static_cast<std::uint32_t>(random());
        const auto [found, inserted] = seen.try_emplace(Digest(changed, ShaderStage::Fragment), Key(changed, ShaderStage::Fragment));
        Require(inserted || found->second == Key(changed, ShaderStage::Fragment), "two content keys share a digest");
    }
    Require(seen.size() > 15000, "the random changes produced too few distinct contents (" + std::to_string(seen.size()) + ")");
}

// The shared digest is the owner's own, found again for the same owner, and never served to a later
// object at the address of a released one.
void SharedDigestFollowsItsOwner() {
    const auto [hitsBefore, missesBefore] = ShaderResources::SharedContentDigestCounts();
    std::shared_ptr<const RecompileResult> owner(new RecompileResult(Program()));
    const auto first = ShaderResources::SharedContentKeyDigest(owner, ShaderStage::Fragment, true);
    Require(first == Digest(*owner, ShaderStage::Fragment), "the shared digest differs from the owner's digest");
    const RecompileResult copy = *owner;
    Require(ShaderResources::SharedContentKeyDigest(owner, ShaderStage::Fragment, true) == Digest(copy, ShaderStage::Fragment), "an owner's kept digest differs from its copy's");
    const auto [hitsAfter, missesAfter] = ShaderResources::SharedContentDigestCounts();
    Require(missesAfter - missesBefore == 1 && hitsAfter - hitsBefore == 1, "the second lookup of an owner did not find its digest");
    Require(ShaderResources::SharedContentKeyDigest(owner, ShaderStage::Vertex, true) == Digest(*owner, ShaderStage::Vertex), "an owner's digest for another stage was served from the first");

    // Owners allocated apart from their control block: a released owner's memory can then hold the
    // next one while the memo still names the old control block.
    std::uint32_t reused = 0;
    for (std::uint32_t round = 0; round < 64; ++round) {
        auto next = Program();
        next.bindings[1].guestDescriptor[round] ^= 0x8000'0000u;
        const auto* address = owner.get();
        owner.reset();
        owner = std::shared_ptr<const RecompileResult>(new RecompileResult(std::move(next)));
        if (owner.get() == address) ++reused;
        Require(ShaderResources::SharedContentKeyDigest(owner, ShaderStage::Fragment, true) == Digest(*owner, ShaderStage::Fragment), "a new owner got a released owner's digest (round " + std::to_string(round) + ")");
    }
    Require(reused != 0, "no new owner took a released owner's address, so the case is not tested");
    std::cout << "a new owner reused a released owner's address in " << reused << " of 64 rounds\n";
}

}

int main() {
    try {
        DigestFollowsTheContentKey();
        EveryTableWordCounts();
        NoDigestSharedByTwoKeys();
        SharedDigestFollowsItsOwner();
        std::cout << "Stage digests follow the content keys and their owners\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
