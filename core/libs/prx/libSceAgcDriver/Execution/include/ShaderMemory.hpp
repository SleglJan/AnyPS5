#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_SHADERMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_SHADERMEMORY_HPP

#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
struct SourceHandle;
class PreparedShaderInvocation;
}

namespace AgcDriver {

// A capture page's 1024 dword flags as 16 words: whole masks are ORed a word at a time and the
// regions are emitted run by run (countr_zero/countr_one), where std::bitset offered only shifts
// of the whole set and per-bit tests (~1,000 tests per page and region pass on an image table).
class PageBits {
public:
    static constexpr std::size_t Bits = 1024;
    static constexpr std::size_t Words = Bits / 64;

    bool test(std::size_t index) const { return ((words[index >> 6u] >> (index & 63u)) & 1u) != 0u; }
    void set(std::size_t index) { words[index >> 6u] |= std::uint64_t{1} << (index & 63u); }
    void set() { words.fill(~std::uint64_t{0}); }
    void reset() { words.fill(0); }
    bool all() const {
        for (const auto word : words) {
            if (word != ~std::uint64_t{0}) return false;
        }
        return true;
    }
    bool none() const {
        for (const auto word : words) {
            if (word != 0) return false;
        }
        return true;
    }
    std::size_t count() const {
        std::size_t bits = 0;
        for (const auto word : words) bits += static_cast<std::size_t>(std::popcount(word));
        return bits;
    }
    // Bits [first, first + count), within the page.
    void setRange(std::size_t first, std::size_t count) {
        while (count != 0) {
            const auto bit = first & 63u;
            const auto take = std::min<std::size_t>(count, 64u - bit);
            const auto mask = take == 64u ? ~std::uint64_t{0} : ((std::uint64_t{1} << take) - 1u) << bit;
            words[first >> 6u] |= mask;
            first += take;
            count -= take;
        }
    }
    // ORs 16 mask words in (bit i of word k is dword 64 k + i).
    void orWords(const std::uint64_t* mask) {
        for (std::size_t word = 0; word < Words; ++word) words[word] |= mask[word];
    }
    PageBits& operator|=(const PageBits& other) {
        orWords(other.words.data());
        return *this;
    }
    bool operator==(const PageBits& other) const = default;
    // The first set (clear) bit at or after `from`; Bits when there is none.
    std::size_t nextSet(std::size_t from) const { return next(from, 0u); }
    std::size_t nextClear(std::size_t from) const { return next(from, ~std::uint64_t{0}); }
    // emit(first, count) for every run of set bits, in order.
    template <class Emit>
    void forEachRun(Emit&& emit) const {
        for (auto first = nextSet(0); first < Bits;) {
            const auto end = nextClear(first);
            emit(first, end - first);
            first = nextSet(end);
        }
    }

private:
    std::size_t next(std::size_t from, std::uint64_t flip) const {
        if (from >= Bits) return Bits;
        auto index = from >> 6u;
        auto word = (words[index] ^ flip) & (~std::uint64_t{0} << (from & 63u));
        for (;;) {
            if (word != 0) return index * 64u + static_cast<std::size_t>(std::countr_zero(word));
            if (++index == Words) return Bits;
            word = words[index] ^ flip;
        }
    }

    std::array<std::uint64_t, Words> words{};
};

// Records the guest words a shader's resource analysis reads, as the memory regions its recompile
// request carries. Guest memory is fetched a 4 KiB page at a time; the regions report exactly the
// dwords read, so cache keys built from them do not change with unrelated bytes nearby. A page that
// `pendingWrite` reports as written by recorded GPU work is read word by word instead, so only a
// dword the capture needs waits for that work through the flush hook, not the whole page; a dword
// the query knows to be left unchanged by that work (Driver.cpp's written-element evidence) is read
// raw, without the hook. Debug aid: APS5_NO_WORDWISE_CAPTURE=1 fetches every page whole.
class ShaderMemory {
public:
    // The driver's answer for a range: nothing recorded writes it; recorded work writes it and the
    // read must go through the flush hook; recorded work writes it but leaves it unchanged, so the
    // bytes are read raw; VerifyRaw reads raw and through the hook, counting a difference;
    // KnownValue: recorded work writes it with bytes the driver already knows (a copy HLE's
    // destination, Driver.cpp's known-value ring entries), copied into `known` when the caller
    // passes a span of exactly `bytes` (empty: the answer alone); VerifyKnownValue serves them and
    // reads through the hook too, counting a difference.
    enum class PendingWrite : std::uint8_t { None, Sync, Raw, VerifyRaw, KnownValue, VerifyKnownValue };
    using PendingWriteQuery = PendingWrite (*)(std::uint64_t address, std::size_t bytes, std::span<std::byte> known);
    // Words served from known values by every capture, and the verified ones and their mismatches
    // (APS5_VERIFY_KNOWN_VALUES=1), for Driver.cpp's [copy] line.
    struct KnownValueCounts {
        std::uint64_t served, verified, mismatches;
    };
    static KnownValueCounts KnownValues();
    // Told, for a dword read that went through the hook, whether the bytes read before the hook's
    // wait were still there after it (the driver's evidence for later raw reads).
    using PendingWriteObserver = void (*)(std::uint64_t address, bool unchanged);
    // The calling thread's count of hook syncs that waited for unfinished GPU work: a read is only
    // observed when the count moved across it (against finished work both reads see the GPU's
    // bytes, and "unchanged" would be no evidence). Null observes every read that went through
    // the hook.
    using HookWaitCounter = std::uint64_t (*)();
    // The calling thread's GPU waits so far in milliseconds (Recorder::ThreadWaitedMs), set once
    // by the driver: the [capture] line charges the waits made inside a capture to it.
    using WaitedMsProvider = double (*)();
    static void SetWaitedMsProvider(WaitedMsProvider provider);
    explicit ShaderMemory(std::span<const ShaderRecompiler::MemoryRegion> initial, PendingWriteQuery pendingWrite = nullptr, PendingWriteObserver observe = nullptr, HookWaitCounter hookWaits = nullptr);
    ShaderMemory(const ShaderMemory&) = delete;
    ShaderMemory& operator=(const ShaderMemory&) = delete;
    // A pipelined draw's commit owns its capture's pages (DrawPipeline): every member moves, the
    // replay state included (std::map moves its nodes, so lastPage and replayLast keep pointing at
    // the same pages, now in this object's map; a replayed page's shared image moves with its
    // node). The moved-from object is left empty, with its shortcuts cleared.
    ShaderMemory(ShaderMemory&& other) noexcept;
    ShaderMemory& operator=(ShaderMemory&&) = delete;
    // Returns what the capture resolved (plan, snapshot, specialization) for
    // ShaderRecompiler::Recompile(request, capture), which then skips its own materialization. One
    // result per call: the draw path captures several stages on one ShaderMemory. With `handle`
    // (the driver's memoized ShaderRecompiler::ResolveSource result) the capture skips the source
    // resolution.
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> Capture(const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::SourceHandle* handle = nullptr);
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> Capture(const ShaderRecompiler::PreparedShaderInvocation& invocation);
    [[nodiscard]] std::vector<ShaderRecompiler::MemoryRegion> Regions() const;
    // The page regions read since the previous call (or construction), a word read again
    // included, the initial regions excluded: one stage's own reads on the draw path's shared
    // ShaderMemory (Regions() stays the union). The spans point into the pages, as Regions()'s do.
    [[nodiscard]] std::vector<ShaderRecompiler::MemoryRegion> TakeRecentRegions();
    // The driver's source handle memo outcomes, for the [capture] line (APS5_PROFILE_DRAW).
    static void CountHandleMemo(bool hit);

private:
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture(const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::SourceHandle* handle, const ShaderRecompiler::PreparedShaderInvocation* invocation);
    static constexpr std::size_t PageBytes = 4096;
    static constexpr std::size_t PageWords = PageBytes / sizeof(std::uint32_t);

    struct Page {
        Page() {}
        // The page's own dwords (a fetch fills it whole, a word-wise page a word at a time), allocated
        // at first use and not zero-filled: only valid dwords are ever read. A page an image-table
        // snapshot replayed (tableReplayPage) refers to the snapshot memo's image instead, shared and
        // never written (the page is valid whole), so its node stays small (a 4 KiB array in every
        // node made each replayed page a large allocation, freed and faulted in again per capture).
        std::unique_ptr<std::uint32_t[]> own;
        PageBits valid;
        PageBits read;
        PageBits recent;
        bool wordwise = false;
        std::shared_ptr<const void> imageOwner;
        const std::uint32_t* image = nullptr;
        const std::uint32_t* data() const { return image != nullptr ? image : own.get(); }
        std::uint32_t* words() {
            if (own == nullptr) own.reset(new std::uint32_t[PageWords]);
            return own.get();
        }
    };

    static bool read(void* context, std::uint64_t address, std::uint32_t* value);
    // `count` reads in address order: a page segment already fetched whole, with no initial region
    // over it, is copied and its read/recent bits set as a range; anything else takes `read`.
    static bool readWords(void* context, std::uint64_t address, std::uint32_t* values, std::uint64_t count);
    static bool accessible(void* context, std::uint64_t address, std::uint64_t bytes);
    static bool pending(void* context, std::uint64_t address, std::uint64_t bytes);
    // ShaderRecompiler::SrtTableMemory for image-table snapshots that skip an unchanged table.
    static std::uint64_t tableCollect(void* context, std::uint64_t address, std::uint64_t bytes);
    static bool tableUnchanged(void* context, std::uint64_t address, std::uint64_t bytes, std::uint64_t generation);
    static bool tableReadPage(void* context, std::uint64_t base, std::uint32_t* words);
    static bool tableReplayPage(void* context, std::uint64_t base, const std::uint32_t* words, const std::uint64_t* mask, const std::shared_ptr<const void>& owner);
    // Whether an initial region overlaps [begin, end).
    bool overlapsInitial(std::uint64_t begin, std::uint64_t end) const;
    Page& page(std::uint64_t base);

    // Regions given at construction (the registered shader's code and header), referenced as given:
    // the caller keeps them alive for as long as the capture is used.
    std::map<std::uint64_t, std::span<const std::byte>> initial;
    std::map<std::uint64_t, Page> pages;
    // The last page `read` looked up when no initial region overlaps it (NoPage otherwise): its
    // fetched words are served without the map lookups.
    static constexpr std::uint64_t NoPage = ~std::uint64_t{0};
    std::uint64_t lastBase = NoPage;
    Page* lastPage = nullptr;
    // The page tableReplayPage took last: a column's pages come in address order, so the next one
    // is found or inserted beside it instead of by a walk down the tree; and the ranges
    // tableUnchanged proved accessible whole in this capture, whose pages a replay takes without
    // asking again (APS5_NO_REPLAY_SHORTCUTS=1: the tree walk and a query per page).
    std::map<std::uint64_t, Page>::iterator replayLast{};
    bool replayHint = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> provenSpans;
    // The proven span that covered the last replayed page (pages come in span order).
    std::size_t provenHint = 0;
    PendingWriteQuery pendingWrite = nullptr;
    PendingWriteObserver observe = nullptr;
    HookWaitCounter hookWaits = nullptr;
};

// The positions, among a dispatch-cache variant's stored words, of the pure flat-SRT leaves a
// capture read (ShaderRecompiler::SrtReadTrace), for a hit that differs only there. `runs` are the
// variant's [begin, end) byte ranges in address order and `words` their dwords in that order;
// `leaves` are (flat offset, address) and `otherReads` the sorted addresses of every other read.
// A leaf is skipped when its address is among the other reads (a walk read aliases the dword,
// counted `aliased`), is not dword-aligned or lies in no run (counted `unmapped`), or its stored
// word differs from the flattened SRT's at that offset (counted `mismatched`). `positions`
// (sorted, may repeat) and `slots` (the flat offset of each) are parallel.
struct DataWordPositionCounts {
    std::uint64_t unmapped = 0;
    std::uint64_t mismatched = 0;
    std::uint64_t aliased = 0;
};
DataWordPositionCounts DataWordPositions(std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, std::span<const std::pair<std::uint32_t, std::uint64_t>> leaves, std::span<const std::uint64_t> otherReads, std::span<const std::uint32_t> words, std::span<const std::uint32_t> flattenedSrt, std::vector<std::uint32_t>& positions, std::vector<std::uint32_t>& slots);

}

#endif
