#include "BdaTests.hpp"
#include "BdaPageReference.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestHeap.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <new>
#include <string>

#ifndef _WIN32
#include <sys/mman.h>
#endif
namespace {

using namespace AgcDriver::Graphics;
using Role = ShaderRecompiler::DescriptorRole;

template<typename TAction>
void reject(TAction action, const char* reason) {
    try { action(); }
    catch (const std::runtime_error& error) {
        Require(std::string(error.what()).find(reason) != std::string::npos, std::string("unexpected BDA test error: ") + error.what());
        return;
    }
    throw std::runtime_error(std::string("expected BDA rejection: ") + reason);
}

void importCrossingTests(const Context& context, const BdaTestAccess& access) {
    constexpr std::size_t half = 65536;
    void* block = ::operator new(2 * half, std::align_val_t{half});
    auto* guest = static_cast<std::uint8_t*>(block);
    std::memset(guest, 0x11, half);
    std::memset(guest + half, 0x22, half);
    const auto first = reinterpret_cast<std::uintptr_t>(block);
    const auto second = first + half;
    auto importing = context;
    importing.hostImportAlignment = half;
    const auto registry = [&](bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        for (auto* range : {guest, guest + half}) {
            if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, half, true, false, true);
            else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
        }
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    registry(true);
    {
        GuestBufferMemory leased(importing);
        leased.AcquireRegistered();
        Require(HostImportCovers(importing, first, half) && HostImportCovers(importing, second, half), "the registered ranges were not imported");
        const auto secondImport = *HostImportFor(importing, second, half);
        leased.AddInPlace(second - 16, 32, false);
        leased.AddInPlace(second + 64, 16, false);
        leased.Upload();
        std::uint32_t adjustment = 0;
        const auto crossing = leased.Address(second - 16, 32, adjustment);
        Require(crossing.second <= access.addressBytes(crossing.first).size(), "a view crossing into the next registered range is addressed past its host import");
        Require(secondImport.base == first && secondImport.bytes == 2 * half, "adjacent registered ranges with the same access were not imported as one span");
        Require(crossing.first + adjustment == secondImport.address + (second - 16 - secondImport.base), "a view crossing into the next registered range is not addressed in place in the spanning import");
        const auto inside = leased.Address(second + 64, 16, adjustment);
        const auto insideBytes = access.addressBytes(inside.first);
        Require(inside.second <= insideBytes.size() && (inside.first + adjustment == secondImport.address + (second + 64 - secondImport.base) || insideBytes[adjustment] == std::byte{0x22}), "a view in the second registered range is addressed past its buffer or misses its bytes");
        for (const auto& range : leased.AddressRanges()) {
            if (range.end <= first || range.begin >= second + half) continue;
            const auto mapped = access.addressBytes(range.deviceAddress);
            Require(mapped.size() >= range.end - range.begin, "a BDA range runs past the end of its host import");
            if (range.begin <= second && second < range.end && (range.deviceAddress < secondImport.address || range.deviceAddress >= secondImport.address + secondImport.bytes)) Require(mapped[second - range.begin] == std::byte{0x22}, "the BDA range over the second registered range misses its bytes");
        }
        leased.Release();
    }
    registry(false);
    Require(HostImportFor(importing, first, half) == nullptr && !HostImportCovers(importing, second, half), "the host imports outlived their ranges");
    ::operator delete(block, std::align_val_t{half});
}

void importSpanTests(const Context& context) {
    constexpr std::size_t piece = 65536;
    void* block = ::operator new(3 * piece, std::align_val_t{piece});
    auto* guest = static_cast<std::uint8_t*>(block);
    const auto first = reinterpret_cast<std::uintptr_t>(block);
    auto importing = context;
    importing.hostImportAlignment = piece;
    const auto change = [&](std::uint8_t* range, bool add, bool writable) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, piece, true, writable, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    change(guest, true, false);
    change(guest + piece, true, false);
    change(guest + 2 * piece, true, true);
    const auto* span = HostImportFor(importing, first + piece - 16, 32);
    Require(span != nullptr && span->base == first && span->bytes == 2 * piece, "adjacent read-only ranges were not imported as one span");
    const auto* writable = HostImportFor(importing, first + 2 * piece, piece);
    Require(writable != nullptr && writable->base == first + 2 * piece && writable->bytes == piece, "a writable range joined the read-only span");
    Require(HostImportFor(importing, first + 2 * piece - 16, 32) == nullptr, "a view crossing ranges of different access was imported");
    change(guest + piece, false, false);
    const auto* alone = HostImportFor(importing, first, piece);
    Require(alone != nullptr && alone->base == first && alone->bytes == piece, "the span outlived the removal of one of its ranges");
    Require(HostImportFor(importing, first + piece, 16) == nullptr, "the removed range is still imported");
    change(guest, false, false);
    change(guest + 2 * piece, false, false);
    ::operator delete(block, std::align_val_t{piece});
}

void failedImportRemapTests(const Context& context, const BdaTestAccess& access) {
    constexpr std::size_t piece = 65536;
    auto* block = static_cast<std::byte*>(::operator new(4 * piece, std::align_val_t{piece}));
    auto importing = context;
    importing.hostImportAlignment = piece;
    const auto address = reinterpret_cast<std::uintptr_t>(block);
    const auto add = [&](std::size_t index) {
        GuestAllocations::Mutation mutation;
        mutation.Add(block + index * piece, piece, true, true, true);
    };
    const auto remove = [&](std::size_t index) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block + index * piece);
    };
    SetHostImportWindow(0);
    for (std::size_t replaced = 0; replaced < 2; ++replaced) {
        add(0);
        add(1);
        std::weak_ptr<const GuestAllocations::Range> owner;
        {
            const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
            const auto found = std::find_if(lease.begin(), lease.end(), [&](const auto& range) { return range->address == address + replaced * piece; });
            Require(found != lease.end(), "failed import fixture has no registered owner");
            owner = *found;
        }
        access.limitMemory(0);
        const auto attempts = access.allocationAttempts();
        Require(HostImportFor(importing, address, 2 * piece) == nullptr && access.allocationAttempts() == attempts + 1, "host import failure was not exercised");
        Require(owner.use_count() == 1, "a failed import pins its owner");
        Require(HostImportFor(importing, address, 2 * piece) == nullptr && access.allocationAttempts() == attempts + 1, "an unchanged failed import was retried");
        add(3);
        Require(HostImportFor(importing, address, 2 * piece) == nullptr && access.allocationAttempts() == attempts + 1, "an unrelated mapping retried a failed import");
        remove(replaced);
        Require(owner.expired(), "a failed import kept the replaced owner alive");
        add(replaced);
        access.limitMemory(std::nullopt);
        const auto* imported = HostImportFor(importing, address, 2 * piece);
        Require(imported != nullptr && imported->base == address && imported->bytes == 2 * piece && access.allocationAttempts() == attempts + 2, "same-address remap retained a failed import decision");
        remove(0);
        remove(1);
        remove(3);
        ClearHostImports(importing.device);
    }
    add(0);
    add(1);
    access.limitMemory(0);
    Require(HostImportFor(importing, address, 2 * piece) == nullptr, "span failure was not exercised");
    remove(1);
    access.limitMemory(std::nullopt);
    const auto* shortened = HostImportFor(importing, address, piece);
    Require(shortened != nullptr && shortened->bytes == piece, "a failed span suppressed an import of its remaining range");
    remove(0);
    ClearHostImports(importing.device);
    ::operator delete(block, std::align_val_t{piece});
}

void spaceImportRunTests(const Context& context) {
    constexpr std::size_t piece = 65536;
    void* block = ::operator new(2 * piece, std::align_val_t{piece});
    const auto first = reinterpret_cast<std::uintptr_t>(block);
    std::memset(block, 0x3c, 2 * piece);
    auto importing = context;
    importing.hostImportAlignment = piece;
    const auto change = [&](std::uint8_t* range, bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, piece, true, false, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    auto* guest = static_cast<std::uint8_t*>(block);
    change(guest, true);
    change(guest + piece, true);
    for (int build = 0; build < 2; ++build) {
        const auto before = AddressSpaceCounters();
        GuestBufferMemory leased(importing);
        leased.AcquireRegistered();
        leased.AddInPlace(first + piece - 256, 512, false);
        leased.Upload();
        std::uint32_t adjustment = 0;
        const auto view = leased.Address(first + piece - 256, 512, adjustment);
        Require(view.first != 0 && view.second == 512 + adjustment, "a V# across one import's ranges has no device address");
        const auto ranges = leased.AddressRanges();
        Require(std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin <= first && first + 2 * piece <= range.end; }), "one import's adjacent ranges are not one BDA table entry");
        leased.Release();
        const auto after = AddressSpaceCounters();
        if (after.enabled && build == 1) Require(after.hits == before.hits + 1 && after.dissolvedOverlap == before.dissolvedOverlap, "a V# across one import's ranges dissolved the cached address space");
    }
    change(guest, false);
    change(guest + piece, false);
    ::operator delete(block, std::align_val_t{piece});
}

void importWindowTests(const Context& context) {
    constexpr std::size_t piece = 65536;
    constexpr std::size_t window = 2 * piece;
    void* block = ::operator new(4 * piece, std::align_val_t{window});
    auto* guest = static_cast<std::uint8_t*>(block);
    const auto first = reinterpret_cast<std::uintptr_t>(block);
    auto importing = context;
    importing.hostImportAlignment = piece;
    SetHostImportWindow(window);
    const auto change = [&](std::uint8_t* range, bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, piece, true, false, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    for (std::size_t index = 0; index < 3; ++index) change(guest + index * piece, true);
    const auto* low = HostImportFor(importing, first + piece - 16, 32);
    Require(low != nullptr && low->base == first && low->bytes == window, "adjacent ranges inside one window were not imported as one span");
    const auto lowBuffer = low->buffer;
    const auto* high = HostImportFor(importing, first + window, piece);
    Require(high != nullptr && high->base == first + window && high->bytes == piece, "a span crossed the import window");
    Require(HostImportFor(importing, first + window - 16, 32) == nullptr, "a view crossing the import window was imported");
    change(guest + 3 * piece, true);
    const auto* grown = HostImportFor(importing, first + window, window);
    Require(grown != nullptr && grown->base == first + window && grown->bytes == window, "the next window did not grow into the new range");
    const auto* kept = HostImportFor(importing, first, 16);
    Require(kept != nullptr && kept->buffer == lowBuffer, "growth in one window re-imported another");
    for (std::size_t index = 0; index < 4; ++index) change(guest + index * piece, false);
    SetHostImportWindow(0);
    ::operator delete(block, std::align_val_t{window});
}

void importJoinTests(const Context& context) {
    constexpr std::size_t piece = 65536;
    constexpr std::size_t window = 2 * piece;
    constexpr std::size_t large = 3 * piece;
    void* block = ::operator new(3 * large + 5 * piece, std::align_val_t{window});
    auto* guest = static_cast<std::uint8_t*>(block);
    std::memset(guest, 0x11, 3 * large);
    const auto first = reinterpret_cast<std::uintptr_t>(block);
    const auto second = first + large;
    const auto third = second + large;
    auto importing = context;
    importing.hostImportAlignment = piece;
    SetHostImportWindow(window);
    const auto change = [&](std::uint8_t* range, std::size_t bytes, bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, bytes, true, false, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    const auto bind = [&](std::uintptr_t address, std::size_t bytes = 32) {
        GuestBufferMemory plain(importing);
        plain.AddInPlace(address, bytes, false);
        plain.Upload();
        std::uint32_t adjustment = 0;
        const auto view = plain.Descriptor(address, bytes, adjustment);
        plain.Release();
        return view.buffer;
    };
    change(guest, large, true);
    change(guest + large, large, true);
    const auto* low = HostImportFor(importing, first, 16);
    Require(low != nullptr && low->base == first && low->bytes == large, "a range larger than the import window was not imported alone");
    const auto* high = HostImportFor(importing, second, 16);
    Require(high != nullptr && high->base == second && high->bytes == large, "the next range larger than the import window was not imported alone");
    Require(HostImportFor(importing, second - 16, 32) == nullptr, "a view across two imported ranges was imported before it was bound");
    {
        GuestBufferMemory leased(importing);
        leased.AcquireRegistered();
        leased.AddInPlace(second - 16, 32, false);
        leased.Upload();
        const auto* joined = HostImportFor(importing, second - 16, 32);
        Require(joined != nullptr && joined->base == first && joined->bytes == 2 * large, "a buffer bound across two adjacent ranges larger than the import window was not imported as one");
        std::uint32_t adjustment = 0;
        const auto crossing = leased.Descriptor(second - 16, 32, adjustment);
        Require(crossing.buffer == joined->buffer && crossing.offset + adjustment == second - 16 - first, "a buffer bound across the joined ranges is not bound in place");
        leased.Release();
    }
    const auto joinedBuffer = HostImportFor(importing, first, 16)->buffer;
    Require(bind(second - 16) == joinedBuffer, "a later bind across the joined ranges did not use the joined import");
    auto* heap = guest + 3 * large + 2 * piece;
    change(heap, piece, true);
    change(heap + piece, piece, true);
    reject([&] { static_cast<void>(bind(reinterpret_cast<std::uintptr_t>(heap + piece) - 16)); }, "has no host import");
    Require(HostImportFor(importing, reinterpret_cast<std::uintptr_t>(heap + piece) - 16, 32) == nullptr, "two pieces no larger than the import window were joined");
    const auto heapBase = reinterpret_cast<std::uintptr_t>(heap);
    const auto piecesBuffer = bind(heapBase, 2 * piece);
    const auto* pieces = HostImportFor(importing, heapBase, 2 * piece);
    Require(pieces != nullptr && pieces->base == heapBase && pieces->bytes == 2 * piece && piecesBuffer == pieces->buffer, "a buffer bound exactly over two pieces no larger than the import window was not imported as one");
    change(heap + 2 * piece, piece, true);
    const auto acrossBuffer = bind(heapBase + 2 * piece - 16);
    const auto* across = HostImportFor(importing, heapBase + 2 * piece - 16, 32);
    Require(across != nullptr && across->base <= heapBase + 2 * piece - 16 && heapBase + 2 * piece + 16 <= across->base + across->bytes && acrossBuffer == across->buffer, "a buffer bound across the end of an exact join was bound in an import that does not cover it");
    const auto* kept = HostImportFor(importing, first, 16);
    Require(kept != nullptr && kept->buffer == joinedBuffer && kept->bytes == 2 * large, "a registry change elsewhere re-imported the joined ranges");
    change(guest + 2 * large, large, true);
    Require(HostImportFor(importing, first, 16)->buffer == joinedBuffer, "a range registered next to the joined ranges re-imported them");
    const auto grownBuffer = bind(third - 16);
    const auto* grown = HostImportFor(importing, third - 16, 32);
    Require(grown != nullptr && grown->base == first && grown->bytes == 3 * large && grownBuffer == grown->buffer, "a bind across the next range did not grow the joined import");
    Require(bind(second - 16) == grownBuffer && HostImportFor(importing, first, 16)->buffer == grownBuffer, "two joins sharing a range retired each other");
    change(heap, piece, false);
    change(heap + piece, piece, false);
    change(heap + 2 * piece, piece, false);
    change(guest + large, large, false);
    const auto* alone = HostImportFor(importing, first, 16);
    Require(alone != nullptr && alone->base == first && alone->bytes == large, "the joined import outlived one of its ranges");
    change(guest, large, false);
    change(guest + 2 * large, large, false);
    Require(HostImportFor(importing, first, 16) == nullptr && HostImportFor(importing, third, 16) == nullptr, "the host imports outlived their ranges");
    SetHostImportWindow(0);
    ::operator delete(block, std::align_val_t{window});
}

class WindowedRange {
public:
    WindowedRange(const Context& context, std::size_t window, std::size_t cells, std::uint64_t budget = 0) : window(window), bytes(window * cells), importing(context) {
        block = ::operator new(bytes, std::align_val_t{piece});
        std::memset(block, 0x33, bytes);
        importing.hostImportAlignment = piece;
        SetHostImportRangeWindow(window);
        SetHostImportBudget(budget);
        change(true);
    }
    ~WindowedRange() {
        change(false);
        SetHostImportBudget(0);
        SetHostImportRangeWindow(std::uint64_t{64} << 20u);
        ::operator delete(block, std::align_val_t{piece});
    }
    WindowedRange(const WindowedRange&) = delete;
    WindowedRange& operator=(const WindowedRange&) = delete;

    std::uintptr_t First() const { return reinterpret_cast<std::uintptr_t>(block); }

    static constexpr std::size_t piece = 65536;
    const std::size_t window;
    const std::size_t bytes;
    Context importing;

private:
    void change(bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, bytes, true, false, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    }

    void* block = nullptr;
};

void importWholeRangeOverWindowTests(const Context& context) {
    WindowedRange range(context, 2 * WindowedRange::piece, 2);
    const auto first = range.First();
    const auto* windowed = HostImportFor(range.importing, first + 16, 32);
    Require(windowed != nullptr && windowed->base == first && windowed->bytes == range.window, "an access to the first window of a large range was not imported as that window");
    GuestBufferMemory leased(range.importing);
    leased.AcquireRegistered();
    leased.Upload();
    std::uint32_t adjustment = 0;
    const auto view = leased.Descriptor(first + range.bytes - WindowedRange::piece + 16, 32, adjustment);
    Require(view.range == 32 + adjustment, "a view in the last window of a large registered range has no descriptor");
    const auto* whole = HostImportFor(range.importing, first, range.bytes);
    Require(whole != nullptr && whole->base == first && whole->bytes == range.bytes && view.buffer == whole->buffer, "an address-based build was served by the range's first window, not an import of the whole range");
    leased.Release();
}

void importRefusedWholeRangeTests(const Context& context) {
    WindowedRange range(context, 2 * WindowedRange::piece, 4, 3 * 2 * WindowedRange::piece);
    const auto first = range.First();
    const auto* windowed = HostImportFor(range.importing, first + 16, 32);
    Require(windowed != nullptr && windowed->bytes == range.window, "the first window of a range over the budget was not imported");
    const auto buffer = windowed->buffer;
    for (int build = 0; build < 3; ++build) {
        GuestBufferMemory leased(range.importing);
        reject([&] { leased.AcquireRegistered(); }, "has no host import");
        Require(HostImportCovers(range.importing, first + 16, 32), "an address-based build refused its whole-range import and retired the range's window (build " + std::to_string(build) + ")");
    }
    const auto* kept = HostImportFor(range.importing, first + 16, 32);
    Require(kept != nullptr && kept->buffer == buffer, "the range's window was imported again after a refused whole-range import");
}

void importNeighbouringEdgesTests(const Context& context) {
    WindowedRange range(context, 2 * WindowedRange::piece, 3);
    const auto first = range.First();
    const auto use = [&](std::uintptr_t address) {
        GuestBufferMemory leased(range.importing);
        leased.AddInPlace(address, 32, false);
        leased.Upload();
        leased.Release();
    };
    const auto acrossFirst = first + range.window - 16;
    const auto acrossSecond = first + 2 * range.window - 16;
    use(acrossFirst);
    Require(HostImportCovers(range.importing, acrossFirst, 32), "a use across the first cell edge was not imported");
    use(acrossSecond);
    Require(HostImportCovers(range.importing, first, range.bytes), "a use across the second cell edge did not grow the first use's window to their union");
    const auto serial = HostImportSerial(range.importing, first, range.bytes, false);
    for (int round = 0; round < 10; ++round) {
        use(acrossFirst);
        use(acrossSecond);
        Require(HostImportSerial(range.importing, first, range.bytes, false) == serial, "a use across a cell edge replaced the shared window (round " + std::to_string(round) + ")");
    }
}

void importAtMapOverBudgetTests(const Context& context) {
    constexpr std::size_t bytes = 65536;
    auto importing = context;
    importing.hostImportAlignment = bytes;
    void* kept = ::operator new(bytes, std::align_val_t{bytes});
    void* used = ::operator new(bytes, std::align_val_t{bytes});
    void* mapped = ::operator new(bytes, std::align_val_t{bytes});
    const auto change = [&](void* block, bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, bytes, true, false, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    change(kept, true);
    change(used, true);
    const auto keptAddress = reinterpret_cast<std::uintptr_t>(kept);
    const auto usedAddress = reinterpret_cast<std::uintptr_t>(used);
    Require(HostImportFor(importing, keptAddress, 16) != nullptr, "the import to keep was not made");
    for (int use = 0; use < 300; ++use) Require(HostImportFor(importing, usedAddress, 16) != nullptr, "the import in use was not made");
    {
        GuestBufferMemory leased(importing);
        leased.AddInPlace(usedAddress, 16, false);
        leased.Upload();
        leased.Release();
    }
    const auto live = HostImportLiveBytes();
    SetHostImportBudget(live + bytes - 1);
    change(mapped, true);
    GuestAllocations::Mapped ranges;
    {
        const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        for (const auto& range : lease) {
            if (range->address == reinterpret_cast<std::uintptr_t>(mapped)) ranges.push_back(range);
        }
    }
    Require(ranges.size() == 1, "the mapped range is not registered");
    static_cast<void>(ImportMappedRanges(importing, ranges, GuestAllocations::GuestAllocationsGeneration_nid_postfix(), true));
    SetHostImportBudget(0);
    Require(HostImportLiveBytes() == live, "a range mapped at the budget retired an import at map time (an eviction without the GPU lock)");
    Require(HostImportCovers(importing, keptAddress, 16) && !HostImportCovers(importing, reinterpret_cast<std::uintptr_t>(mapped), 16), "a range mapped at the budget was imported, or the import to keep was retired");
    for (auto* block : {kept, used, mapped}) change(block, false);
    for (auto* block : {kept, used, mapped}) ::operator delete(block, std::align_val_t{bytes});
}

void importEvictionTests(const Context& context) {
    constexpr std::size_t bytes = 65536;
    auto importing = context;
    importing.hostImportAlignment = bytes;
    void* idle = ::operator new(bytes, std::align_val_t{bytes});
    void* used = ::operator new(bytes, std::align_val_t{bytes});
    void* fresh = ::operator new(bytes, std::align_val_t{bytes});
    const auto change = [&](void* block, bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, bytes, true, false, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    for (auto* block : {idle, used, fresh}) change(block, true);
    const auto idleAddress = reinterpret_cast<std::uintptr_t>(idle);
    const auto usedAddress = reinterpret_cast<std::uintptr_t>(used);
    const auto freshAddress = reinterpret_cast<std::uintptr_t>(fresh);
    Require(HostImportFor(importing, idleAddress, 16) != nullptr, "the idle import was not made");
    for (int use = 0; use < 300; ++use) Require(HostImportFor(importing, usedAddress, 16) != nullptr, "the import in use was not made");
    {
        GuestBufferMemory leased(importing);
        leased.AddInPlace(usedAddress, 16, false);
        leased.Upload();
        leased.Release();
    }
    const auto live = HostImportLiveBytes();
    const auto budget = live + bytes - 1;
    SetHostImportBudget(budget);
    const auto* made = HostImportFor(importing, freshAddress, 16);
    const auto after = HostImportLiveBytes();
    SetHostImportBudget(0);
    Require(made != nullptr, "an import past the budget was refused instead of retiring an import nothing used recently");
    Require(after <= budget, "an import past the budget left the live imports over the budget");
    Require(HostImportCovers(importing, usedAddress, 16), "an import past the budget retired the import the last build used");
    for (auto* block : {idle, used, fresh}) change(block, false);
    for (auto* block : {idle, used, fresh}) ::operator delete(block, std::align_val_t{bytes});
}

ShaderRecompiler::DescriptorBinding binding(Role role, std::uint32_t slot) {
    return {ShaderRecompiler::DescriptorKind::StorageBuffer, role, 0, slot, 1, {}, false};
}

void importedFreshTests(const Context& context) {
#ifndef _WIN32
    if (!GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix()) {
        std::cout << "write watch unavailable: imported never-collected pages not tested\n";
        return;
    }
    constexpr std::size_t blockBytes = 65536;
    void* area = mmap(nullptr, 2 * blockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(area != MAP_FAILED, "cannot map the imported block");
    const auto base = (reinterpret_cast<std::uintptr_t>(area) + blockBytes - 1) & ~(blockBytes - 1);
    std::memset(reinterpret_cast<void*>(base), 0x55, blockBytes);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(reinterpret_cast<void*>(base), blockBytes);
    auto importing = context;
    importing.hostImportAlignment = blockBytes;
    const auto registry = [&](bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, reinterpret_cast<void*>(base), blockBytes, true, true, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, reinterpret_cast<void*>(base));
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    registry(true);
    if (HostImportFor(importing, base, blockBytes) == nullptr) {
        std::cout << "the never-collected block was not imported: imported never-collected pages not tested\n";
    } else {
        const auto gpuWrite = AgcDriver::GuestMemory::MarkWritten(base, 4);
        Require(gpuWrite != 0, "a GPU write into an imported never-collected block was not stamped");
        reinterpret_cast<volatile std::uint8_t*>(base)[5 * 4096 + 8] = 0x66;
        static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(base, blockBytes));
        Require(AgcDriver::GuestMemory::WrittenSince(base + 5 * 4096, 4096, gpuWrite), "a CPU write to an imported page protected after the import was not seen");
    }
    registry(false);
    Require(HostImportFor(importing, base, blockBytes) == nullptr, "the import outlived its range");
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(reinterpret_cast<void*>(base), blockBytes);
    munmap(area, 2 * blockBytes);
#else
    static_cast<void>(context);
#endif
}

void importWatchTests() {
#ifndef _WIN32
    if (!GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix()) {
        std::cout << "write watch unavailable: import watch not tested\n";
        return;
    }
    constexpr std::size_t bytes = 4 * 65536;
    void* block = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(block != MAP_FAILED, "cannot map the import watch block");
    auto* guest = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uintptr_t>(block);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    std::memset(block, 0x22, bytes);
    const auto written = [&] {
        std::size_t reported = 0;
        Require(GuestWriteWatch::GuestWriteWatchCollect_nid_postfix(address, bytes, [](void* context, std::uintptr_t begin, std::uintptr_t end) { *static_cast<std::size_t*>(context) += end - begin; }, &reported), "the write watch did not collect the block");
        return reported;
    };
    Require(AgcDriver::GuestMemory::ImportWatched(address, bytes, [] { return true; }), "importing a block nothing had collected failed");
    Require(written() == bytes, "an import write-protected pages nothing had collected");
    guest[3 * 4096] = 0x33;
    Require(AgcDriver::GuestMemory::ImportWatched(address, bytes, [] { return true; }), "importing a collected block failed");
    Require(written() == 0, "an import did not take the writes to pages a collect had protected");
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
    munmap(block, bytes);
    constexpr std::size_t blockBytes = 65536;
    void* area = mmap(nullptr, bytes + 2 * blockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(area != MAP_FAILED, "cannot map the GPU write block");
    const auto aligned = (reinterpret_cast<std::uintptr_t>(area) + blockBytes - 1) & ~(blockBytes - 1);
    const auto imported = aligned + 4096;
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(reinterpret_cast<void*>(imported), bytes);
    Require(AgcDriver::GuestMemory::ImportWatched(imported, bytes, [] { return true; }), "importing a fresh range failed");
    const auto gpuWrite = AgcDriver::GuestMemory::MarkWritten(imported + 4096, 4096);
    Require(gpuWrite != 0, "a GPU write into the imported range was not stamped");
    static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(imported, bytes));
    Require(!AgcDriver::GuestMemory::WrittenSince(imported + 4096, 4096, gpuWrite), "a collect after a GPU write into an uncollected import read as a CPU write over the GPU's results");
    reinterpret_cast<volatile std::uint8_t*>(imported)[4096 + 100] = 0x44;
    static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(imported, bytes));
    Require(AgcDriver::GuestMemory::WrittenSince(imported + 4096, 4096, gpuWrite), "a CPU write after a GPU write into an import was not seen");
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(reinterpret_cast<void*>(imported), bytes);
    munmap(area, bytes + 2 * blockBytes);

    void* labels = mmap(nullptr, 2 * blockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(labels != MAP_FAILED, "cannot map the label block");
    const auto label = (reinterpret_cast<std::uintptr_t>(labels) + blockBytes - 1) & ~(blockBytes - 1);
    auto* words = reinterpret_cast<volatile std::uint32_t*>(label);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(reinterpret_cast<void*>(label), blockBytes);
    words[0] = 1;
    static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(label, 4));
    const auto before = AgcDriver::GuestMemory::TrackerGeneration();
    Require(AgcDriver::GuestMemory::MarkWritten(label, 4) != 0, "a GPU write next to never-collected pages was not stamped");
    Require(AgcDriver::GuestMemory::UnchangedSinceCollected(label, 4, before), "a GPU write next to never-collected pages read as a CPU write of its block");
    Require(AgcDriver::GuestMemory::StoredOver(label + 8 * 4096, 4096, before), "a never-collected page a GPU write collected is not reported as possibly stored");
    const std::array<std::uint64_t, 1> since{before};
    std::array<std::uint8_t, 1> changed{};
    std::array<std::uint8_t, 1> cpu{};
    Require(AgcDriver::GuestMemory::ChangedBlocks(label, blockBytes, since, changed, cpu) && changed[0] == AgcDriver::GuestMemory::BlockWritten && cpu[0] == 0, "collecting never-collected pages at a GPU write stamped their block as written by the CPU");
    const auto now = AgcDriver::GuestMemory::TrackerGeneration();
    Require(AgcDriver::GuestMemory::MarkWritten(label, 4) == now + 1, "a GPU write into a block with nothing left to collect advanced the generation more than once");
    words[1] = 2;
    Require(!AgcDriver::GuestMemory::UnchangedSinceCollected(label, 4, before), "a CPU write next to a GPU write was not seen");
    void* awaitedArea = mmap(nullptr, 2 * blockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(awaitedArea != MAP_FAILED, "cannot map the awaited block");
    const auto awaitedBlock = (reinterpret_cast<std::uintptr_t>(awaitedArea) + blockBytes - 1) & ~(blockBytes - 1);
    const auto awaited = awaitedBlock + 5 * 4096;
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(reinterpret_cast<void*>(awaitedBlock), blockBytes);
    static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(awaitedBlock, 4));
    const auto closed = AgcDriver::GuestMemory::TrackerGeneration();
    *reinterpret_cast<volatile std::uint32_t*>(awaited) = 7;
    Require(AgcDriver::GuestMemory::MarkWritten(awaitedBlock, 4) != 0, "a GPU write into the awaited block was not stamped");
    Require(!AgcDriver::GuestMemory::UnchangedSinceCollected(awaited, 4, closed), "a CPU store to a never-collected page that a GPU write elsewhere in its block collected was not seen");
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(reinterpret_cast<void*>(awaitedBlock), blockBytes);
    munmap(awaitedArea, 2 * blockBytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(reinterpret_cast<void*>(label), blockBytes);
    munmap(labels, 2 * blockBytes);
#endif
}

}

void gpuMappingTests(const Context& context) {
    constexpr std::size_t bytes = 65536;
    auto importing = context;
    importing.hostImportAlignment = bytes;
    auto* cpu = static_cast<std::byte*>(::operator new(bytes, std::align_val_t{bytes}));
    auto* gpu = static_cast<std::byte*>(::operator new(bytes, std::align_val_t{bytes}));
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(cpu, bytes, true, true, false);
        mutation.Add(gpu, bytes, true, true, true);
    }
    {
        GuestBufferMemory leased(importing);
        leased.AcquireRegistered();
        leased.Upload();
        const auto ranges = leased.AddressRanges();
        const auto has = [&](const std::byte* block) { return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin == reinterpret_cast<std::uintptr_t>(block); }); };
        Require(has(gpu), "a GPU-mapped range is missing from the BDA table");
        Require(!has(cpu), "a range mapped without GPU access is in the BDA table");
        leased.Release();
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(cpu);
        mutation.Remove(gpu);
    }
    ClearHostImports(importing.device);
    ::operator delete(cpu, std::align_val_t{bytes});
    ::operator delete(gpu, std::align_val_t{bytes});
}

std::byte* reserveSparse(std::size_t reserved, std::size_t committed) {
#ifdef _WIN32
    auto* block = static_cast<std::byte*>(VirtualAlloc(nullptr, reserved, MEM_RESERVE, PAGE_NOACCESS));
    Require(block != nullptr && VirtualAlloc(block, committed, MEM_COMMIT, PAGE_READWRITE) != nullptr, "cannot reserve the sparse heap");
#else
    void* mapped = mmap(nullptr, reserved, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(mapped != MAP_FAILED && mprotect(mapped, committed, PROT_READ | PROT_WRITE) == 0, "cannot reserve the sparse heap");
    auto* block = static_cast<std::byte*>(mapped);
#endif
    return block;
}

void hostWriteWatchExclusionTests() {
#ifdef _WIN32
    std::vector<std::uint8_t> host(64, 0x31);
    const auto address = reinterpret_cast<std::uintptr_t>(host.data());
    Require(!AgcDriver::GuestMemory::Watched(address, host.size()), "ordinary host heap memory was reported as write-watched");
    Require(AgcDriver::GuestMemory::CollectWritesUncached(address, host.size()) == 0, "ordinary host heap memory returned a guest write generation");
    Require(!AgcDriver::GuestMemory::UnchangedSince(address, host.size(), 1), "ordinary host heap memory was proven unchanged");
    const std::array queries{AgcDriver::GuestMemory::UnchangedQuery{address, host.size(), 1}};
    Require(!AgcDriver::GuestMemory::UnchangedSinceAll(queries), "a host heap range passed a bulk unchanged proof");
    Require(std::all_of(host.begin(), host.end(), [](std::uint8_t value) { return value == 0x31; }), "write-watch eligibility changed host bytes");
#endif
}

static void pageTableRootTests(const Context& shared, const BdaTestAccess& access) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    namespace Ref = BdaPageReference;
    if (!GuestArena::GuestArenaAvailable_nid_postfix() || !GuestArena::GuestArenaWriteWatched_nid_postfix()) {
        std::cout << "guest arena unavailable or not write-watched: the page table of a split heap not tested\n";
        return;
    }
    auto context = shared;
    context.bufferPool.reset();
    context.hostImportAlignment = 65536;
    constexpr std::size_t bytes = 2 * 65536;
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536);
#ifdef _WIN32
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#endif
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(block);
    const auto registry = [&](bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, bytes, true, true, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    const auto sweep = [&] {
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        leased.Upload();
        leased.Release();
    };
    registry(true);
    {
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        leased.AddInPlace(address + 16, 32, true);
        leased.Upload();
        BdaResources bda(context, leased);
        bda.PrepareUse(context, VK_NULL_HANDLE, nullptr);
        const auto table = bda.Table();
        const auto root = access.bytes(table.buffer).subspan(static_cast<std::size_t>(table.offset), static_cast<std::size_t>(table.range));
        Abi::Header header{};
        std::memcpy(&header, root.data(), sizeof(header));
        Require(header.version == Abi::TableVersion && header.grants == 1u && header.pageTable != 0u, "an address-based build over a registered heap bound no page table or no write grant");
        std::vector<Abi::Range> list(header.count);
        std::memcpy(list.data(), root.data() + sizeof(header), list.size() * sizeof(Abi::Range));
        std::vector<Abi::Grant> grants(header.grants);
        std::memcpy(grants.data(), root.data() + sizeof(header) + list.size() * sizeof(Abi::Range), grants.size() * sizeof(Abi::Grant));
        Require(grants.size() == 1u && grants[0].begin == address + 16u && grants[0].end == address + 48u, "the write interval is not the build's write grant");
        const auto first = access.addressBytes(header.pageTable);
        Ref::Leaves leaves;
        for (std::uint32_t window = 0; window < Abi::WindowCount; ++window) {
            std::uint64_t leaf = 0;
            std::memcpy(&leaf, first.data() + window * sizeof(leaf), sizeof(leaf));
            if (leaf == 0u) continue;
            const auto entries = access.addressBytes(leaf);
            std::vector<Abi::PageEntry> decoded(entries.size() / sizeof(Abi::PageEntry));
            std::memcpy(decoded.data(), entries.data(), decoded.size() * sizeof(Abi::PageEntry));
            leaves.emplace(window, std::move(decoded));
        }
        const auto mapped = leased.AddressRanges();
        const auto own = leased.OwnRanges();
        Require(std::equal(list.begin(), list.end(), own.begin(), own.end(), [](const Abi::Range& left, const Abi::Range& right) { return std::memcmp(&left, &right, sizeof(left)) == 0; }), "the table root does not hold the build's own ranges");
        Require(std::none_of(list.begin(), list.end(), [&](const auto& range) { return range.begin < address + bytes && address < range.end; }), "the heap the page table maps is also an own range of the build");
        const auto heap = std::find_if(mapped.begin(), mapped.end(), [&](const auto& range) { return range.begin <= address && address < range.end; });
        Require(heap != mapped.end() && heap->end >= address + bytes && heap->permissions == (Abi::Read | Abi::Write), "the imported writable heap is not one writable range");
        const auto device = heap->deviceAddress + (address - heap->begin);
        Require(Ref::Translate(leaves, list, address + 8u, 64u, Abi::Read, grants) == Ref::Translation{device + 8u, 0u}, "a 64-byte read across the write grant of the heap faulted");
        for (const std::uint64_t offset : {0u, 15u, 16u, 44u, 45u, 48u, 65532u, 65536u, static_cast<unsigned>(bytes) - 4u}) {
            Require(Ref::Translate(leaves, {}, address + offset, 4u, Abi::Read, grants) == Ref::Translation{device + offset, 0u}, "the page table alone did not translate the registered heap at +" + std::to_string(offset));
            Require(Ref::Translate(leaves, {}, address + offset, 4u, Abi::Write) == Ref::Translation{device + offset, 0u}, "the page table alone kept a store from the writable heap at +" + std::to_string(offset));
            Require(Ref::Translate(leaves, {}, address + offset, 4u, Abi::Write, grants) == Ref::Translation{device + offset, 0u}, "a write grant kept a store from the writable heap at +" + std::to_string(offset));
        }
        const Ref::Leaves none;
        for (const auto probe : Ref::Probes(mapped)) {
            for (const auto size : Ref::ProbeBytes) {
                for (const auto permission : {Abi::Read, Abi::Write, Abi::Read | Abi::Write}) {
                    const auto actual = Ref::Translate(leaves, list, probe, size, permission, grants);
                    const auto expected = Ref::Translate(none, mapped, probe, size, permission, grants);
                    Require(actual == expected, "the page table of a granted heap translated 0x" + std::to_string(probe) + " (" + std::to_string(size) + " bytes, permission " + std::to_string(permission) + ") unlike its ranges");
                }
            }
        }
        leased.Release();
    }
    registry(false);
    sweep();
}

void dissolvedSpaceTableTests(const Context& context) {
    constexpr std::size_t piece = 65536;
    void* block = ::operator new(4 * piece, std::align_val_t{piece});
    const auto first = reinterpret_cast<std::uintptr_t>(block);
    std::memset(block, 0x5a, 4 * piece);
    auto importing = context;
    importing.hostImportAlignment = piece;
    auto* guest = static_cast<std::uint8_t*>(block);
    const auto change = [&](std::uint8_t* range, bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, piece, true, true, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    change(guest, true);
    change(guest + 2 * piece, true);
    {
        GuestBufferMemory leased(importing);
        leased.AcquireRegistered();
        leased.Upload();
        const auto pages = leased.PageTable(importing);
        if (AddressSpaceCounters().enabled) Require(pages != nullptr && leased.OwnRanges().empty(), "a build served by its address space alone has own ranges or no page table");
        leased.Release();
    }
    {
        GuestBufferMemory leased(importing);
        leased.AcquireRegistered();
        leased.AddInPlace(first + piece - 16, 32, false);
        reject([&] { leased.Upload(); }, "has no host import");
    }
    change(guest, false);
    change(guest + 2 * piece, false);
    ::operator delete(block, std::align_val_t{piece});
}

static void rangeListTests(const Context& shared) {
    constexpr std::size_t bytes = 65536;
    auto context = shared;
    context.bufferPool.reset();
    context.hostImportAlignment = bytes;
    void* block = ::operator new(bytes, std::align_val_t{bytes});
    std::memset(block, 0, bytes);
    const auto base = reinterpret_cast<std::uintptr_t>(block);
    const auto registry = [&](bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, bytes, true, true, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    registry(true);
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&] {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        return seed;
    };
    for (std::uint32_t round = 0; round < 16u; ++round) {
        GuestBufferMemory memory(context);
        for (std::uint32_t descriptor = 0; descriptor < 6u; ++descriptor) {
            const auto offset = (next() % 3000u) & ~std::uint64_t{3};
            const auto size = 4u + (next() % 1000u);
            if (next() % 2u == 0u) memory.AddInPlace(base + offset, size, true);
            else memory.AddInPlace(base + offset, size, false);
        }
        memory.Upload();
        const auto ranges = memory.AddressRanges();
        ValidateBdaRangeList(ranges);
        BdaResources bda(context, memory);
        memory.Release();
    }
    registry(false);
    ClearHostImports(context.device);
    ::operator delete(block, std::align_val_t{bytes});
}

static std::span<std::byte> descriptorBytes(const BdaTestAccess& access, std::uint32_t binding, std::size_t since) {
    const auto info = access.descriptor(binding, since);
    return access.bytes(info.buffer).subspan(static_cast<std::size_t>(info.offset), static_cast<std::size_t>(info.range));
}

void RunBdaResourceTests(const Context& context, const BdaTestAccess& access) {
    constexpr std::size_t blockBytes = 65536;
    void* raw = ::operator new(blockBytes, std::align_val_t{blockBytes});
    auto* guest = static_cast<std::uint32_t*>(raw);
    std::memset(raw, 0, blockBytes);
    guest[0] = 123;
    const auto address = reinterpret_cast<std::uintptr_t>(raw);
    auto importing = context;
    importing.hostImportAlignment = blockBytes;
    const auto registry = [&](bool add) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, raw, blockBytes, true, true, true);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, raw);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    registry(true);
    {
        GuestBufferMemory memory(importing);
        memory.AddInPlace(address, 64, true);
        memory.AddInPlace(address + 16, 16, true);
        memory.Upload();
        const auto* import = HostImportFor(importing, address, 64);
        Require(import != nullptr, "the registered range was not imported");
        std::uint32_t adjustment = 0;
        const auto first = memory.Address(address, 64, adjustment);
        Require(adjustment == 0 && first.first == import->address + (address - import->base), "a view at its import's start is not addressed in place");
        const auto alias = memory.Address(address + 16, 16, adjustment);
        Require(alias.first + adjustment == first.first + 16 && alias.second == 16 + adjustment, "views of one import have different owners");
        const auto ranges = memory.AddressRanges();
        Require(ranges.size() == 1 && ranges[0].begin == address && ranges[0].deviceAddress == first.first, "incorrect in-place BDA range");
        Require(ranges[0].permissions == (ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write), "an in-place written range is not writable in the BDA table");
        const auto grants = memory.WriteGrants();
        Require(grants.size() == 1 && grants[0].begin == address && grants[0].end == address + 64, "the written in-place views are not one write grant");
        Require(memory.OwnRanges().empty(), "an in-place build has ranges of its own");
        memory.Release();
        reject([&] { memory.Release(); }, "cannot be released twice");
        reject([&] { memory.AddInPlace(address, 64, true); }, "frozen");
    }
    {
        GuestBufferMemory unserved(context);
        unserved.AddInPlace(address, 64, false);
        reject([&] { unserved.Upload(); }, "has no host import");
    }
    {
        auto aligned = importing;
        aligned.limits.minStorageBufferOffsetAlignment = 16;
        GuestBufferMemory unaligned(aligned);
        unaligned.AddInPlace(address, 64, true);
        unaligned.Upload();
        std::uint32_t adjustment = 0;
        const auto owner = unaligned.Address(address, 4, adjustment).first;
        const auto view = unaligned.Address(address + 4, 4, adjustment);
        Require(adjustment == 4 && view.first == owner && view.second == 8, "a view off the address alignment is addressed from below it");
        const auto odd = unaligned.Address(address + 2, 6, adjustment);
        Require(adjustment == 2 && odd.first == owner && odd.second == 8, "a view off a DWORD boundary does not address the DWORDs around it");
        const auto late = unaligned.Address(address + 0x13, 8, adjustment);
        Require(adjustment == 3 && late.first == owner + 16 && late.second == 12, "a view off a DWORD boundary is not addressed from the address alignment below it");
        unaligned.Release();
    }
    GuestBufferMemory overflow(context);
    reject([&] { overflow.AddInPlace(std::numeric_limits<std::uint64_t>::max() - 3, 8, false); }, "overflow");

    ShaderRecompiler::RecompileResult shader;
    shader.bindings = {binding(Role::BdaPagetable, 4), binding(Role::FaultBuffer, 5)};
    CompiledShader compiled{ShaderRecompiler::ShaderStage::Compute, &shader, 0};
    reject([&] { ShaderResources resources(importing, compiled); }, "ABI version");
    shader.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
    auto disabled = importing;
    disabled.bufferDeviceAddress = false;
    reject([&] { ShaderResources resources(disabled, compiled); }, "not enabled");
    {
        ShaderRecompiler::RecompileResult control;
        control.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
        control.bindings = {binding(Role::FaultBuffer, 5)};
        const std::array<CompiledShader, 1> stages{{{ShaderRecompiler::ShaderStage::TessellationControl, &control, 0}}};
        const auto since = access.descriptorWrites();
        ShaderResources resources(disabled, stages, ColorTarget{}, 0, 0, 0);
        Require(resources.ChecksFaults(), "a fault-only build does not check faults");
        Require(resources.NeedsCompletion() && !resources.UsesAddressTable(), "a fault-only build reaches guest memory through the BDA table");
        resources.CheckFaults();
        const auto fault = descriptorBytes(access, 5, since);
        std::memset(fault.data(), 0xa5, fault.size());
        resources.PrepareDeviceRecords(nullptr, VK_NULL_HANDLE);
        for (const auto byte : fault.first(ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t))) Require(byte == std::byte{}, "rect-list fault buffer was not cleared before its first use");
        const ShaderRecompiler::BdaAbi::Fault report{ShaderRecompiler::BdaAbi::FaultState::Ready, ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle, 0, 0, 0, 0, 0};
        std::memcpy(fault.data(), &report, sizeof(report));
        resources.CheckFaults();
        resources.FinishDeviceRecords(VK_NULL_HANDLE);
        reject([&] { resources.CheckFaults(); }, "rect-list requires");
    }
    {
        shader.bdaWrites = true;
        ShaderResources resources(importing, compiled);
        const auto writable = resources.Writable();
        Require(writable.writableImports && writable.ranges.empty(), "a build of a program that stores through the page table may not write the writable imports");
        resources.WriteBack();
        shader.bdaWrites = false;
    }
    {
        const auto since = access.descriptorWrites();
        ShaderResources resources(importing, compiled);
        Require(resources.UsesAddressTable(), "an address-based build does not use the BDA table");
        Require(!resources.Writable().writableImports && resources.Writable().ranges.empty(), "a build of a program that stores nothing may write");
        const auto table = descriptorBytes(access, 4, since);
        ShaderRecompiler::BdaAbi::Header header{};
        std::memcpy(&header, table.data(), sizeof(header));
        Require(header.version == ShaderRecompiler::BdaAbi::TableVersion && header.count == 0 && header.grants == 0 && header.reserved == 0 && header.touchTable == 0, "BDA header layout mismatch: version " + std::to_string(header.version) + ", " + std::to_string(header.count) + " own ranges, " + std::to_string(header.grants) + " write grants");
        Require(header.pageTable != 0, "an address-based build over an imported range has no page table");
        const auto fault = descriptorBytes(access, 5, since);
        std::memset(fault.data(), 0xa5, fault.size());
        resources.PrepareDeviceRecords(nullptr, VK_NULL_HANDLE);
        for (const auto byte : fault.first(ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t))) Require(byte == std::byte{}, "fault buffer was not cleared before its first use");
        const ShaderRecompiler::BdaAbi::Fault unmapped{ShaderRecompiler::BdaAbi::FaultState::Ready, ShaderRecompiler::BdaAbi::FaultReason::Unmapped, 0x7fff99880000ULL, 4, 0, 0x44, 0};
        std::memcpy(fault.data(), &unmapped, sizeof(unmapped));
        resources.FinishDeviceRecords(VK_NULL_HANDLE);
        reject([&] { resources.WriteBack(); }, "BDA access failed");
        resources.PrepareDeviceRecords(nullptr, VK_NULL_HANDLE);
        resources.FinishDeviceRecords(VK_NULL_HANDLE);
        resources.WriteBack();
    }
    {
        void* plain = GuestHeap::GuestHeapAllocate_nid_postfix(64);
        void* gpu = GuestHeap::GuestHeapAlignForGpu_nid_postfix(16, 64);
        const auto lookup = [](const void* pointer) {
            GuestAllocations::Mutation mutation;
            return mutation.Find(pointer);
        };
        Require(!lookup(plain).gpu && lookup(gpu).gpu, "a guest heap block's GPU access does not follow how it was allocated");
        gpu = GuestHeap::GuestHeapReallocate_nid_postfix(gpu, 128);
        plain = GuestHeap::GuestHeapRealign_nid_postfix(plain, 128, 64);
        Require(!lookup(plain).gpu && lookup(gpu).gpu, "a reallocated guest heap block changed its GPU access");
        GuestHeap::GuestHeapFree_nid_postfix(plain);
        GuestHeap::GuestHeapFree_nid_postfix(gpu);
    }
    {
        const auto registered = [&] {
            const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
            return std::any_of(lease.begin(), lease.end(), [&](const auto& range) { return range->address == address; });
        };
        Require(registered(), "the guest block is not registered");
        const auto before = AddressSpaceCounters();
        for (int build = 0; build < 2; ++build) {
            GuestBufferMemory leased(importing);
            leased.AcquireRegistered();
            Require(leased.HoldsLease(), "address-based build holds no lease");
            leased.Upload();
            std::uint32_t adjustment = 0;
            const auto view = leased.Address(address, 64, adjustment);
            Require(view.second == 64 + adjustment, "the leased block has no device address");
            const auto ranges = leased.AddressRanges();
            Require(std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin == address && range.end == address + blockBytes; }), "the leased block is missing from the BDA table");
            leased.Release();
            Require(!leased.HoldsLease(), "the release kept the lease");
        }
        const auto after = AddressSpaceCounters();
        if (after.enabled) Require(after.hits == before.hits + 1 && after.rebuiltFirst + after.rebuiltGeneration + after.rebuiltWaiterDrop + after.rebuiltEpoch + after.rebuiltDevice == before.rebuiltFirst + before.rebuiltGeneration + before.rebuiltWaiterDrop + before.rebuiltEpoch + before.rebuiltDevice + 1, "the second build did not take the cached address space");
        registry(false);
        Require(!registered(), "the removed guest block remains registered");
        const auto dropped = AddressSpaceCounters();
        if (dropped.enabled) Require(dropped.waiterDrops == after.waiterDrops + 1 && LeaseCounters().cacheDrops == dropped.waiterDrops, "the removal did not drop the cached address space");
    }
    ::operator delete(raw, std::align_val_t{blockBytes});
    importCrossingTests(context, access);
    importSpanTests(context);
    failedImportRemapTests(context, access);
    spaceImportRunTests(context);
    importWindowTests(context);
    importJoinTests(context);
    importWholeRangeOverWindowTests(context);
    importRefusedWholeRangeTests(context);
    importNeighbouringEdgesTests(context);
    importAtMapOverBudgetTests(context);
    importEvictionTests(context);
    pageTableRootTests(context, access);
    dissolvedSpaceTableTests(context);
    rangeListTests(context);
    importWatchTests();
    importedFreshTests(context);
    gpuMappingTests(context);
    hostWriteWatchExclusionTests();
}
