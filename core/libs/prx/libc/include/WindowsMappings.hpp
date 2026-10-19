#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP

#ifdef _WIN32
#include "prx/libc/include/GuestArena.hpp"
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>

namespace GuestArena {

class WindowsMappings {
public:
    enum class Collection { All, Armed, Fresh };

    static WindowsMappings& Get() {
        static auto* mappings = new WindowsMappings;
        return *mappings;
    }

    static std::uint64_t MemoryQueries() {
        return queries.load(std::memory_order_relaxed);
    }

    void* Reserve(void* address, std::size_t bytes) {
        return allocate(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    }

    using Guarded = std::vector<std::uintptr_t>;

    std::vector<std::pair<std::uintptr_t, std::size_t>> Commit(void* address, std::size_t bytes, DWORD protection, std::size_t granule, bool watched) {
        std::vector<std::pair<std::uintptr_t, std::size_t>> created;
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(address); cursor < end;) {
            const auto memory = query(cursor);
            const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
            if (memory.State == MEM_RESERVE) {
                const auto limit = std::min(end, cursor + granule);
                auto placeholderEnd = stop;
                while (placeholderEnd < limit) {
                    const auto next = query(placeholderEnd);
                    if (next.State != MEM_RESERVE) break;
                    placeholderEnd = reinterpret_cast<std::uintptr_t>(next.BaseAddress) + next.RegionSize;
                }
                const auto size = std::min(limit, placeholderEnd) - cursor;
                reset(cursor, size);
                const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watched ? MEM_WRITE_WATCH : 0);
                if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(cursor), size, flags, protection, nullptr, 0)) fail("replace guest placeholder with private memory");
                privateAllocations.emplace(cursor, cursor + size);
                if (!created.empty() && created.back().first + created.back().second == cursor) created.back().second += size;
                else created.emplace_back(cursor, size);
                cursor += size;
            } else {
                if (memory.State != MEM_COMMIT) throw std::runtime_error("guest memory is not committed");
                const auto mapped = views.find(cursor & ~(pageBytes - 1));
                if (mapped != views.end()) {
                    mapped->second.protection = protection;
                    mapped->second.armed = 0;
                    invalidate(*mapped->second.page);
                    dropGuard(*mapped->second.page);
                }
                reprotectPrivate(cursor, stop, protection);
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(cursor), stop - cursor, protection, &previous)) fail("protect guest memory");
                cursor = stop;
            }
        }
        return created;
    }

    void Reset(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reset(reinterpret_cast<std::uintptr_t>(address), bytes);
    }

    void Map(void* address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection, bool watched = false, bool resident = false) {
        std::unique_lock lock(mutex);
        collectableRanges.clear();
        auto cursor = reinterpret_cast<std::uintptr_t>(address);
        reset(cursor, bytes);
        if (resident) watchResident = watched;
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) fail("keep shared guest section");
        const auto owned = std::make_shared<Section>(duplicate);
        const auto slotOf = [&](std::size_t done) -> const std::shared_ptr<Physical>& {
            auto& slot = physical[std::make_pair(reinterpret_cast<std::uintptr_t>(section), offset + done)];
            if (!slot) slot = std::make_shared<Physical>();
            return slot;
        };
        const auto pageOf = [&](Physical& slot, std::size_t done) {
            auto shared = slot.page.lock();
            if (!shared) {
                shared = std::make_shared<SharedPage>();
                shared->section = reinterpret_cast<std::uintptr_t>(section);
                shared->offset = offset + done;
                slot.page = shared;
            }
            return shared;
        };
        const auto single = [&](std::size_t done) {
            const auto& slot = *slotOf(done);
            const auto shared = slot.page.lock();
            return resident && !sharedImports && !slot.stored && (!shared || shared->aliases.empty());
        };
        const bool perPage = !resident || sharedImports;
        if (perPage) {
            lock.unlock();
            commitSection(section, offset, bytes);
            lock.lock();
        }
        std::uintptr_t rest = 0;
        std::uintptr_t restEnd = 0;
        for (std::size_t done = 0; done < bytes;) {
            const auto base = cursor + done;
            yieldToFaults(lock);
            if (single(done)) {
                rest = 0;
                auto runBytes = pageBytes;
                while (done + runBytes < bytes && (base + runBytes) % chunkBytes != 0 && single(done + runBytes)) runBytes += pageBytes;
                split(base, runBytes);
                commitResident(base, runBytes);
                residentRuns.emplace(base, Run{base + runBytes, {}, static_cast<std::uint32_t>(runBytes >> 12)});
                for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                    const auto slot = slotOf(done + at);
                    const auto shared = pageOf(*slot, done + at);
                    if (slot->saved) {
                        std::memcpy(reinterpret_cast<void*>(base + at), slot->saved->bytes + slot->savedOffset, pageBytes);
                        slot->saved.reset();
                    }
                    shared->aliases.push_back(base + at);
                    views.emplace(base + at, View{shared, protection, {}, 0, owned, offset + done + at, 0, 0, true, slot});
                    invalidate(*shared);
                }
                remember(freshRanges, base, base + runBytes);
                DWORD previous;
                if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect resident guest pages");
                done += runBytes;
                continue;
            }
            const auto slot = slotOf(done);
            const auto shared = pageOf(*slot, done);
            const auto aliases = shared->aliases;
            for (const auto alias : aliases) {
                if (!views.at(alias).resident) continue;
                demote(alias);
                rest = 0;
            }
            lock.unlock();
            if (base == rest && base + pageBytes <= restEnd) {
                if (base + pageBytes != restEnd && !VirtualFree(reinterpret_cast<void*>(base), pageBytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
            } else {
                restEnd = split(base, pageBytes);
            }
            rest = base + pageBytes;
            mapShared(base, section, offset + done, protection, perPage);
            lock.lock();
            collectableRanges.clear();
            slot->stored = true;
            shared->aliases.push_back(base);
            views.emplace(base, View{shared, protection, {}, 0, owned, offset + done, 0, 0, false, slot});
            remember(freshRanges, base, base + pageBytes);
            invalidate(*shared);
            done += pageBytes;
        }
    }

    void SetProtection(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        for (auto it = views.lower_bound(address); it != views.end() && it->first < address + bytes; ++it) {
            it->second.protection = protection;
            it->second.armed = 0;
            invalidate(*it->second.page);
            dropGuard(*it->second.page);
        }
        reprotectPrivate(address, address + bytes, protection);
    }

    GuardRefusal Guard(std::uintptr_t address, std::size_t bytes) {
        if (address % guardBytes != 0 || bytes % guardBytes != 0) throw std::invalid_argument("guest write guards cover whole 64 KiB pages");
        std::lock_guard lock(mutex);
        if (unsettledImport(address, address + bytes)) return GuardRefusal::UnsettledImport;
        std::vector<std::shared_ptr<SharedPage>> pages;
        std::vector<std::uintptr_t> privates;
        for (auto piece = address; piece < address + bytes; piece += pageBytes) {
            const auto view = views.find(piece);
            const auto refusal = view == views.end() ? privateRefusal(piece) : sharedRefusal(view->second);
            if (refusal != GuardRefusal::None) {
                prunePrivate(privates);
                return refusal;
            }
            if (view == views.end()) privates.push_back(piece);
            else pages.push_back(view->second.page);
        }
        for (const auto piece : privates) {
            auto& page = privatePieces.at(piece);
            page.guarded = true;
            const auto arming = static_cast<std::uint8_t>(allSubpages & ~page.armed & ~page.accessGuarded);
            if (!writable(page.protection) || arming == 0) continue;
            protectSubpages(piece, arming, readOnly(page.protection), "arm a private guest write guard");
            page.armed |= arming;
        }
        for (const auto& page : pages) {
            page->guarded = true;
            for (const auto alias : page->aliases) {
                auto& view = views.at(alias);
                const auto arming = static_cast<std::uint8_t>(allSubpages & ~view.armed & ~page->accessGuarded);
                if (!writable(view.protection) || arming == 0) continue;
                protectSubpages(alias, arming, view.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY, "arm a guest write guard");
                view.armed |= arming;
            }
        }
        return GuardRefusal::None;
    }

    void Unguard(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) it->second.page->guarded = false;
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < address + bytes;) {
            it->second.guarded = false;
            it = pruned(it);
        }
    }

    void NoteImport(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end; ++it) {
            if (it->second.accessGuarded != 0) throw std::runtime_error("host import of private guest memory over access guarded pages: their GPU stores were not written back first");
        }
        const auto stamp = ++importStamp;
        for (auto page = address / guardBytes; page <= (end - 1) / guardBytes; ++page) unsettledImports.insert_or_assign(page, stamp);
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end;) {
            auto& page = it->second;
            dropPrivateGuard(it->first);
            if (page.armed != 0) protectSubpages(it->first, page.armed, page.protection, "open private guest memory to a host import");
            page.armed = 0;
            it = pruned(it);
        }
    }

    std::uint64_t ImportStamp() {
        std::lock_guard lock(mutex);
        return importStamp;
    }

    void SettleImports(std::uintptr_t address, std::size_t bytes, std::uint64_t stamp) {
        std::lock_guard lock(mutex);
        refusalClears.push_back({address, bytes, RefusalBit(GuardRefusal::UnsettledImport)});
        for (auto page = address / guardBytes; page <= (address + bytes - 1) / guardBytes; ++page) {
            if (const auto found = unsettledImports.find(page); found != unsettledImports.end() && found->second <= stamp) unsettledImports.erase(found);
        }
    }

    bool ImportSettled(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        return !unsettledImport(address, address + bytes);
    }

    bool IsGuarded(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        const auto base = address & ~(pageBytes - 1);
        const auto view = views.find(base);
        if (view != views.end()) return view->second.page->guarded;
        const auto piece = privatePieces.find(base);
        return piece != privatePieces.end() && piece->second.guarded;
    }

    bool IsView(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        return views.find(address & ~(pageBytes - 1)) != views.end();
    }

    GuardRefusal GuardAccess(std::uintptr_t address, std::size_t bytes) {
        if (address % subpageBytes != 0 || bytes % subpageBytes != 0) throw std::invalid_argument("guest access guards cover whole 4 KiB pages");
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        if (unsettledImport(address, end)) return GuardRefusal::UnsettledImport;
        std::vector<std::pair<std::shared_ptr<SharedPage>, std::uint8_t>> pages;
        std::vector<std::uintptr_t> privates;
        for (auto piece = address & ~(pageBytes - 1); piece < end; piece += pageBytes) {
            const auto view = views.find(piece);
            const auto refusal = view == views.end() ? privateRefusal(piece) : sharedRefusal(view->second);
            if (refusal != GuardRefusal::None) {
                prunePrivate(privates);
                return refusal;
            }
            if (view == views.end()) privates.push_back(piece);
            else pages.emplace_back(view->second.page, subpagesOf(piece, address, end));
        }
        for (const auto piece : privates) {
            auto& page = privatePieces.at(piece);
            const auto adding = static_cast<std::uint8_t>(subpagesOf(piece, address, end) & ~page.accessGuarded);
            if (adding == 0) continue;
            page.accessGuarded |= adding;
            protectSubpages(piece, adding, PAGE_NOACCESS, "arm a private guest access guard");
            page.armed &= static_cast<std::uint8_t>(~adding);
        }
        for (const auto& [page, mask] : pages) {
            const auto adding = static_cast<std::uint8_t>(mask & ~page->accessGuarded);
            if (adding == 0) continue;
            page->accessGuarded |= adding;
            for (const auto alias : page->aliases) {
                auto& view = views.at(alias);
                protectSubpages(alias, adding, PAGE_NOACCESS, "arm a guest access guard");
                view.armedUnderAccess |= static_cast<std::uint8_t>(view.armed & adding);
                view.armed &= static_cast<std::uint8_t>(~adding);
            }
        }
        return GuardRefusal::None;
    }

    bool AccessGuarded(std::uintptr_t address) {
        const auto lock = faultLock();
        const auto base = address & ~(pageBytes - 1);
        const auto view = views.find(base);
        if (view != views.end()) return (view->second.page->accessGuarded & subpagesOf(base, address, address + 1)) != 0;
        const auto piece = privatePieces.find(base);
        return piece != privatePieces.end() && (piece->second.accessGuarded & subpagesOf(base, address, address + 1)) != 0;
    }

    void UnguardAccess(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            auto& page = *it->second.page;
            const auto dropping = static_cast<std::uint8_t>(page.accessGuarded & subpagesOf(it->first, address, end));
            if (dropping == 0) continue;
            page.accessGuarded &= static_cast<std::uint8_t>(~dropping);
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                const auto rearming = static_cast<std::uint8_t>(writable(view.protection) ? (page.guarded ? dropping : dropping & view.armedUnderAccess) : 0);
                view.armedUnderAccess &= static_cast<std::uint8_t>(~dropping);
                if (rearming != 0) {
                    protectSubpages(alias, rearming, view.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY, "drop a guest access guard behind a write guard");
                    view.armed |= rearming;
                }
                if (const auto open = static_cast<std::uint8_t>(dropping & ~rearming); open != 0) protectSubpages(alias, open, view.protection, "drop a guest access guard");
            }
        }
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end;) {
            auto& page = it->second;
            const auto dropping = static_cast<std::uint8_t>(page.accessGuarded & subpagesOf(it->first, address, end));
            if (dropping != 0) {
                page.accessGuarded &= static_cast<std::uint8_t>(~dropping);
                if (page.guarded && writable(page.protection)) {
                    protectSubpages(it->first, dropping, readOnly(page.protection), "drop a private guest access guard behind a write guard");
                    page.armed |= dropping;
                } else {
                    protectSubpages(it->first, dropping, page.protection, "drop a private guest access guard");
                }
            }
            it = pruned(it);
        }
    }

    Guarded TakeGuardReports() {
        const auto lock = faultLock();
        Guarded reports;
        reports.swap(guardReports);
        return reports;
    }

    struct Demoted {
        std::uintptr_t begin;
        std::uintptr_t end;
        DWORD protection;
    };

    std::vector<Demoted> TakeDemoted() {
        std::lock_guard lock(mutex);
        std::vector<Demoted> runs;
        runs.swap(demotedRuns);
        return runs;
    }

    struct RefusalClear {
        std::uintptr_t address;
        std::size_t bytes;
        std::uint32_t cleared;
    };

    std::vector<RefusalClear> TakeRefusalClears() {
        std::lock_guard lock(mutex);
        std::vector<RefusalClear> clears;
        clears.swap(refusalClears);
        return clears;
    }

    std::vector<std::pair<std::uintptr_t, std::size_t>> Aliases(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        std::vector<std::pair<std::uintptr_t, std::size_t>> ranges;
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            const auto from = std::max(address, it->first) - it->first;
            const auto to = std::min<std::uintptr_t>(end, it->first + pageBytes) - it->first;
            for (const auto alias : it->second.page->aliases) {
                if (alias != it->first) ranges.emplace_back(alias + from, to - from);
            }
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<std::uintptr_t, std::size_t>> merged;
        for (const auto& range : ranges) {
            if (!merged.empty() && merged.back().first + merged.back().second >= range.first) merged.back().second = std::max(merged.back().first + merged.back().second, range.first + range.second) - merged.back().first;
            else merged.push_back(range);
        }
        return merged;
    }

    bool Backing(std::uintptr_t page, std::uint64_t* section, std::uint64_t* offset, bool* uniform, bool* aliased) {
        std::lock_guard lock(mutex);
        const auto base = page & ~(guardBytes - 1);
        const auto first = views.find(base);
        *uniform = true;
        *aliased = false;
        if (first == views.end()) {
            *section = 0;
            *offset = base;
            for (auto piece = base; piece < base + guardBytes; piece += pageBytes) {
                if (views.contains(piece)) *uniform = false;
            }
            return false;
        }
        *section = first->second.page->section;
        *offset = first->second.page->offset;
        for (auto piece = base; piece < base + guardBytes; piece += pageBytes) {
            const auto view = views.find(piece);
            if (view == views.end() || view->second.page->section != *section || view->second.page->offset != *offset + (piece - base)) {
                *uniform = false;
                continue;
            }
            if (view->second.page->aliases.size() > 1) *aliased = true;
        }
        return true;
    }

    void Pin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            ++page.pins;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (view.armed == 0) continue;
                protectSubpages(alias, view.armed, view.protection, "pin shared guest page writable");
                view.armed = 0;
            }
            invalidate(page);
        }
        for (auto piece = address & ~(pageBytes - 1); piece < address + bytes; piece += pageBytes) {
            if (views.contains(piece)) continue;
            auto* page = privatePiece(piece);
            if (page == nullptr) continue;
            ++page->pins;
            if (page->armed != 0) protectSubpages(piece, page->armed, page->protection, "pin private guest page writable");
            page->armed = 0;
            dropPrivateGuard(piece);
        }
    }

    void Unpin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        clearRefusals(address & ~(pageBytes - 1), address + bytes, RefusalBit(GuardRefusal::PrivatePinned) | RefusalBit(GuardRefusal::SharedPinned));
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            if (page.pins != 0) --page.pins;
            invalidate(page);
        }
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < address + bytes;) {
            if (it->second.pins != 0) --it->second.pins;
            it = pruned(it);
        }
    }

    bool HandleWrite(std::uintptr_t address) {
        const auto lock = faultLock();
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        if (found == views.end()) return handlePrivateWrite(base, address);
        if (!writable(found->second.protection)) return false;
        auto& view = found->second;
        if (view.resident) {
            const auto memory = query(address);
            return memory.State == MEM_COMMIT && writable(memory.Protect & 0xffu);
        }
        const auto written = subpagesOf(base, address, address + 1);
        invalidate(*view.page, written);
        protectSubpages(base, written, view.protection, "resume shared memory write");
        view.armed &= static_cast<std::uint8_t>(~written);
        dropGuard(*view.page);
        return true;
    }

    bool HandleRead(std::uintptr_t address) {
        const auto lock = faultLock();
        if (!views.contains(address & ~(pageBytes - 1))) return false;
        const auto memory = query(address);
        return memory.State == MEM_COMMIT && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
    }

    bool ResidentSpan(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end) {
        std::lock_guard lock(mutex);
        auto run = residentRuns.upper_bound(address);
        if (run != residentRuns.begin() && std::prev(run)->second.end > address) --run;
        if (run == residentRuns.end() || run->first >= address + bytes) return false;
        *start = run->first;
        for (; run != residentRuns.end() && run->first < address + bytes; ++run) *end = run->second.end;
        return true;
    }

    void ForgetSection(HANDLE section) {
        std::lock_guard lock(mutex);
        const auto key = reinterpret_cast<std::uintptr_t>(section);
        physical.erase(physical.lower_bound(std::make_pair(key, std::uint64_t{0})), physical.lower_bound(std::make_pair(key + 1, std::uint64_t{0})));
    }

    std::uintptr_t FirstAccessGuarded(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        return firstAccessGuarded(address, address + bytes);
    }

    bool BeginHostWrite(std::uintptr_t address, std::size_t bytes) {
        if (!ReserveHostWrite(address, bytes)) return false;
        if (OpenHostWrite(address, bytes)) return true;
        CancelHostWrite(address, bytes);
        return false;
    }

    bool ReserveHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        privateHostWrites.emplace_back(address, end);
        for (auto it = first; it != views.end() && it->first < end; ++it) ++it->second.hostWrites;
        return true;
    }

    void CancelHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        clearRefusals(address & ~(pageBytes - 1), end, RefusalBit(GuardRefusal::PrivateHostWrite) | RefusalBit(GuardRefusal::AliasHostWrite));
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) --it->second.hostWrites;
        const auto open = std::find(privateHostWrites.begin(), privateHostWrites.end(), std::make_pair(address, end));
        if (open != privateHostWrites.end()) privateHostWrites.erase(open);
    }

    bool OpenHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        if (firstAccessGuarded(address, end) != 0) return false;
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end;) {
            auto& page = it->second;
            dropPrivateGuard(it->first);
            const auto opened = static_cast<std::uint8_t>(page.armed & subpagesOf(it->first, address, end));
            if (opened != 0) protectSubpages(it->first, opened, page.protection, "open private memory to a host write");
            page.armed &= static_cast<std::uint8_t>(~opened);
            it = pruned(it);
        }
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            const auto written = subpagesOf(it->first, address, end);
            invalidate(*view.page, written);
            dropGuard(*view.page);
            const auto opened = static_cast<std::uint8_t>(view.armed & written);
            if (opened == 0) continue;
            protectSubpages(it->first, opened, view.protection, "open shared memory to a host write");
            view.armed &= static_cast<std::uint8_t>(~opened);
        }
        return true;
    }

    void EndHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        clearRefusals(address & ~(pageBytes - 1), end, RefusalBit(GuardRefusal::PrivateHostWrite) | RefusalBit(GuardRefusal::AliasHostWrite));
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            --it->second.hostWrites;
            invalidate(*it->second.page, subpagesOf(it->first, address, end));
        }
        const auto open = std::find(privateHostWrites.begin(), privateHostWrites.end(), std::make_pair(address, end));
        if (open != privateHostWrites.end()) privateHostWrites.erase(open);
    }

    void SetSharedImports(bool imported) {
        std::lock_guard lock(mutex);
        sharedImports = imported;
    }

    void* MapAlias(std::uintptr_t address, std::size_t bytes) {
        std::unique_lock lock(mutex);
        const auto refuse = [&](const char* reason) {
            char text[192];
            std::snprintf(text, sizeof(text), "read-write alias of shared guest memory 0x%llx+0x%llx: %s", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), reason);
            return std::runtime_error(text);
        };
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) throw refuse("the range is not made of whole shared pages");
        const auto contiguity = [&] {
            auto page = views.find(address);
            if (page == views.end()) throw refuse("the range does not start at a shared view");
            const auto section = page->second.section;
            const auto offset = page->second.offset;
            bool contiguous = true;
            bool resident = false;
            for (std::size_t done = 0; done < bytes; done += pageBytes, ++page) {
                if (done % faultYieldBytes == 0 && yieldToFaults(lock)) page = views.find(address + done);
                if (page == views.end() || page->first != address + done) throw refuse("the range is not a run of shared views");
                if (page->second.resident) {
                    resident = true;
                    const auto run = std::prev(residentRuns.upper_bound(address + done));
                    if (run->first < address || run->second.end > address + bytes) throw refuse("a resident run of direct memory extends outside the range");
                }
                if (page->second.offset != offset + done || (page->second.section != section && !sameSection(page->second.section->handle, section->handle))) contiguous = false;
            }
            return std::make_pair(contiguous, resident);
        };
        auto [contiguous, resident] = contiguity();
        if (resident) {
            for (std::size_t done = 0; done < bytes; done += pageBytes) {
                const auto found = views.find(address + done);
                if (found != views.end() && found->second.resident) demote(address + done);
            }
            contiguous = contiguity().first;
        }
        const auto view = views.find(address);
        if (view == views.end()) throw refuse("the range does not start at a shared view");
        const auto section = view->second.section;
        const auto offset = view->second.offset;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        if (!contiguous) return mapAliasRuns(address, bytes);
        const auto lead = offset % system.dwAllocationGranularity;
        void* alias = map(section->handle, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (alias == nullptr) {
            char text[160];
            std::snprintf(text, sizeof(text), "MapViewOfFile3 of a read-write alias of shared guest memory 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), text);
        }
        return static_cast<char*>(alias) + lead;
    }

    void UnmapAlias(void* alias) {
        if (alias == nullptr) return;
        {
            std::lock_guard lock(mutex);
            if (const auto found = splitAliases.find(reinterpret_cast<std::uintptr_t>(alias)); found != splitAliases.end()) {
                const auto error = releaseAlias(found->first, found->second);
                splitAliases.erase(found);
                if (error != ERROR_SUCCESS) throw std::system_error(static_cast<int>(error), std::system_category(), "unmap a run of a shared guest alias");
                return;
            }
        }
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const auto base = reinterpret_cast<std::uintptr_t>(alias) & ~(static_cast<std::uintptr_t>(system.dwAllocationGranularity) - 1);
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(base), 0)) fail("unmap shared guest alias");
    }

    bool Protection(std::uintptr_t address, std::uint32_t* protection, std::uintptr_t* end = nullptr) {
        std::lock_guard lock(mutex);
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        if (found != views.end()) {
            *protection = found->second.protection;
            return true;
        }
        const auto piece = privatePieces.lower_bound(base);
        if (piece != privatePieces.end() && piece->first == base) {
            if (end != nullptr) *end = base + pageBytes;
            *protection = piece->second.protection;
            return true;
        }
        if (end != nullptr) *end = piece != privatePieces.end() ? piece->first : ~std::uintptr_t{0};
        return false;
    }

    void CollectWork(std::uint64_t* systemCalls, std::uint64_t* pieces) {
        std::lock_guard lock(mutex);
        *systemCalls = collectSystemCalls;
        *pieces = collectPieces;
    }

    bool Collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear, Collection collection = Collection::All) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        if (collection == Collection::Fresh) {
            const auto fresh = freshRanges.upper_bound(address);
            if ((fresh == freshRanges.begin() || std::prev(fresh)->second <= address) && (fresh == freshRanges.end() || fresh->first >= end)) return true;
            for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
                if (it->second.resident) {
                    if (!watchResident) return false;
                    continue;
                }
                if (it->second.protection != PAGE_NOACCESS) continue;
                for (auto at = std::max(address, it->first); at < std::min(end, it->first + pageBytes); at += 4096) {
                    if ((it->second.collected & (1u << ((at - it->first) / 4096))) == 0) return false;
                }
            }
        } else if (clear && !collectable(address, end)) return false;
        auto position = views.lower_bound(address & ~(pageBytes - 1));
        std::uintptr_t cleanStart = 0;
        std::uintptr_t cleanEnd = 0;
        const auto flushClean = [&] {
            if (cleanEnd > cleanStart) rememberClean(cleanStart, cleanEnd);
            cleanStart = cleanEnd = 0;
        };
        for (auto cursor = address; cursor < end;) {
            const auto nextClean = cleanRanges.upper_bound(cursor);
            if (nextClean != cleanRanges.begin()) {
                const auto clean = std::prev(nextClean);
                if (cursor < clean->second) {
                    cursor = std::min(end, clean->second);
                    continue;
                }
            }
            const auto base = cursor & ~(pageBytes - 1);
            const auto found = viewAt(base, position);
            if (found != views.end() && found->second.resident) {
                flushClean();
                if (!watchResident) return false;
                const auto available = capacity - *count;
                if (available == 0) return true;
                const auto run = std::prev(residentRuns.upper_bound(cursor));
                const auto stop = std::min({end, run->second.end, (cursor & ~std::uintptr_t{4095}) + available * 4096});
                collectResident(run->first, run->second, cursor, stop, pages, count, clear, collection);
                if (*count == capacity) return true;
                cursor = stop;
            } else if (found != views.end()) {
                auto& view = found->second;
                auto stop = std::min(std::min(end, base + pageBytes), nextClean == cleanRanges.end() ? end : nextClean->first);
                if (collection != Collection::All) {
                    const auto selected = [&](std::uintptr_t at) {
                        const bool collected = (view.collected & (1u << ((at - base) / 4096))) != 0;
                        return collection == Collection::Armed ? collected : !collected;
                    };
                    while (cursor < stop && !selected(cursor)) cursor = std::min(stop, cursor + 4096);
                    if (cursor == stop) continue;
                    for (auto at = cursor + 4096; at < stop; at += 4096) {
                        if (!selected(at)) {
                            stop = at;
                            break;
                        }
                    }
                }
                if (view.protection == PAGE_NOACCESS) {
                    flushClean();
                    return false;
                }
                const bool pinned = view.page->pins != 0;
                std::uint8_t settled = 0;
                auto settledEnd = cursor;
                bool full = false;
                for (std::size_t sub = 0; sub < subpages; ++sub) {
                    const auto from = std::max(cursor, base + sub * subpageBytes);
                    const auto to = std::min(stop, base + (sub + 1) * subpageBytes);
                    if (from >= to) continue;
                    if (collection == Collection::Fresh || pinned || view.seen[sub] != view.page->generation[sub]) {
                        if (*count == capacity) {
                            full = true;
                            break;
                        }
                        pages[(*count)++] = reinterpret_cast<void*>(from);
                    }
                    settled |= static_cast<std::uint8_t>(1u << sub);
                    settledEnd = to;
                }
                if (clear) {
                    for (std::size_t sub = 0; sub < subpages; ++sub) {
                        const auto bit = static_cast<std::uint8_t>(1u << sub);
                        if ((settled & bit) == 0 || (view.collected & bit) != 0) continue;
                        view.collected |= bit;
                        forget(freshRanges, base + sub * subpageBytes, base + (sub + 1) * subpageBytes);
                    }
                }
                if (clear && !pinned && settled != 0) {
                    std::uint8_t untracked = 0;
                    for (const auto alias : view.page->aliases) {
                        auto& other = views.at(alias);
                        if (other.resident) continue;
                        const auto arming = static_cast<std::uint8_t>(settled & ~other.armed & ~view.page->accessGuarded);
                        if (!writable(other.protection) || arming == 0) continue;
                        if (other.hostWrites != 0) {
                            untracked |= arming;
                            continue;
                        }
                        const DWORD protection = other.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
                        protectSubpages(alias, arming, protection, "arm shared memory write tracking");
                        other.armed |= arming;
                    }
                    const auto tracked = static_cast<std::uint8_t>(settled & ~untracked);
                    for (std::size_t sub = 0; sub < subpages; ++sub) {
                        if ((tracked & (1u << sub)) == 0) continue;
                        const auto from = std::max(cursor, base + sub * subpageBytes);
                        const auto to = std::min(settledEnd, base + (sub + 1) * subpageBytes);
                        if (from == base + sub * subpageBytes && to == base + (sub + 1) * subpageBytes) view.seen[sub] = view.page->generation[sub];
                        if (from != cleanEnd) {
                            flushClean();
                            cleanStart = from;
                        }
                        cleanEnd = to;
                    }
                }
                if (full) {
                    flushClean();
                    return true;
                }
                cursor = stop;
            } else {
                flushClean();
                if (collection == Collection::Fresh) {
                    const auto next = views.lower_bound(cursor);
                    cursor = next == views.end() ? end : std::min(end, next->first);
                    continue;
                }
                const auto allocationEnd = privateAllocationEnd(cursor);
                if (allocationEnd == 0) return false;
                const auto stop = std::min(end, allocationEnd);
                if (capacity == *count) return true;
                const auto touched = (((stop + 4095) & ~static_cast<std::uintptr_t>(4095)) - (cursor & ~static_cast<std::uintptr_t>(4095))) / 4096;
                ULONG_PTR available = std::min<ULONG_PTR>(capacity - *count, touched);
                DWORD granularity = 0;
                ++collectSystemCalls;
                collectPieces += (stop + pageBytes - 1) / pageBytes - cursor / pageBytes;
                if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(cursor), stop - cursor, pages + *count, &available, &granularity) != 0) fail("collect private guest writes");
                *count += available;
                if (*count == capacity) return true;
                cursor = stop;
            }
        }
        flushClean();
        return true;
    }

private:
    std::unique_lock<std::mutex> faultLock() {
        std::unique_lock lock(mutex, std::try_to_lock);
        if (lock.owns_lock()) return lock;
        faultWaiters.fetch_add(1, std::memory_order_relaxed);
        lock.lock();
        faultWaiters.fetch_sub(1, std::memory_order_relaxed);
        return lock;
    }

    bool yieldToFaults(std::unique_lock<std::mutex>& lock) {
        if (faultWaiters.load(std::memory_order_relaxed) == 0) return false;
        lock.unlock();
        for (std::uint32_t spin = 0; spin < faultYieldSpins && faultWaiters.load(std::memory_order_relaxed) != 0; ++spin) SwitchToThread();
        lock.lock();
        return true;
    }

    bool collectable(std::uintptr_t address, std::uintptr_t end) {
        const auto proven = collectableRanges.upper_bound(address);
        if (proven != collectableRanges.begin() && std::prev(proven)->second >= end) return true;
        auto position = views.lower_bound(address & ~(pageBytes - 1));
        for (auto cursor = address; cursor < end;) {
            const auto base = cursor & ~(pageBytes - 1);
            if (const auto found = viewAt(base, position); found != views.end()) {
                if (found->second.protection == PAGE_NOACCESS || (found->second.resident && !watchResident)) return false;
                cursor = std::min(end, base + pageBytes);
                continue;
            }
            const auto allocationEnd = privateAllocationEnd(cursor);
            if (allocationEnd == 0) return false;
            cursor = std::min(end, allocationEnd);
        }
        remember(collectableRanges, address, end);
        return true;
    }

    std::uintptr_t privateAllocationEnd(std::uintptr_t address) const {
        const auto next = privateAllocations.upper_bound(address);
        if (next == privateAllocations.begin()) return 0;
        const auto allocation = std::prev(next);
        return address < allocation->second ? allocation->second : 0;
    }

    static constexpr std::size_t pageBytes = 0x4000;
    static constexpr std::size_t subpageBytes = 0x1000;
    static_assert(subpageBytes == 4096, "Collect reports one page per subpage");
    static constexpr std::size_t subpages = pageBytes / subpageBytes;
    static constexpr std::uint8_t allSubpages = (1u << subpages) - 1;
    static constexpr std::size_t chunkBytes = 0x200000;
    struct SharedPage {
        std::array<std::uint64_t, subpages> generation = [] {
            std::array<std::uint64_t, subpages> initial{};
            initial.fill(1);
            return initial;
        }();
        std::vector<std::uintptr_t> aliases;
        std::uint32_t pins = 0;
        bool guarded = false;
        std::uint8_t accessGuarded = 0;
        std::uintptr_t section = 0;
        std::uint64_t offset = 0;
    };
    static constexpr std::size_t guardBytes = 0x10000;

    void dropGuard(SharedPage& page) {
        if (!page.guarded) return;
        page.guarded = false;
        for (const auto alias : page.aliases) {
            const auto guardPage = alias & ~(guardBytes - 1);
            if (std::find(guardReports.begin(), guardReports.end(), guardPage) != guardReports.end()) continue;
            guardReports.push_back(guardPage);
            for (auto piece = guardPage; piece < guardPage + guardBytes; piece += pageBytes) {
                if (const auto view = views.find(piece); view != views.end()) view->second.page->guarded = false;
            }
        }
    }
    struct PrivatePiece {
        DWORD protection = PAGE_NOACCESS;
        std::uint8_t armed = 0;
        std::uint8_t accessGuarded = 0;
        std::uint32_t pins = 0;
        bool guarded = false;
    };
    using PrivatePieces = std::map<std::uintptr_t, PrivatePiece>;

    static DWORD readOnly(DWORD protection) {
        return protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
    }

    PrivatePiece* privatePiece(std::uintptr_t piece) {
        if (const auto found = privatePieces.find(piece); found != privatePieces.end()) return &found->second;
        const auto memory = query(piece);
        if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || memory.RegionSize < pageBytes || (memory.Protect & ~DWORD{0xff}) != 0) return nullptr;
        PrivatePiece page;
        page.protection = memory.Protect;
        return &privatePieces.emplace(piece, page).first->second;
    }

    std::uintptr_t firstAccessGuarded(std::uintptr_t address, std::uintptr_t end) const {
        std::uintptr_t found = 0;
        const auto take = [&](std::uintptr_t piece, std::uint8_t guarded) {
            const auto mask = static_cast<std::uint8_t>(guarded & subpagesOf(piece, address, end));
            if (mask == 0) return;
            const auto first = piece + static_cast<std::uintptr_t>(std::countr_zero(mask)) * subpageBytes;
            if (found == 0 || first < found) found = first;
        };
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) take(it->first, it->second.page->accessGuarded);
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end; ++it) take(it->first, it->second.accessGuarded);
        return found;
    }

    bool unsettledImport(std::uintptr_t address, std::uintptr_t end) const {
        if (unsettledImports.empty() || end <= address) return false;
        for (auto page = address / guardBytes; page <= (end - 1) / guardBytes; ++page) {
            if (unsettledImports.contains(page)) return true;
        }
        return false;
    }

    GuardRefusal privateRefusal(std::uintptr_t piece) {
        for (const auto& [begin, end] : privateHostWrites) {
            if (begin < piece + pageBytes && piece < end) return GuardRefusal::PrivateHostWrite;
        }
        const auto* page = privatePiece(piece);
        if (page == nullptr) return GuardRefusal::Unavailable;
        return page->pins == 0 ? GuardRefusal::None : GuardRefusal::PrivatePinned;
    }

    GuardRefusal sharedRefusal(const auto& view) const {
        if (view.resident) return GuardRefusal::ResidentView;
        if (view.page->pins != 0) return GuardRefusal::SharedPinned;
        for (const auto alias : view.page->aliases) {
            if (views.at(alias).hostWrites != 0) return GuardRefusal::AliasHostWrite;
        }
        return GuardRefusal::None;
    }

    void clearRefusals(std::uintptr_t address, std::uintptr_t end, std::uint32_t cleared) {
        refusalClears.push_back({address, end - address, cleared});
        for (auto it = views.lower_bound(address); it != views.end() && it->first < end; ++it) {
            for (const auto alias : it->second.page->aliases) {
                if (alias != it->first) refusalClears.push_back({alias, pageBytes, cleared});
            }
        }
    }

    PrivatePieces::iterator pruned(PrivatePieces::iterator it) {
        const auto& page = it->second;
        if (page.guarded || page.armed != 0 || page.accessGuarded != 0 || page.pins != 0) return std::next(it);
        return privatePieces.erase(it);
    }

    void prunePrivate(const std::vector<std::uintptr_t>& pieces) {
        for (const auto piece : pieces) {
            if (const auto found = privatePieces.find(piece); found != privatePieces.end()) pruned(found);
        }
    }

    void dropPrivateGuard(std::uintptr_t piece) {
        const auto found = privatePieces.find(piece);
        if (found == privatePieces.end() || !found->second.guarded) return;
        const auto guardPage = piece & ~(guardBytes - 1);
        for (auto it = privatePieces.lower_bound(guardPage); it != privatePieces.end() && it->first < guardPage + guardBytes; ++it) it->second.guarded = false;
        if (std::find(guardReports.begin(), guardReports.end(), guardPage) == guardReports.end()) guardReports.push_back(guardPage);
    }

    void reprotectPrivate(std::uintptr_t address, std::uintptr_t end, DWORD protection) {
        for (auto it = privatePieces.lower_bound(address & ~(pageBytes - 1)); it != privatePieces.end() && it->first < end;) {
            dropPrivateGuard(it->first);
            it->second.protection = protection;
            it->second.armed = 0;
            it = pruned(it);
        }
    }

    bool handlePrivateWrite(std::uintptr_t base, std::uintptr_t address) {
        const auto found = privatePieces.find(base);
        if (found == privatePieces.end()) return false;
        auto& page = found->second;
        const auto written = subpagesOf(base, address, address + 1);
        if ((page.armed & written) == 0) {
            if (!writable(page.protection) || (page.accessGuarded & written) != 0) return false;
            const auto memory = query(address);
            return memory.State == MEM_COMMIT && writable(memory.Protect & 0xffu);
        }
        protectSubpages(base, written, page.protection, "resume private memory write");
        page.armed &= static_cast<std::uint8_t>(~written);
        dropPrivateGuard(base);
        pruned(found);
        return true;
    }

    struct Section {
        HANDLE handle;
        explicit Section(HANDLE handle) : handle(handle) {}
        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;
        ~Section() { CloseHandle(handle); }
    };
    struct Saved {
        unsigned char* bytes;
        explicit Saved(unsigned char* bytes) : bytes(bytes) {}
        Saved(const Saved&) = delete;
        Saved& operator=(const Saved&) = delete;
        ~Saved() { VirtualFree(bytes, 0, MEM_RELEASE); }
    };
    struct Physical {
        std::weak_ptr<SharedPage> page;
        bool stored = false;
        std::shared_ptr<Saved> saved;
        std::size_t savedOffset = 0;
    };
    struct View {
        std::shared_ptr<SharedPage> page;
        DWORD protection;
        std::array<std::uint64_t, subpages> seen;
        std::uint8_t armed;
        std::shared_ptr<Section> section;
        std::uint64_t offset;
        std::uint32_t hostWrites;
        std::uint8_t collected = 0;
        bool resident = false;
        std::shared_ptr<Physical> slot;
        std::uint8_t armedUnderAccess = 0;
    };
    struct Run {
        std::uintptr_t end;
        std::array<std::uint64_t, 8> collected;
        std::uint32_t fresh;
    };
    struct Window {
        void* view = nullptr;
        unsigned char* bytes = nullptr;
    };
    using Ranges = std::map<std::uintptr_t, std::uintptr_t>;

    static bool runCollected(const Run& run, std::size_t bit) {
        return (run.collected[bit >> 6] >> (bit & 63) & 1) != 0;
    }

    void collectResident(std::uintptr_t start, Run& run, std::uintptr_t from, std::uintptr_t stop, void** pages, std::size_t* count, bool clear, Collection collection) {
        const auto first = from & ~std::uintptr_t{4095};
        std::vector<std::uintptr_t> written;
        if (collection != Collection::Fresh) {
            std::vector<void*> found((stop - first + 4095) / 4096);
            ULONG_PTR reported = found.size();
            DWORD granularity = 0;
            if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(first), stop - first, found.data(), &reported, &granularity) != 0) fail("collect resident guest writes");
            for (ULONG_PTR index = 0; index < reported; ++index) written.push_back(reinterpret_cast<std::uintptr_t>(found[index]));
        }
        auto next = written.begin();
        std::uintptr_t resetFrom = 0;
        const auto flushReset = [&](std::uintptr_t to) {
            if (resetFrom != 0 && ResetWriteWatch(reinterpret_cast<void*>(resetFrom), to - resetFrom) != 0) fail("reset resident guest writes");
            resetFrom = 0;
        };
        for (auto at = first; at < stop; at += 4096) {
            const auto bit = (at - start) >> 12;
            const bool fresh = !runCollected(run, bit);
            bool wrote = false;
            while (next != written.end() && *next < at) ++next;
            if (next != written.end() && *next == at) wrote = true;
            const bool report = collection == Collection::Fresh ? fresh : collection == Collection::Armed ? !fresh && wrote : fresh || wrote;
            if (report) pages[(*count)++] = reinterpret_cast<void*>(std::max(at, from));
            const bool settle = clear && fresh && collection != Collection::Armed;
            if (settle) {
                run.collected[bit >> 6] |= 1ull << (bit & 63);
                --run.fresh;
                forget(freshRanges, at, at + 4096);
                if (collection == Collection::Fresh && resetFrom == 0) resetFrom = at;
            } else if (collection == Collection::Fresh) {
                flushReset(at);
            }
        }
        if (collection == Collection::Fresh) flushReset(stop);
    }

    Window open(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        const auto lead = offset % granularity;
        Window window;
        window.view = map(section, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (window.view == nullptr) fail("open a window on direct memory");
        window.bytes = static_cast<unsigned char*>(window.view) + lead;
        return window;
    }

    void close(Window& window) {
        if (window.view != nullptr && !unmap(GetCurrentProcess(), window.view, 0)) fail("close a window on direct memory");
        window = {};
    }

    void commitResident(std::uintptr_t base, std::size_t bytes) {
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watchResident ? MEM_WRITE_WATCH : 0);
        if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(base), bytes, flags, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder with resident direct memory");
    }

    void commitSection(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        auto window = open(section, offset, bytes);
        const auto memory = query(reinterpret_cast<std::uintptr_t>(window.bytes));
        if ((memory.State != MEM_COMMIT || memory.RegionSize < bytes) && !VirtualAlloc(window.bytes, bytes, MEM_COMMIT, PAGE_READWRITE)) {
            const auto error = GetLastError();
            close(window);
            throw std::system_error(static_cast<int>(error), std::system_category(), "commit direct memory pages");
        }
        close(window);
    }

    void mapShared(std::uintptr_t base, HANDLE section, std::uint64_t offset, DWORD protection, bool committed = false) {
        void* page = reinterpret_cast<void*>(base);
        if (!map(section, GetCurrentProcess(), page, offset, pageBytes, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0)) fail("map shared guest page");
        if (!committed && query(base).State == MEM_RESERVE && !VirtualAlloc(page, pageBytes, MEM_COMMIT, PAGE_EXECUTE_READWRITE)) fail("commit shared guest page");
        DWORD previous;
        if (!VirtualProtect(page, pageBytes, protection, &previous)) fail("protect shared guest page");
    }

    static bool blank(std::uintptr_t page) {
        const auto* words = reinterpret_cast<const std::uint64_t*>(page);
        for (std::size_t i = 0; i < pageBytes / sizeof(std::uint64_t); ++i) {
            if (words[i] != 0) return false;
        }
        return true;
    }

    void demote(std::uintptr_t address) {
        const auto run = std::prev(residentRuns.upper_bound(address));
        const auto start = run->first;
        const auto stop = run->second.end;
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("fence resident guest pages");
        const auto section = views.at(start).section;
        demotedRuns.push_back({start, stop, views.at(start).protection});
        refusalClears.push_back({start, stop - start, RefusalBit(GuardRefusal::ResidentView)});
        auto window = open(section->handle, views.at(start).offset, stop - start);
        if (query(reinterpret_cast<std::uintptr_t>(window.bytes)).State == MEM_RESERVE && !VirtualAlloc(window.bytes, stop - start, MEM_COMMIT, PAGE_READWRITE)) fail("commit direct memory pages");
        std::memcpy(window.bytes, reinterpret_cast<const void*>(start), stop - start);
        close(window);
        if (!VirtualFree(reinterpret_cast<void*>(start), stop - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release resident guest pages");
        for (auto page = start; page < stop; page += pageBytes) {
            auto& view = views.at(page);
            split(page, pageBytes);
            mapShared(page, view.section->handle, view.offset, view.protection);
            view.slot->stored = true;
            view.resident = false;
            view.seen.fill(0);
            view.armed = 0;
            view.collected = 0;
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                if (runCollected(run->second, ((page - start) >> 12) + sub)) view.collected |= static_cast<std::uint8_t>(1u << sub);
            }
            invalidate(*view.page);
        }
        residentRuns.erase(run);
    }

    void breakRun(std::uintptr_t at) {
        const auto next = residentRuns.upper_bound(at);
        if (next == residentRuns.begin()) return;
        const auto run = std::prev(next);
        if (at > run->first && at < run->second.end) demote(at);
    }

    void evict(std::map<std::uintptr_t, Run>::iterator run) {
        const auto start = run->first;
        const auto stop = run->second.end;
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("open resident guest pages");
        std::shared_ptr<Saved> saved;
        for (auto page = start; page < stop; page += pageBytes) {
            const auto found = views.find(page);
            auto& view = found->second;
            if (!blank(page)) {
                if (!saved) {
                    auto* bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, stop - start, MEM_RESERVE, PAGE_READWRITE));
                    if (bytes == nullptr) fail("reserve unmapped direct memory");
                    saved = std::make_shared<Saved>(bytes);
                }
                if (!VirtualAlloc(saved->bytes + (page - start), pageBytes, MEM_COMMIT, PAGE_READWRITE)) fail("keep unmapped direct memory");
                std::memcpy(saved->bytes + (page - start), reinterpret_cast<const void*>(page), pageBytes);
                view.slot->saved = saved;
                view.slot->savedOffset = page - start;
            }
            std::erase(view.page->aliases, page);
            views.erase(found);
        }
        forget(freshRanges, start, stop);
        residentRuns.erase(run);
    }

    static void forget(Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second > start) --it;
        while (it != ranges.end() && it->first < end) {
            const auto first = it->first;
            const auto last = it->second;
            it = ranges.erase(it);
            if (first < start) ranges.emplace(first, start);
            if (last > end) it = ranges.emplace(end, last).first;
        }
    }

    static void remember(Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second >= start) --it;
        while (it != ranges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = ranges.erase(it);
        }
        ranges.emplace(start, end);
    }

    void forgetClean(std::uintptr_t start, std::uintptr_t end) {
        forget(cleanRanges, start, end);
    }

    void rememberClean(std::uintptr_t start, std::uintptr_t end) {
        remember(cleanRanges, start, end);
    }

    void invalidate(SharedPage& page, std::uint8_t mask = allSubpages) {
        for (std::size_t sub = 0; sub < subpages; ++sub) {
            if ((mask & (1u << sub)) == 0) continue;
            ++page.generation[sub];
            for (const auto alias : page.aliases) forgetClean(alias + sub * subpageBytes, alias + (sub + 1) * subpageBytes);
        }
    }

    static std::uint8_t subpagesOf(std::uintptr_t base, std::uintptr_t address, std::uintptr_t end) {
        const auto from = std::max(address, base);
        const auto to = std::min(end, base + pageBytes);
        if (to <= from) return 0;
        const auto first = (from - base) / subpageBytes;
        const auto last = (to - 1 - base) / subpageBytes;
        return static_cast<std::uint8_t>(((1u << (last + 1)) - 1) & ~((1u << first) - 1));
    }

    static void protectSubpages(std::uintptr_t base, std::uint8_t mask, DWORD protection, const char* operation) {
        for (std::size_t sub = 0; sub < subpages;) {
            if ((mask & (1u << sub)) == 0) {
                ++sub;
                continue;
            }
            auto stop = sub;
            while (stop < subpages && (mask & (1u << stop)) != 0) ++stop;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(base + sub * subpageBytes), (stop - sub) * subpageBytes, protection, &previous)) fail(operation);
            sub = stop;
        }
    }

    bool sameSection(HANDLE first, HANDLE second) const {
        return compare != nullptr && compare(first, second);
    }

    static bool writable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE;
    }
    using AllocateFunction = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapFunction = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapFunction = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);
    using CompareFunction = BOOL (WINAPI*)(HANDLE, HANDLE);

    WindowsMappings() {
        const auto module = GetModuleHandleW(L"KernelBase.dll");
        if (!module) fail("load Windows memory API");
        allocate = reinterpret_cast<AllocateFunction>(GetProcAddress(module, "VirtualAlloc2"));
        map = reinterpret_cast<MapFunction>(GetProcAddress(module, "MapViewOfFile3"));
        unmap = reinterpret_cast<UnmapFunction>(GetProcAddress(module, "UnmapViewOfFile2"));
        if (!allocate || !map || !unmap) throw std::runtime_error("Windows placeholder memory APIs are required");
        compare = reinterpret_cast<CompareFunction>(GetProcAddress(module, "CompareObjectHandles"));
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        granularity = system.dwAllocationGranularity;
    }

    [[noreturn]] static void fail(const char* operation) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }

    static MEMORY_BASIC_INFORMATION query(std::uintptr_t address) {
        MEMORY_BASIC_INFORMATION memory{};
        queries.fetch_add(1, std::memory_order_relaxed);
        if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory)) fail("query guest memory");
        return memory;
    }

    static std::uintptr_t split(std::uintptr_t address, std::size_t bytes) {
        auto memory = query(address);
        memory = query(reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
        if (memory.State != MEM_RESERVE) throw std::runtime_error("guest mapping requires a placeholder");
        const auto base = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
        if (address != base) {
            if (!VirtualFree(reinterpret_cast<void*>(base), address - base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder prefix");
            memory = query(address);
        }
        if (memory.RegionSize < bytes) throw std::runtime_error("guest placeholder is too small");
        if (memory.RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
        return address + memory.RegionSize;
    }

    void reset(std::uintptr_t address, std::size_t bytes) {
        collectableRanges.clear();
        const auto end = address + bytes;
        privatePieces.erase(privatePieces.lower_bound(address), privatePieces.lower_bound(end));
        privateAllocations.erase(privateAllocations.lower_bound(address), privateAllocations.lower_bound(end));
        for (auto page = (address + guardBytes - 1) / guardBytes; page < end / guardBytes; ++page) unsettledImports.erase(page);
        forgetClean(address, end);
        breakRun(address);
        breakRun(end);
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State == MEM_RESERVE) {
                cursor = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                continue;
            }
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != cursor) throw std::runtime_error("cannot release part of a host allocation");
            auto allocationEnd = cursor;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd > end || query(allocationEnd).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            if (memory.Type == MEM_MAPPED) {
                if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(cursor), MEM_PRESERVE_PLACEHOLDER)) fail("unmap shared guest page");
                const auto found = views.find(cursor);
                if (found != views.end()) {
                    std::erase(found->second.page->aliases, cursor);
                    views.erase(found);
                    forget(freshRanges, cursor, cursor + pageBytes);
                }
            } else if (memory.Type == MEM_PRIVATE) {
                if (const auto run = residentRuns.find(cursor); run != residentRuns.end()) evict(run);
                if (!VirtualFree(reinterpret_cast<void*>(cursor), allocationEnd - cursor, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            } else {
                throw std::runtime_error("unsupported guest mapping type");
            }
            cursor = allocationEnd;
        }
        const auto last = query(reinterpret_cast<std::uintptr_t>(query(end - 1).AllocationBase));
        const auto lastBase = reinterpret_cast<std::uintptr_t>(last.BaseAddress);
        if (lastBase + last.RegionSize > end) split(lastBase, end - lastBase);
        const auto first = query(address);
        split(address, std::min(bytes, reinterpret_cast<std::uintptr_t>(first.BaseAddress) + first.RegionSize - address));
        if (query(address).RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) fail("coalesce guest placeholders");
    }

    void* mapAliasRuns(std::uintptr_t address, std::size_t bytes) {
        auto* reserved = allocate(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        if (reserved == nullptr) fail("reserve a read-write alias of shared guest memory");
        const auto base = reinterpret_cast<std::uintptr_t>(reserved);
        try {
            for (std::size_t done = 0; done < bytes;) {
                const auto& first = views.at(address + done);
                auto run = pageBytes;
                while (done + run < bytes) {
                    const auto& next = views.at(address + done + run);
                    if (next.offset != first.offset + run || (next.section != first.section && !sameSection(next.section->handle, first.section->handle))) break;
                    run += pageBytes;
                }
                split(base + done, run);
                if (!map(first.section->handle, GetCurrentProcess(), reinterpret_cast<void*>(base + done), first.offset, run, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) fail("map a run of a read-write alias of shared guest memory");
                done += run;
            }
        } catch (...) {
            releaseAlias(base, bytes);
            throw;
        }
        splitAliases.emplace(base, bytes);
        return reserved;
    }

    DWORD releaseAlias(std::uintptr_t base, std::size_t bytes) {
        DWORD error = ERROR_SUCCESS;
        for (auto cursor = base; cursor < base + bytes;) {
            const auto memory = query(cursor);
            const auto released = memory.State == MEM_RESERVE ? VirtualFree(memory.BaseAddress, 0, MEM_RELEASE) : unmap(GetCurrentProcess(), memory.BaseAddress, 0);
            if (!released && error == ERROR_SUCCESS) error = GetLastError();
            cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        }
        return error;
    }

    Ranges cleanRanges;
    Ranges collectableRanges;
    Ranges privateAllocations;
    std::uint64_t collectSystemCalls = 0;
    std::uint64_t collectPieces = 0;
    Ranges freshRanges;
    std::map<std::uintptr_t, View> views;
    Guarded guardReports;
    std::vector<RefusalClear> refusalClears;

    using Views = std::map<std::uintptr_t, View>;

    Views::iterator viewAt(std::uintptr_t base, Views::iterator& position) {
        if (position != views.end() && position->first < base) {
            ++position;
            if (position != views.end() && position->first < base) position = views.lower_bound(base);
        }
        return position != views.end() && position->first == base ? position : views.end();
    }
    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::shared_ptr<Physical>> physical;
    std::map<std::uintptr_t, Run> residentRuns;
    PrivatePieces privatePieces;
    std::unordered_map<std::uintptr_t, std::uint64_t> unsettledImports;
    std::uint64_t importStamp = 0;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> privateHostWrites;
    std::vector<Demoted> demotedRuns;
    std::map<std::uintptr_t, std::size_t> splitAliases;
    static constexpr std::size_t faultYieldBytes = 0x1000000;
    static constexpr std::uint32_t faultYieldSpins = 4096;
    std::atomic<std::uint32_t> faultWaiters{0};
    std::size_t granularity = 0x10000;
    bool watchResident = false;
    bool sharedImports = false;
    static inline std::atomic<std::uint64_t> queries{0};
    std::mutex mutex;
    AllocateFunction allocate = nullptr;
    MapFunction map = nullptr;
    UnmapFunction unmap = nullptr;
    CompareFunction compare = nullptr;
};

}
#endif

#endif
