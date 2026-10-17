#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

struct PageSpan {
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    bool empty = true;
};

PageSpan PagesOf(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return {};
    return {address >> ResidencyPageShift, (address + (bytes - 1)) >> ResidencyPageShift, false};
}

constexpr std::uint32_t WordBits = 64;
constexpr std::uint64_t TouchPagesPerFrame = 4096;

ResidencyMask RunMask(std::uint32_t begin, std::uint32_t end) {
    if (end <= begin) return 0;
    const auto first = begin >> ResidencySubpageShift;
    const auto last = (end - 1) >> ResidencySubpageShift;
    const auto count = last - first + 1;
    const std::uint32_t bits = count >= ResidencySubpages ? 0xffffu : (1u << count) - 1u;
    return static_cast<ResidencyMask>(bits << first);
}

}

ResidencyToken MergeTokens(const ResidencyToken& first, const ResidencyToken& second) {
    ResidencyToken merged{};
    for (std::size_t stream = 0; stream < ResidencyStreams; ++stream) merged[stream] = std::max(first[stream], second[stream]);
    return merged;
}

bool TokenEmpty(const ResidencyToken& token) {
    return std::all_of(token.begin(), token.end(), [](std::uint64_t value) { return value == 0; });
}

ResidencyMask SubpageMask(std::uint64_t page, std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return 0;
    const auto base = page << ResidencyPageShift;
    const auto first = std::max(address, base);
    const auto last = std::min(address + (bytes - 1), base + (ResidencyPageBytes - 1));
    if (first > last) return 0;
    const auto firstSubpage = static_cast<std::uint32_t>((first - base) >> ResidencySubpageShift);
    const auto lastSubpage = static_cast<std::uint32_t>((last - base) >> ResidencySubpageShift);
    const auto count = lastSubpage - firstSubpage + 1;
    const std::uint32_t bits = count >= ResidencySubpages ? 0xffffu : (1u << count) - 1u;
    return static_cast<ResidencyMask>(bits << firstSubpage);
}

ResidencyManager::ResidencyManager(IResidencyDevice& device, const ResidencyPolicy& policy)
    : device(device), chunkSlots(policy.chunkSlots), hotFaults(policy.hotFaults), hotEpochs(policy.hotEpochs), dirtyLeaves(new std::atomic<DirtyLeaf*>[DirtyLeaves]()) {
    if (chunkSlots == 0 || chunkSlots > (std::numeric_limits<std::uint32_t>::max() >> ResidencyPageShift))
        throw std::invalid_argument("residency: a chunk holds 1 to 65535 pages, not " + std::to_string(chunkSlots));
    if (hotEpochs == 0 || hotEpochs > 64 || hotFaults == 0 || hotFaults > hotEpochs)
        throw std::invalid_argument("residency: a hot page faults in 1 to " + std::to_string(hotEpochs) + " of the last 1 to 64 epochs, not " + std::to_string(hotFaults) + " of " + std::to_string(hotEpochs));
    SetBudget(policy.budgetBytes);
}

ResidencyManager::~ResidencyManager() {
    std::vector<std::uint64_t> guarded;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].guarded) guarded.push_back(pages[id].index);
        if (pages[id].accessGuarded) accessGuards.insert(pages[id].index);
    }
    if (!guarded.empty()) device.Unguard(guarded);
    for (const auto page : accessGuards) device.UnguardAccess(page);
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunks[chunk].live) device.ReleaseChunk(chunk);
    }
    for (std::size_t leaf = 0; leaf < DirtyLeaves; ++leaf) delete dirtyLeaves[leaf].load(std::memory_order_relaxed);
}

void ResidencyManager::publishDirty(std::uint64_t page, ResidencyMask mask) {
    const auto leafIndex = page >> DirtyLeafShift;
    if (leafIndex >= DirtyLeaves) throw std::out_of_range("residency: page " + std::to_string(page) + " lies past the 47-bit guest address space");
    auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
    if (leaf == nullptr) {
        if (mask == 0) return;
        leaf = new DirtyLeaf();
        dirtyLeaves[leafIndex].store(leaf, std::memory_order_release);
    }
    leaf->masks[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].store(mask, std::memory_order_release);
}

void ResidencyManager::publishStale(std::uint64_t page, ResidencyMask mask) {
    const auto leafIndex = page >> DirtyLeafShift;
    if (leafIndex >= DirtyLeaves) throw std::out_of_range("residency: page " + std::to_string(page) + " lies past the 47-bit guest address space");
    auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
    if (leaf == nullptr) {
        if (mask == 0) return;
        leaf = new DirtyLeaf();
        dirtyLeaves[leafIndex].store(leaf, std::memory_order_release);
    }
    leaf->stale[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].store(mask, std::memory_order_release);
}

void ResidencyManager::publishUnsettled(std::uint64_t page, std::uint8_t reason, bool set) {
    const auto leafIndex = page >> DirtyLeafShift;
    if (leafIndex >= DirtyLeaves) throw std::out_of_range("residency: page " + std::to_string(page) + " lies past the 47-bit guest address space");
    auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
    if (leaf == nullptr) {
        if (!set) return;
        leaf = new DirtyLeaf();
        dirtyLeaves[leafIndex].store(leaf, std::memory_order_release);
    }
    auto& bits = leaf->unsettled[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)];
    const auto before = bits.load(std::memory_order_relaxed);
    const auto after = static_cast<std::uint8_t>(set ? before | reason : before & ~reason);
    if (after == before) return;
    if (before == 0) unsettledPages.fetch_add(1, std::memory_order_acq_rel);
    bits.store(after, std::memory_order_release);
    if (after == 0) unsettledPages.fetch_sub(1, std::memory_order_acq_rel);
}

void ResidencyManager::addUnsettled(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint64_t page, const ResidencyToken& token, std::uint8_t reason) {
    auto& entry = entries[page];
    entry = MergeTokens(entry, token);
    publishUnsettled(page, reason, true);
}

void ResidencyManager::eraseUnsettled(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint64_t page, std::uint8_t reason) {
    if (entries.erase(page) != 0) publishUnsettled(page, reason, false);
}

bool ResidencyManager::CpuReadSettled(std::uint64_t address, std::uint64_t bytes) const {
    if (DirtyIn(address, bytes)) return false;
    if (unsettledPages.load(std::memory_order_acquire) == 0) return true;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return true;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto leafIndex = page >> DirtyLeafShift;
        if (leafIndex >= DirtyLeaves) return true;
        const auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            page |= (std::uint64_t{1} << DirtyLeafShift) - 1;
            continue;
        }
        if (leaf->unsettled[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].load(std::memory_order_acquire) != 0) return false;
    }
    return true;
}

bool ResidencyManager::DirtyIn(std::uint64_t address, std::uint64_t bytes) const {
    const auto span = PagesOf(address, bytes);
    if (span.empty || (dirtyPages.load(std::memory_order_acquire) == 0 && staleCount.load(std::memory_order_acquire) == 0)) return false;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto leafIndex = page >> DirtyLeafShift;
        if (leafIndex >= DirtyLeaves) return false;
        const auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            page |= (std::uint64_t{1} << DirtyLeafShift) - 1;
            continue;
        }
        const auto index = page & ((std::uint64_t{1} << DirtyLeafShift) - 1);
        const auto mask = static_cast<ResidencyMask>(leaf->masks[index].load(std::memory_order_acquire) | leaf->stale[index].load(std::memory_order_acquire));
        if ((mask & SubpageMask(page, address, bytes)) != 0) return true;
    }
    return false;
}

bool ResidencyManager::StaleIn(std::uint64_t address, std::uint64_t bytes) const {
    if (staleCount.load(std::memory_order_acquire) == 0 || bytes == 0) return false;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return false;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto leafIndex = page >> DirtyLeafShift;
        if (leafIndex >= DirtyLeaves) return false;
        const auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            page |= (std::uint64_t{1} << DirtyLeafShift) - 1;
            continue;
        }
        const auto mask = leaf->stale[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].load(std::memory_order_acquire);
        if ((mask & SubpageMask(page, address, bytes)) != 0) return true;
    }
    return false;
}

void ResidencyManager::publishPage(std::uint64_t page) {
    ResidencyMask mask = 0;
    if (const auto id = find(page)) mask = pages[*id].dirty;
    publishDirty(page, mask);
    const std::shared_lock lock(ownedMutex);
    ResidencyMask stale = 0;
    if (const auto entry = owned.find(page); entry != owned.end()) stale = entry->second.stale;
    publishStale(page, stale);
}

bool ResidencyManager::isStale(std::uint64_t page) const {
    const std::shared_lock lock(ownedMutex);
    return staleOwned.contains(page);
}

std::vector<std::uint64_t> ResidencyManager::staleIn(std::uint64_t address, std::uint64_t bytes) const {
    std::vector<std::uint64_t> stale;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    const std::shared_lock lock(ownedMutex);
    if (span.empty || staleOwned.empty()) return stale;
    if (span.last - span.first >= staleOwned.size()) {
        for (const auto page : staleOwned) {
            if (page >= span.first && page <= span.last) stale.push_back(page);
        }
        std::sort(stale.begin(), stale.end());
        return stale;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (staleOwned.contains(page)) stale.push_back(page);
    }
    return stale;
}

bool ResidencyManager::retileStale(std::span<const std::uint64_t> stalePages, std::span<const IResidencyOwner* const> except) {
    struct Work {
        std::shared_ptr<IResidencyOwner> owner;
        std::vector<ResidencyRange> ranges;
    };
    std::vector<Work> work;
    {
        const std::unique_lock lock(ownedMutex);
        for (const auto page : stalePages) {
            const auto entry = owned.find(page);
            if (entry == owned.end() || entry->second.stale == 0) continue;
            auto* owner = entry->second.owner;
            if (std::find(except.begin(), except.end(), owner) != except.end()) continue;
            std::uint32_t bits = entry->second.stale;
            entry->second.stale = 0;
            if (staleOwned.erase(page) != 0) --staleCount;
            publishStale(page, 0);
            auto at = std::find_if(work.begin(), work.end(), [&](const Work& item) { return item.owner.get() == owner; });
            if (at == work.end()) {
                auto held = owner->Hold();
                if (held == nullptr) continue;
                at = work.insert(work.end(), {std::move(held), {}});
            }
            auto& ranges = at->ranges;
            while (bits != 0) {
                const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
                const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
                const auto address = (page << ResidencyPageShift) + (std::uint64_t{first} << ResidencySubpageShift);
                const auto length = std::uint64_t{run} << ResidencySubpageShift;
                if (!ranges.empty() && ranges.back().address + ranges.back().bytes == address) ranges.back().bytes += length;
                else ranges.push_back({address, length});
                counts.retiledBytes += length;
                bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
            }
        }
    }
    for (const auto& item : work) {
        ++counts.retiles;
        item.owner->Retile(item.ranges);
    }
    return !work.empty();
}

void ResidencyManager::Own(std::uint64_t address, std::uint64_t bytes, IResidencyOwner* owner) {
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    std::vector<std::uint64_t> previous;
    bool changed = false;
    {
        const std::shared_lock lock(ownedMutex);
        for (auto page = span.first; page <= span.last; ++page) {
            const auto entry = owned.find(page);
            if (entry == owned.end() || entry->second.owner != owner) changed = true;
            if (entry != owned.end() && entry->second.owner != owner && entry->second.stale != 0) previous.push_back(page);
        }
    }
    retileStale(previous);
    if (changed) {
        const std::unique_lock lock(ownedMutex);
        for (auto page = span.first; page <= span.last; ++page) owned[page].owner = owner;
        ownershipSerial.fetch_add(1, std::memory_order_acq_rel);
    }
    admit(span.first, span.last);
}

void ResidencyManager::MarkStale(std::uint64_t address, std::uint64_t bytes, IResidencyOwner* owner) {
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    std::vector<ResidencyRange> eager;
    const auto addEager = [&](std::uint64_t begin, std::uint64_t end) {
        if (!eager.empty() && eager.back().address + eager.back().bytes == begin) eager.back().bytes += end - begin;
        else eager.push_back({begin, end - begin});
    };
    for (auto page = span.first; page <= span.last; ++page) {
        const auto base = page << ResidencyPageShift;
        const auto begin = std::max(address, base);
        const auto end = std::min(address + bytes, base + ResidencyPageBytes);
        const auto mask = SubpageMask(page, address, bytes);
        const auto id = find(page);
        bool mine = false;
        {
            const std::shared_lock lock(ownedMutex);
            const auto entry = owned.find(page);
            mine = entry != owned.end() && entry->second.owner == owner;
        }
        if (!mine) {
            addEager(begin, end);
            continue;
        }
        if (id && pages[*id].shared && !aliasKeys.contains(page)) {
            auto& record = pages[*id];
            const auto adding = static_cast<ResidencyMask>(mask & ~record.accessMask);
            if (adding != 0) {
                if (device.GuardAccess(page, adding)) {
                    record.accessMask = static_cast<ResidencyMask>(record.accessMask | adding);
                    record.accessGuarded = true;
                    accessGuards.insert(page);
                } else {
                    ++counts.accessUnguardable;
                }
            }
        }
        const std::unique_lock lock(ownedMutex);
        auto& entry = owned[page];
        if (entry.stale == 0 && staleOwned.insert(page).second) ++staleCount;
        entry.stale = static_cast<ResidencyMask>(entry.stale | mask);
        publishStale(page, entry.stale);
    }
    if (eager.empty()) return;
    ++counts.eagerRetiles;
    for (const auto& range : eager) counts.retiledBytes += range.bytes;
    owner->Retile(eager);
}

bool ResidencyManager::RetileOwned(std::uint64_t address, std::uint64_t bytes, std::span<const IResidencyOwner* const> except) {
    applyQueued();
    if (staleCount.load(std::memory_order_acquire) == 0) return false;
    return retileStale(staleIn(address, bytes), except);
}

bool ResidencyManager::RetileAllOwned(std::span<const IResidencyOwner* const> except) {
    std::vector<std::uint64_t> stale;
    {
        const std::shared_lock lock(ownedMutex);
        if (staleOwned.empty()) return false;
        stale.assign(staleOwned.begin(), staleOwned.end());
    }
    std::sort(stale.begin(), stale.end());
    return retileStale(stale, except);
}

void ResidencyManager::Disown(IResidencyOwner* owner) {
    applyQueued();
    std::vector<std::uint64_t> stale;
    {
        const std::shared_lock lock(ownedMutex);
        for (const auto& [page, record] : owned) {
            if (record.owner == owner && record.stale != 0) stale.push_back(page);
        }
    }
    std::sort(stale.begin(), stale.end());
    retileStale(stale);
    Forget(owner);
}

void ResidencyManager::Forget(IResidencyOwner* owner) {
    const std::unique_lock lock(ownedMutex);
    for (auto entry = owned.begin(); entry != owned.end();) {
        if (entry->second.owner != owner) {
            ++entry;
            continue;
        }
        if (entry->second.stale != 0) {
            if (staleOwned.erase(entry->first) != 0) --staleCount;
            publishStale(entry->first, 0);
        }
        entry = owned.erase(entry);
    }
    ownershipSerial.fetch_add(1, std::memory_order_acq_rel);
}

void ResidencyManager::ClearStale(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) {
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    const std::unique_lock lock(ownedMutex);
    for (auto page = span.first; page <= span.last; ++page) {
        const auto entry = owned.find(page);
        if (entry == owned.end() || entry->second.owner != owner || entry->second.stale == 0) continue;
        entry->second.stale = static_cast<ResidencyMask>(entry->second.stale & ~SubpageMask(page, address, bytes));
        publishStale(page, entry->second.stale);
        if (entry->second.stale == 0 && staleOwned.erase(page) != 0) --staleCount;
    }
}

IResidencyOwner* ResidencyManager::OwnerOf(std::uint64_t page) const {
    const std::shared_lock lock(ownedMutex);
    const auto entry = owned.find(page);
    return entry == owned.end() ? nullptr : entry->second.owner;
}

ResidencyMask ResidencyManager::Stale(std::uint64_t page) const {
    const std::shared_lock lock(ownedMutex);
    const auto entry = owned.find(page);
    return entry == owned.end() ? ResidencyMask{0} : entry->second.stale;
}

std::vector<std::shared_ptr<IResidencyOwner>> ResidencyManager::HoldOwners(std::uint64_t address, std::uint64_t bytes) const {
    std::vector<std::shared_ptr<IResidencyOwner>> held;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return held;
    std::vector<IResidencyOwner*> seen;
    const auto take = [&](IResidencyOwner* owner) {
        if (std::find(seen.begin(), seen.end(), owner) != seen.end()) return;
        seen.push_back(owner);
        if (auto alive = owner->Hold()) held.push_back(std::move(alive));
    };
    const std::shared_lock lock(ownedMutex);
    if (owned.empty()) return held;
    if (span.last - span.first >= owned.size()) {
        std::vector<std::pair<std::uint64_t, IResidencyOwner*>> found;
        for (const auto& [page, record] : owned) {
            if (page >= span.first && page <= span.last) found.emplace_back(page, record.owner);
        }
        std::sort(found.begin(), found.end());
        for (const auto& [page, owner] : found) take(owner);
        return held;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (const auto entry = owned.find(page); entry != owned.end()) take(entry->second.owner);
    }
    return held;
}

std::vector<std::shared_ptr<IResidencyOwner>> ResidencyManager::HoldStaleOwners() const {
    std::vector<std::shared_ptr<IResidencyOwner>> held;
    std::vector<IResidencyOwner*> seen;
    const std::shared_lock lock(ownedMutex);
    for (const auto page : staleOwned) {
        const auto entry = owned.find(page);
        if (entry == owned.end() || std::find(seen.begin(), seen.end(), entry->second.owner) != seen.end()) continue;
        seen.push_back(entry->second.owner);
        if (auto alive = entry->second.owner->Hold()) held.push_back(std::move(alive));
    }
    return held;
}

bool ResidencyManager::StaleOfOthers(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) const {
    if (!StaleIn(address, bytes)) return false;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    const std::shared_lock lock(ownedMutex);
    if (span.last - span.first >= staleOwned.size()) {
        for (const auto page : staleOwned) {
            if (page < span.first || page > span.last) continue;
            const auto entry = owned.find(page);
            if (entry != owned.end() && entry->second.owner != owner && (entry->second.stale & SubpageMask(page, address, bytes)) != 0) return true;
        }
        return false;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        const auto entry = owned.find(page);
        if (entry != owned.end() && entry->second.owner != owner && (entry->second.stale & SubpageMask(page, address, bytes)) != 0) return true;
    }
    return false;
}

void ResidencyManager::WriteBackAll() {
    applyQueued();
    RetileAllOwned();
    std::vector<std::uint32_t> ids;
    ResidencyToken pending{};
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].dirtyRuns.empty()) continue;
        ids.push_back(id);
        const auto writer = writers.find(pages[id].index);
        if (writer != writers.end() && !settled(writer->second)) pending = MergeTokens(pending, writer->second);
    }
    if (!TokenEmpty(pending)) waitFor(pending);
    writeBackDirty(ids);
    ResidencyToken wait{};
    for (const auto& [page, token] : writingBack) wait = MergeTokens(wait, token);
    if (!settled(wait)) waitFor(wait);
    eraseUnsettledIf(writingBack, UnsettledWriteBack, [](const auto&) { return true; });
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (!pages[id].accessGuarded) continue;
        accessGuards.insert(pages[id].index);
        pages[id].accessGuarded = false;
        pages[id].accessMask = 0;
    }
    for (const auto page : accessGuards) device.UnguardAccess(page);
    accessGuards.clear();
}

bool ResidencyManager::settled(const ResidencyToken& token) {
    return TokenEmpty(token) || device.Reached(token);
}

void ResidencyManager::waitFor(const ResidencyToken& token) {
    ++counts.waits;
    device.Wait(token);
}

std::optional<std::uint32_t> ResidencyManager::find(std::uint64_t page) const {
    const auto found = resident.find(page);
    if (found == resident.end()) return std::nullopt;
    return found->second;
}

std::vector<std::uint32_t> ResidencyManager::residentIn(std::uint64_t address, std::uint64_t bytes) const {
    std::vector<std::uint32_t> ids;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return ids;
    if (span.last - span.first >= resident.size()) {
        for (const auto& [page, id] : resident) {
            if (page >= span.first && page <= span.last) ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end(), [this](std::uint32_t left, std::uint32_t right) { return pages[left].index < pages[right].index; });
        return ids;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (const auto id = find(page)) ids.push_back(*id);
    }
    return ids;
}

std::vector<ResidencyManager::Fill> ResidencyManager::withAliases(std::uint64_t address, std::uint64_t bytes) {
    std::vector<Fill> fills;
    if (bytes == 0) return fills;
    auto spans = device.Aliases(address, bytes);
    spans.push_back({address, bytes});
    std::map<std::uint32_t, Run> runs;
    for (const auto& range : spans) {
        for (const auto id : residentIn(range.address, range.bytes)) {
            const auto base = pages[id].index << ResidencyPageShift;
            const auto begin = static_cast<std::uint32_t>(std::max(range.address, base) - base);
            const auto end = static_cast<std::uint32_t>(std::min(range.address + range.bytes, base + ResidencyPageBytes) - base);
            const auto [entry, inserted] = runs.emplace(id, Run{begin, end});
            if (inserted) continue;
            entry->second.first = std::min(entry->second.first, begin);
            entry->second.second = std::max(entry->second.second, end);
        }
    }
    for (const auto& [id, run] : runs) fills.push_back({id, run.first, run.second});
    std::sort(fills.begin(), fills.end(), [this](const Fill& left, const Fill& right) { return pages[left.id].index < pages[right.id].index; });
    return fills;
}

void ResidencyManager::unlink(std::uint32_t id) {
    auto& page = pages[id];
    if (page.older != None) pages[page.older].newer = page.newer;
    else oldest = page.newer;
    if (page.newer != None) pages[page.newer].older = page.older;
    else newest = page.older;
    page.older = None;
    page.newer = None;
}

void ResidencyManager::linkNewest(std::uint32_t id) {
    auto& page = pages[id];
    page.older = newest;
    page.newer = None;
    if (newest != None) pages[newest].newer = id;
    else oldest = id;
    newest = id;
}

void ResidencyManager::changed(std::uint64_t page) {
    changes.insert(page);
    ++serial;
}

std::uint32_t ResidencyManager::newPage(std::uint64_t page, MirrorSlot slot, const ResidencyImport& import, bool guarded) {
    std::uint32_t id;
    if (!freePages.empty()) {
        id = freePages.back();
        freePages.pop_back();
    } else {
        id = static_cast<std::uint32_t>(pages.size());
        pages.emplace_back();
    }
    auto& record = pages[id];
    record = Page{};
    record.index = page;
    record.slot = slot;
    record.gpuWritable = import.gpuWritable;
    record.cpuWritable = import.cpuWritable;
    record.shared = import.section != 0;
    record.importSerial = import.serial;
    record.guarded = guarded;
    {
        const std::lock_guard lock(residentMutex);
        resident.emplace(page, id);
    }
    linkNewest(id);
    changed(page);
    return id;
}

void ResidencyManager::releasePage(std::uint32_t id) {
    pages[id] = Page{};
    freePages.push_back(id);
}

void ResidencyManager::forget(std::uint32_t id) {
    const auto page = pages[id].index;
    unlink(id);
    {
        const std::lock_guard lock(residentMutex);
        resident.erase(page);
    }
    if (!pages[id].dirtyRuns.empty()) --dirtyPages;
    pages[id].dirtyRuns.clear();
    pages[id].dirty = 0;
    publishDirty(page, 0);
    unmarkHot(page);
    eraseUnsettled(writers, page, UnsettledWriter);
    if (const auto key = aliasKeys.find(page); key != aliasKeys.end()) {
        canonical.erase(key->second);
        aliasKeys.erase(key);
    }
    changed(page);
    releasePage(id);
}

void ResidencyManager::markDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    if (end <= begin) return;
    auto& page = pages[id];
    if (page.dirtyRuns.empty()) ++dirtyPages;
    auto& runs = page.dirtyRuns;
    auto at = std::lower_bound(runs.begin(), runs.end(), Run{begin, begin}, [](const Run& left, const Run& right) { return left.second < right.first; });
    auto last = at;
    while (last != runs.end() && last->first <= end) {
        begin = std::min(begin, last->first);
        end = std::max(end, last->second);
        ++last;
    }
    at = runs.erase(at, last);
    runs.insert(at, Run{begin, end});
    page.dirty = static_cast<ResidencyMask>(page.dirty | RunMask(begin, end));
    publishPage(page.index);
}

void ResidencyManager::clearDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    auto& page = pages[id];
    if (page.dirtyRuns.empty() || end <= begin) return;
    std::vector<Run> kept;
    kept.reserve(page.dirtyRuns.size() + 1);
    for (const auto& run : page.dirtyRuns) {
        if (run.second <= begin || run.first >= end) {
            kept.push_back(run);
            continue;
        }
        if (run.first < begin) kept.push_back({run.first, begin});
        if (run.second > end) kept.push_back({end, run.second});
    }
    page.dirtyRuns = std::move(kept);
    page.dirty = 0;
    for (const auto& run : page.dirtyRuns) page.dirty = static_cast<ResidencyMask>(page.dirty | RunMask(run.first, run.second));
    publishPage(page.index);
    if (page.dirtyRuns.empty()) --dirtyPages;
}

void ResidencyManager::clearAllDirty(std::uint32_t id) {
    auto& page = pages[id];
    if (page.dirtyRuns.empty()) return;
    page.dirtyRuns.clear();
    page.dirty = 0;
    publishPage(page.index);
    --dirtyPages;
}

void ResidencyManager::appendDirty(std::uint32_t id, std::vector<ResidencyCopy>& copies) const {
    const auto& page = pages[id];
    for (const auto& [begin, end] : page.dirtyRuns) copies.push_back({page.index, page.slot, begin, end - begin});
}

void ResidencyManager::appendClean(std::uint32_t id, std::uint32_t begin, std::uint32_t end, std::vector<ResidencyCopy>& copies) const {
    const auto& page = pages[id];
    auto cursor = begin;
    for (const auto& [dirtyBegin, dirtyEnd] : page.dirtyRuns) {
        if (dirtyEnd <= cursor) continue;
        if (dirtyBegin >= end) break;
        if (dirtyBegin > cursor) copies.push_back({page.index, page.slot, cursor, dirtyBegin - cursor});
        cursor = std::max(cursor, dirtyEnd);
        if (cursor >= end) return;
    }
    if (cursor < end) copies.push_back({page.index, page.slot, cursor, end - cursor});
}

std::optional<MirrorSlot> ResidencyManager::takeFree() {
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        auto& record = chunks[chunk];
        if (!record.live || record.draining || record.freeSlots == 0) continue;
        for (std::uint32_t word = 0; word < record.freeBits.size(); ++word) {
            if (record.freeBits[word] == 0) continue;
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(record.freeBits[word]));
            record.freeBits[word] &= ~(std::uint64_t{1} << bit);
            --record.freeSlots;
            return MirrorSlot{chunk, word * WordBits + bit};
        }
    }
    return std::nullopt;
}

void ResidencyManager::freeSlot(MirrorSlot slot) {
    auto& record = chunks[slot.chunk];
    record.freeBits[slot.slot / WordBits] |= std::uint64_t{1} << (slot.slot % WordBits);
    ++record.freeSlots;
    if (record.draining && record.freeSlots == chunkSlots) releaseChunk(slot.chunk);
}

void ResidencyManager::retireReached() {
    while (!retiring.empty() && settled(retiring.front().token)) {
        const auto slot = retiring.front().slot;
        retiring.pop_front();
        freeSlot(slot);
    }
}

std::uint64_t ResidencyManager::budgetBytes() {
    if (budget != ResidencyMeasuredBudget) return budget;
    if (measuredEpoch != Epoch() || measuredEpoch == 0) {
        measured = device.Budget();
        measuredEpoch = Epoch();
    }
    return measured;
}

bool ResidencyManager::fullResidency() {
    return budgetBytes() >= device.GuestBytes();
}

std::uint32_t ResidencyManager::chunkLimit(bool growing) {
    const auto chunkBytes = std::uint64_t{chunkSlots} << ResidencyPageShift;
    auto bytes = budgetBytes();
    if (budget == ResidencyMeasuredBudget && growing) bytes = bytes > chunkBytes ? bytes - chunkBytes : 0;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes / chunkBytes, std::numeric_limits<std::uint32_t>::max()));
}

std::optional<std::uint32_t> ResidencyManager::allocateChunk() {
    if (liveChunks >= maxChunks && limitEpoch != Epoch()) {
        limitEpoch = Epoch();
        maxChunks = std::max(maxChunks, chunkLimit(true));
    }
    if (liveChunks >= maxChunks) return std::nullopt;
    std::uint32_t chunk = 0;
    while (chunk < chunks.size() && chunks[chunk].live) ++chunk;
    const auto address = device.AllocateChunk(chunk, chunkSlots);
    if (chunk == chunks.size()) chunks.emplace_back();
    auto& record = chunks[chunk];
    record.address = address;
    record.freeBits.assign((chunkSlots + WordBits - 1) / WordBits, ~std::uint64_t{0});
    if (chunkSlots % WordBits != 0) record.freeBits.back() = (std::uint64_t{1} << (chunkSlots % WordBits)) - 1;
    record.freeSlots = chunkSlots;
    record.live = true;
    record.draining = false;
    ++liveChunks;
    ++counts.chunksAllocated;
    return chunk;
}

void ResidencyManager::releaseChunk(std::uint32_t chunk) {
    device.ReleaseChunk(chunk);
    auto& record = chunks[chunk];
    record = Chunk{};
    --liveChunks;
    ++counts.chunksReleased;
}

std::optional<std::uint32_t> ResidencyManager::victim(std::uint64_t protectFirst, std::uint64_t protectLast) {
    for (auto id = oldest; id != None; id = pages[id].newer) {
        const auto& page = pages[id];
        if (page.index >= protectFirst && page.index <= protectLast) continue;
        if (touchFrame != 0 && page.touchFrame == touchFrame) continue;
        if (isStale(page.index)) continue;
        if (!settled(page.pin)) continue;
        return id;
    }
    return std::nullopt;
}

std::optional<MirrorSlot> ResidencyManager::acquireSlot(std::uint64_t protectFirst, std::uint64_t protectLast, WriteBacks& writeBacks) {
    retireReached();
    if (const auto slot = takeFree()) return slot;
    if (allocateChunk()) return takeFree();
    const auto id = victim(protectFirst, protectLast);
    if (!id) return std::nullopt;
    ++counts.evicted;
    leave(*id, writeBacks);
    retireReached();
    return takeFree();
}

void ResidencyManager::leave(std::uint32_t id, WriteBacks& writeBacks) {
    const auto& page = pages[id];
    if (page.guarded) writeBacks.unguard.push_back(page.index);
    if (!page.dirtyRuns.empty()) {
        appendDirty(id, writeBacks.copies);
        writeBacks.slots.push_back({page.slot, page.pin});
        writeBacks.pages.push_back(page.index);
    } else {
        retiring.push_back({page.slot, MergeTokens(page.pin, device.Newest())});
    }
    forget(id);
}

void ResidencyManager::recordWriteBacks(WriteBacks& writeBacks) {
    if (!writeBacks.unguard.empty()) device.Unguard(writeBacks.unguard);
    if (writeBacks.copies.empty()) {
        writeBacks = WriteBacks{};
        return;
    }
    const auto token = device.WriteBack(writeBacks.copies);
    for (const auto& copy : writeBacks.copies) counts.writtenBackBytes += copy.bytes;
    const auto newestToken = device.Newest();
    for (const auto& entry : writeBacks.slots) retiring.push_back({entry.slot, MergeTokens(MergeTokens(entry.token, token), newestToken)});
    for (const auto page : writeBacks.pages) addUnsettled(writingBack, page, token, UnsettledWriteBack);
    writeBacks = WriteBacks{};
}

void ResidencyManager::copyIn(std::span<const Fill> fills) {
    std::vector<ResidencyCopy> copies;
    for (const auto& fill : fills) appendClean(fill.id, fill.begin, fill.end, copies);
    if (copies.empty()) return;
    const auto generation = device.Generation();
    const auto copied = device.CopyIn(copies);
    for (const auto& copy : copies) counts.copiedInBytes += copy.bytes;
    for (const auto& fill : fills) {
        auto& page = pages[fill.id];
        page.pin = MergeTokens(page.pin, copied);
        if (fill.begin == 0 && fill.end == WholeEnd) page.generation = generation;
    }
}

void ResidencyManager::waitWriters(std::span<const Fill> fills) {
    ResidencyToken pending{};
    for (const auto& fill : fills) {
        const auto writer = writers.find(pages[fill.id].index);
        if (writer != writers.end() && !settled(writer->second)) pending = MergeTokens(pending, writer->second);
    }
    if (!TokenEmpty(pending)) waitFor(pending);
}

void ResidencyManager::refresh(std::span<const Fill> fills) {
    waitWriters(fills);
    refreshOrdered(fills);
}

void ResidencyManager::refreshOrdered(std::span<const Fill> fills) {
    std::vector<Fill> needed;
    for (const auto& fill : fills) {
        const auto& page = pages[fill.id];
        if ((RunMask(fill.begin, fill.end) & page.dirty) != 0) {
            const bool overlaps = std::any_of(page.dirtyRuns.begin(), page.dirtyRuns.end(), [&](const Run& run) { return run.first < fill.end && run.second > fill.begin; });
            if (overlaps) ++counts.conflicts;
        }
        needed.push_back(fill);
        ++counts.refreshed;
    }
    copyIn(needed);
}

void ResidencyManager::applyDrops() {
    std::vector<ResidencyRange> queue;
    {
        const std::lock_guard lock(queueMutex);
        queue.swap(dropQueue);
        queued.store(!cpuQueue.empty() || !gpuQueue.empty(), std::memory_order_release);
    }
    for (const auto& range : queue) {
        for (const auto id : residentIn(range.address, range.bytes)) {
            const auto page = pages[id].index;
            const auto import = device.Import(page);
            if (import.whole && import.serial == pages[id].importSerial) continue;
            ++counts.dropped;
            if (pages[id].accessGuarded) {
                device.UnguardAccess(page);
                accessGuards.erase(page);
            }
            retiring.push_back({pages[id].slot, MergeTokens(pages[id].pin, device.Newest())});
            eraseUnsettled(writingBack, page, UnsettledWriteBack);
            forget(id);
        }
        const auto span = PagesOf(range.address, range.bytes);
        if (span.empty) continue;
        std::vector<std::uint64_t> candidates;
        {
            const std::shared_lock lock(ownedMutex);
            for (const auto& [index, record] : owned) {
                if (index >= span.first && index <= span.last) candidates.push_back(index);
            }
        }
        std::vector<std::uint64_t> dropped;
        for (const auto index : candidates) {
            const auto import = device.Import(index);
            const auto id = find(index);
            if (import.whole && id && import.serial == pages[*id].importSerial) continue;
            dropped.push_back(index);
        }
        if (!dropped.empty()) {
            const std::unique_lock lock(ownedMutex);
            for (const auto index : dropped) {
                if (staleOwned.erase(index) != 0) --staleCount;
                owned.erase(index);
                publishStale(index, 0);
            }
            ownershipSerial.fetch_add(1, std::memory_order_acq_rel);
        }
        eraseUnsettledIf(writingBack, UnsettledWriteBack, [&](const auto& entry) { return entry.first >= span.first && entry.first <= span.last; });
        std::erase_if(hot, [&](const auto& entry) { return entry.first >= span.first && entry.first <= span.last; });
        for (auto it = accessGuards.begin(); it != accessGuards.end();) {
            if (*it < span.first || *it > span.last) {
                ++it;
                continue;
            }
            device.UnguardAccess(*it);
            it = accessGuards.erase(it);
        }
    }
}

void ResidencyManager::QueueGpuWritten(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    const std::lock_guard lock(queueMutex);
    gpuQueue.push_back({address, bytes});
    queued.store(true, std::memory_order_release);
}

void ResidencyManager::applyQueued() {
    applyDrops();
    std::vector<ResidencyRange> gpuWritten;
    {
        const std::lock_guard lock(queueMutex);
        gpuWritten.swap(gpuQueue);
    }
    for (const auto& range : gpuWritten) applyRecorded(range.address, range.bytes);
    std::vector<std::uint64_t> written;
    {
        const std::lock_guard lock(queueMutex);
        written.swap(cpuQueue);
        queued.store(!dropQueue.empty() || !gpuQueue.empty(), std::memory_order_release);
    }
    if (!written.empty()) applyFaults(std::move(written));
}

void ResidencyManager::applyRecorded(std::uint64_t address, std::uint64_t bytes) {
    const auto fills = withAliases(address, bytes);
    if (fills.empty()) return;
    counts.recordRefreshed += fills.size();
    refreshOrdered(fills);
}

void ResidencyManager::applyFaults(std::vector<std::uint64_t> written) {
    std::sort(written.begin(), written.end());
    written.erase(std::unique(written.begin(), written.end()), written.end());
    const auto window = hotEpochs >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << hotEpochs) - 1;
    const auto now = Epoch();
    std::vector<Fill> fills;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> partial;
    for (const auto index : written) {
        const auto id = find(index);
        if (!id) continue;
        auto& page = pages[*id];
        ++counts.faults;
        if (hotResident.contains(index)) ++counts.hotRefreshed;
        const auto distance = now - page.faultEpoch;
        page.faultHistory = (distance >= 64 ? 0 : page.faultHistory << distance) | 1u;
        page.faultEpoch = now;
        if (static_cast<std::uint32_t>(std::popcount(page.faultHistory & window)) >= hotFaults && !hotResident.contains(index)) {
            ++counts.hot;
            markHot(index, now);
            page.guarded = false;
        } else if (hotResident.contains(index)) {
            markHot(index, now);
            page.guarded = false;
        } else if (page.guarded && !device.Guard(index)) {
            ++counts.unguardable;
            page.guarded = false;
            markHot(index, now);
        }
        const auto writes = device.PageWrites(index, page.generation);
        if (writes.other) {
            fills.push_back({*id});
            continue;
        }
        std::uint32_t bits = writes.cpu;
        while (bits != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
            fills.push_back({*id, first << ResidencySubpageShift, (first + run) << ResidencySubpageShift});
            bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
        }
        partial.emplace_back(*id, writes.generation);
    }
    refresh(fills);
    for (const auto& [id, generation] : partial) pages[id].generation = std::max(pages[id].generation, generation);
}

void ResidencyManager::coolHot(std::uint64_t first, std::span<const std::uint8_t> written) {
    const auto generation = device.Generation();
    const auto now = Epoch();
    for (std::size_t k = 0; k < written.size(); ++k) {
        const auto entry = hot.find(first + k);
        if (entry == hot.end()) continue;
        if (written[k] != 0) {
            entry->second = {generation, now};
            continue;
        }
        if (now - entry->second.epoch < hotEpochs) continue;
        hot.erase(entry);
        ++counts.cooled;
    }
}

bool ResidencyManager::Advance() {
    epoch.fetch_add(1, std::memory_order_relaxed);
    return queued.load(std::memory_order_acquire);
}

void ResidencyManager::Apply() {
    applyQueued();
    retireReached();
    settleAccessGuards();
    admitEager();
}

void ResidencyManager::admitEager() {
    constexpr std::uint64_t PagesPerCall = 512;
    if (!fullResidency()) return;
    std::uint64_t generation = importsGeneration;
    auto imports = device.WholeImports(generation);
    if (generation != importsGeneration) {
        importsGeneration = generation;
        pendingImports = std::move(imports);
    }
    std::uint64_t quota = PagesPerCall;
    while (!pendingImports.empty() && quota != 0) {
        const auto range = pendingImports.back();
        const auto first = range.address >> ResidencyPageShift;
        const auto last = (range.address + range.bytes - 1) >> ResidencyPageShift;
        auto& entry = pendingImports.back();
        const auto next = admitPages(first, last, &quota, false, true);
        if (next <= last) {
            entry.address = next << ResidencyPageShift;
            entry.bytes = (last + 1 - next) << ResidencyPageShift;
            break;
        }
        pendingImports.pop_back();
    }
}

void ResidencyManager::Prepare() {
    Advance();
    Apply();
}

void ResidencyManager::SetBudget(std::uint64_t bytes) {
    budget = bytes;
    measuredEpoch = 0;
    ApplyBudget();
}

void ResidencyManager::ApplyBudget() {
    applyQueued();
    measuredEpoch = 0;
    const auto shrunk = chunkLimit(false);
    maxChunks = shrunk < liveChunks ? shrunk : std::max(liveChunks, chunkLimit(true));
    limitEpoch = Epoch();
    std::uint32_t ordinal = 0;
    for (auto& record : chunks) {
        if (!record.live) continue;
        record.draining = ordinal >= maxChunks;
        ++ordinal;
    }
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunks[chunk].live && chunks[chunk].draining && chunks[chunk].freeSlots == chunkSlots) releaseChunk(chunk);
    }
    evictDraining();
}

void ResidencyManager::evictDraining() {
    if (liveChunks <= maxChunks) return;
    std::vector<std::uint64_t> stale;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (chunks[pages[id].slot.chunk].draining && isStale(pages[id].index)) stale.push_back(pages[id].index);
    }
    retileStale(stale);
    std::vector<std::uint32_t> leaving;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (chunks[pages[id].slot.chunk].draining && settled(pages[id].pin)) leaving.push_back(id);
    }
    WriteBacks writeBacks;
    for (const auto id : leaving) {
        ++counts.evicted;
        leave(id, writeBacks);
    }
    recordWriteBacks(writeBacks);
    retireReached();
}

void ResidencyManager::Pin(std::uint64_t address, std::uint64_t bytes, ResidencyAccess access, const ResidencyToken& token) {
    for (const auto id : residentIn(address, bytes)) {
        auto& page = pages[id];
        page.pin = MergeTokens(page.pin, token);
        if (access == ResidencyAccess::Write) {
            addUnsettled(writers, page.index, token, UnsettledWriter);
        }
    }
}

void ResidencyManager::Use(std::uint64_t address, std::uint64_t bytes, ResidencyAccess access) {
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    const auto count = span.last - span.first + 1;
    const auto present = residentIn(address, bytes);
    std::vector<std::uint64_t> generations(count, ResidencyUnchecked);
    bool check = !present.empty();
    for (const auto id : present) generations[pages[id].index - span.first] = pages[id].generation;
    if (!hot.empty()) {
        for (std::uint64_t k = 0; k < count; ++k) {
            const auto entry = hot.find(span.first + k);
            if (entry == hot.end()) continue;
            generations[k] = entry->second.generation;
            check = true;
        }
    }
    if (check) {
        std::vector<std::uint8_t> stale(count, 0);
        device.ChangedPages(span.first, generations, stale);
        std::vector<Fill> fills;
        for (const auto id : present) {
            unlink(id);
            linkNewest(id);
            if (stale[pages[id].index - span.first] != 0) fills.push_back({id});
            stale[pages[id].index - span.first] = 0;
        }
        refresh(fills);
        if (!hot.empty()) coolHot(span.first, stale);
    }
    if (present.size() < count) {
        std::vector<std::uint32_t> aliases;
        for (const auto& fill : withAliases(address, bytes)) {
            if ((pages[fill.id].index < span.first || pages[fill.id].index > span.last) && !pages[fill.id].dirtyRuns.empty()) aliases.push_back(fill.id);
        }
        writeBackDirty(aliases);
        admit(span.first, span.last);
    }
    Pin(address, bytes, access, device.Current());
}

void ResidencyManager::AdmitTouched(std::span<const std::uint64_t> touched) {
    applyQueued();
    ++touchFrame;
    std::vector<std::uint64_t> sorted(touched.begin(), touched.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::vector<std::uint64_t> missing;
    for (const auto page : sorted) {
        const auto id = find(page);
        if (!id) {
            missing.push_back(page);
            continue;
        }
        pages[*id].touchFrame = touchFrame;
        unlink(*id);
        linkNewest(*id);
    }
    std::uint64_t quota = TouchPagesPerFrame;
    for (std::size_t first = 0; first < missing.size();) {
        auto last = first;
        while (last + 1 < missing.size() && missing[last + 1] == missing[last] + 1) ++last;
        if (quota == 0) {
            counts.touchDeferred += last - first + 1;
            first = last + 1;
            continue;
        }
        const auto before = counts.admitted;
        const auto next = admitPages(missing[first], missing[last], &quota);
        const auto admitted = counts.admitted - before;
        counts.touchAdmitted += admitted;
        if (next <= missing[last]) counts.touchDeferred += missing[last] - next + 1;
        for (auto index = first; index <= last; ++index) {
            if (const auto id = find(missing[index])) pages[*id].touchFrame = touchFrame;
        }
        first = last + 1;
    }
}

void ResidencyManager::admit(std::uint64_t first, std::uint64_t last) {
    admitPages(first, last, nullptr);
}

std::uint64_t ResidencyManager::admitPages(std::uint64_t first, std::uint64_t last, std::uint64_t* quota, bool evict, bool sharedOnly) {
    WriteBacks writeBacks;
    std::vector<Fill> fills;
    bool exhausted = false;
    auto page = first;
    for (; page <= last; ++page) {
        if (quota != nullptr && *quota == 0) break;
        if (find(page) || hot.contains(page)) continue;
        const auto import = device.Import(page);
        if (!import.whole) {
            ++counts.inadmissible;
            continue;
        }
        if (sharedOnly && import.section == 0) continue;
        const auto key = std::pair{import.section, import.offset};
        if (import.aliased && canonical.contains(key)) {
            ++counts.aliasSkipped;
            continue;
        }
        std::optional<MirrorSlot> slot;
        if (!exhausted) {
            if (evict) {
                slot = acquireSlot(first, last, writeBacks);
            } else {
                retireReached();
                slot = takeFree();
                if (!slot && allocateChunk()) slot = takeFree();
            }
        }
        if (!slot) {
            ++counts.deferred;
            exhausted = exhausted || !evict || !victim(first, last);
            if (quota != nullptr) break;
            continue;
        }
        recordWriteBacks(writeBacks);
        bool guarded = false;
        if (import.cpuWritable) guarded = device.Guard(page);
        const auto id = newPage(page, *slot, import, guarded);
        if (import.cpuWritable && !guarded) {
            ++counts.unguardable;
            markHot(page, Epoch());
        }
        fills.push_back({id});
        if (import.aliased) {
            canonical.emplace(key, page);
            aliasKeys.emplace(page, key);
        }
        ++counts.admitted;
        if (quota != nullptr) --*quota;
    }
    recordWriteBacks(writeBacks);
    copyIn(fills);
    return page;
}

void ResidencyManager::demote(std::span<const std::uint32_t> ids) {
    std::vector<std::uint64_t> stale;
    for (const auto id : ids) {
        if (isStale(pages[id].index)) stale.push_back(pages[id].index);
    }
    retileStale(stale);
    ResidencyToken pending{};
    for (const auto id : ids) {
        const auto writer = writers.find(pages[id].index);
        if (writer != writers.end() && !settled(writer->second)) pending = MergeTokens(pending, writer->second);
    }
    if (!TokenEmpty(pending)) waitFor(pending);
    WriteBacks writeBacks;
    for (const auto id : ids) {
        ++counts.demoted;
        leave(id, writeBacks);
    }
    recordWriteBacks(writeBacks);
}

void ResidencyManager::Demote(std::uint64_t address, std::uint64_t bytes) {
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (const auto& fill : withAliases(address, bytes)) ids.push_back(fill.id);
    demote(ids);
}

void ResidencyManager::DemoteWritable() {
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].gpuWritable) ids.push_back(id);
    }
    demote(ids);
}

void ResidencyManager::DropRange(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    const std::lock_guard lock(queueMutex);
    dropQueue.push_back({address, bytes});
    queued.store(true, std::memory_order_release);
}

void ResidencyManager::NoteImportWritten(std::uint64_t address, std::uint64_t bytes) {
    applyQueued();
    const auto fills = withAliases(address, bytes);
    waitWriters(fills);
    for (const auto& fill : fills) clearDirty(fill.id, fill.begin, fill.end);
    refresh(fills);
}

void ResidencyManager::RefreshImportWritten(std::uint64_t address, std::uint64_t bytes) {
    applyQueued();
    const auto fills = withAliases(address, bytes);
    for (const auto& fill : fills) clearDirty(fill.id, fill.begin, fill.end);
    refreshOrdered(fills);
}

void ResidencyManager::QueueCpuWritten(std::span<const std::uint64_t> written) {
    if (written.empty()) return;
    const std::lock_guard lock(queueMutex);
    cpuQueue.insert(cpuQueue.end(), written.begin(), written.end());
    queued.store(true, std::memory_order_release);
}

void ResidencyManager::NoteGpuWritten(std::uint64_t address, std::uint64_t bytes, bool written) {
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    std::vector<std::uint32_t> unguardable;
    std::vector<std::uint32_t> aliases;
    std::vector<Fill> stored;
    for (const auto& fill : withAliases(address, bytes)) {
        auto& page = pages[fill.id];
        if (page.index < span.first || page.index > span.last) {
            aliases.push_back(fill.id);
            continue;
        }
        const auto mask = RunMask(fill.begin, fill.end);
        const auto adding = static_cast<ResidencyMask>(mask & ~page.accessMask);
        if (!page.shared || aliasKeys.contains(page.index)) {
            unguardable.push_back(fill.id);
            stored.push_back(fill);
            continue;
        }
        if (adding != 0) {
            if (!device.GuardAccess(page.index, adding)) {
                ++counts.accessUnguardable;
                unguardable.push_back(fill.id);
                stored.push_back(fill);
                continue;
            }
            page.accessMask = static_cast<ResidencyMask>(page.accessMask | adding);
            page.accessGuarded = true;
            accessGuards.insert(page.index);
        }
        markDirty(fill.id, fill.begin, fill.end);
    }
    if (!aliases.empty()) {
        counts.aliasDemoted += aliases.size();
        demote(aliases);
    }
    if (unguardable.empty()) return;
    if (!written) {
        demote(unguardable);
        return;
    }
    for (const auto& fill : stored) markDirty(fill.id, fill.begin, fill.end);
    writeBackDirty(unguardable);
    demote(unguardable);
}

void ResidencyManager::FlushForCpuRead(std::uint64_t address, std::uint64_t bytes) {
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    const bool retiled = retileStale(staleIn(address, bytes));
    const auto inSpan = [&](std::uint64_t page) { return page >= span.first && page <= span.last; };
    ResidencyToken pending = retiled ? device.Current() : ResidencyToken{};
    for (const auto& [page, token] : writers) {
        if (inSpan(page) && !settled(token)) pending = MergeTokens(pending, token);
    }
    if (!TokenEmpty(pending)) waitFor(pending);
    std::vector<ResidencyCopy> copies;
    std::vector<std::uint32_t> written;
    for (const auto id : residentIn(address, bytes)) {
        if (pages[id].dirtyRuns.empty()) continue;
        appendDirty(id, copies);
        written.push_back(id);
    }
    ResidencyToken wait{};
    if (!copies.empty()) {
        for (const auto id : written) clearAllDirty(id);
        wait = device.WriteBack(copies);
        ++counts.cpuWriteBacks;
        for (const auto& copy : copies) counts.writtenBackBytes += copy.bytes;
        for (const auto id : written) pages[id].pin = MergeTokens(pages[id].pin, wait);
    }
    for (const auto& [page, token] : writingBack) {
        if (inSpan(page)) wait = MergeTokens(wait, token);
    }
    if (!settled(wait)) waitFor(wait);
    eraseUnsettledIf(writingBack, UnsettledWriteBack, [&](const auto& entry) { return inSpan(entry.first); });
    eraseUnsettledIf(writers, UnsettledWriter, [&](const auto& entry) { return inSpan(entry.first) && settled(entry.second); });
    for (auto page = span.first; page <= span.last; ++page) {
        const auto id = find(page);
        if (id && !pages[*id].dirtyRuns.empty()) continue;
        if (isStale(page)) continue;
        if (!accessGuards.contains(page) && !(id && pages[*id].accessGuarded) && span.last - span.first >= 16) continue;
        device.UnguardAccess(page);
        accessGuards.erase(page);
        if (!id) continue;
        pages[*id].accessGuarded = false;
        pages[*id].accessMask = 0;
        rearm(*id);
    }
}

void ResidencyManager::rearm(std::uint32_t id) {
    auto& page = pages[id];
    if (!page.cpuWritable || page.guarded) return;
    if (hotResident.contains(page.index)) return;
    if (device.Guard(page.index)) {
        page.guarded = true;
        return;
    }
    ++counts.unguardable;
    markHot(page.index, Epoch());
}

void ResidencyManager::writeBackDirty(std::span<const std::uint32_t> ids) {
    std::vector<ResidencyCopy> copies;
    std::vector<std::uint32_t> written;
    for (const auto id : ids) {
        if (pages[id].dirtyRuns.empty()) continue;
        appendDirty(id, copies);
        written.push_back(id);
    }
    if (copies.empty()) return;
    for (const auto id : written) clearAllDirty(id);
    const auto token = device.WriteBack(copies);
    for (const auto& copy : copies) counts.writtenBackBytes += copy.bytes;
    for (const auto id : written) {
        auto& page = pages[id];
        page.pin = MergeTokens(page.pin, token);
        addUnsettled(writingBack, page.index, token, UnsettledWriteBack);
    }
}

void ResidencyManager::PrepareImportWrite(std::uint64_t address, std::uint64_t bytes) {
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (const auto& fill : withAliases(address, bytes)) ids.push_back(fill.id);
    if (ids.empty()) return;
    writeBackDirty(ids);
}

void ResidencyManager::settleAccessGuards() {
    std::vector<std::uint64_t> settledPages;
    for (const auto& [page, token] : writingBack) {
        if (settled(token)) settledPages.push_back(page);
    }
    for (const auto page : settledPages) {
        eraseUnsettled(writingBack, page, UnsettledWriteBack);
        if (!accessGuards.contains(page)) continue;
        const auto id = find(page);
        if (id && !pages[*id].dirtyRuns.empty()) continue;
        device.UnguardAccess(page);
        accessGuards.erase(page);
        if (!id) continue;
        pages[*id].accessGuarded = false;
        pages[*id].accessMask = 0;
        rearm(*id);
    }
}

void ResidencyManager::markHot(std::uint64_t page, std::uint64_t at) {
    const auto [entry, inserted] = hotResident.insert_or_assign(page, at);
    static_cast<void>(entry);
    hotCooling.emplace_back(at, page);
    if (!inserted) return;
    auto next = hotRuns.upper_bound(page);
    auto first = page;
    auto last = page;
    if (next != hotRuns.begin()) {
        const auto previous = std::prev(next);
        if (previous->second + 1 == page) {
            first = previous->first;
            hotRuns.erase(previous);
        }
    }
    if (next != hotRuns.end() && next->first == page + 1) {
        last = next->second;
        hotRuns.erase(next);
    }
    hotRuns.emplace(first, last);
}

void ResidencyManager::unmarkHot(std::uint64_t page) {
    if (hotResident.erase(page) == 0) return;
    auto run = std::prev(hotRuns.upper_bound(page));
    const auto first = run->first;
    const auto last = run->second;
    hotRuns.erase(run);
    if (first < page) hotRuns.emplace(first, page - 1);
    if (page < last) hotRuns.emplace(page + 1, last);
}

void ResidencyManager::RefreshHot() {
    applyQueued();
    if (hotResident.empty()) return;
    const auto now = Epoch();
    if (sweptEpoch != now) {
        sweptEpoch = now;
        ++counts.hotSweeps;
        for (const auto& [first, last] : hotRuns) device.CollectWrites(first, last - first + 1);
        applyQueued();
    }
    std::vector<std::uint32_t> cooled;
    while (!hotCooling.empty() && now - hotCooling.front().first >= hotEpochs) {
        const auto [at, index] = hotCooling.front();
        hotCooling.pop_front();
        const auto entry = hotResident.find(index);
        if (entry == hotResident.end() || entry->second != at) continue;
        unmarkHot(index);
        ++counts.cooled;
        if (const auto id = find(index)) cooled.push_back(*id);
    }
    for (const auto id : cooled) rearm(id);
}

void ResidencyManager::Retire() {
    applyQueued();
    retireReached();
    settleAccessGuards();
    admitEager();
    evictDraining();
    eraseUnsettledIf(writers, UnsettledWriter, [&](const auto& entry) { return settled(entry.second); });
    eraseUnsettledIf(writingBack, UnsettledWriteBack, [&](const auto& entry) { return settled(entry.second); });
}

std::optional<MirrorSlot> ResidencyManager::Slot(std::uint64_t page) {
    applyDrops();
    const auto id = find(page);
    if (!id) return std::nullopt;
    return pages[*id].slot;
}

std::optional<MirrorSlot> ResidencyManager::WriteSlot(std::uint64_t page) {
    applyDrops();
    const auto id = find(page);
    if (!id || !pages[*id].shared) return std::nullopt;
    return pages[*id].slot;
}

std::optional<std::uint64_t> ResidencyManager::Translate(std::uint64_t page) {
    const auto slot = Slot(page);
    if (!slot) return std::nullopt;
    return chunks[slot->chunk].address + (std::uint64_t{slot->slot} << ResidencyPageShift);
}

ResidencyMask ResidencyManager::Dirty(std::uint64_t page) {
    applyDrops();
    const auto id = find(page);
    return id ? pages[*id].dirty : ResidencyMask{0};
}

bool ResidencyManager::AnyResident(std::uint64_t address, std::uint64_t bytes) {
    const auto span = PagesOf(address, bytes);
    if (span.empty) return false;
    const std::lock_guard lock(residentMutex);
    if (span.last - span.first >= resident.size()) {
        return std::any_of(resident.begin(), resident.end(), [&](const auto& entry) { return entry.first >= span.first && entry.first <= span.last; });
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (resident.contains(page)) return true;
    }
    return false;
}

std::vector<ResidencyRange> ResidencyManager::ResidentRanges() {
    applyDrops();
    if (rangesSerial == serial) return ranges;
    std::vector<std::uint64_t> indices;
    indices.reserve(resident.size());
    for (const auto& entry : resident) indices.push_back(entry.first);
    std::sort(indices.begin(), indices.end());
    ranges.clear();
    for (const auto page : indices) {
        const auto address = page << ResidencyPageShift;
        if (!ranges.empty() && ranges.back().address + ranges.back().bytes == address) ranges.back().bytes += ResidencyPageBytes;
        else ranges.push_back({address, ResidencyPageBytes});
    }
    rangesSerial = serial;
    return ranges;
}

std::vector<std::uint64_t> ResidencyManager::TakeChanges() {
    applyDrops();
    std::vector<std::uint64_t> taken(changes.begin(), changes.end());
    changes.clear();
    std::sort(taken.begin(), taken.end());
    return taken;
}

std::vector<std::uint32_t> ResidencyManager::Chunks() const {
    std::vector<std::uint32_t> live;
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunks[chunk].live) live.push_back(chunk);
    }
    return live;
}

std::uint64_t ResidencyManager::ChunkAddress(std::uint32_t chunk) const {
    if (chunk >= chunks.size() || !chunks[chunk].live) throw std::out_of_range("residency: chunk " + std::to_string(chunk) + " is not live");
    return chunks[chunk].address;
}

ResidencyCounts ResidencyManager::Counts() const {
    auto result = counts;
    result.chunks = liveChunks;
    result.resident = static_cast<std::uint32_t>(resident.size());
    result.dirty = dirtyPages.load(std::memory_order_relaxed);
    {
        const std::shared_lock lock(ownedMutex);
        result.ownedPages = static_cast<std::uint32_t>(owned.size());
        result.stalePages = static_cast<std::uint32_t>(staleOwned.size());
    }
    result.accessGuarded = static_cast<std::uint32_t>(accessGuards.size());
    result.hotResident = static_cast<std::uint32_t>(hotResident.size());
    result.retiring = static_cast<std::uint32_t>(retiring.size());
    result.free = 0;
    for (const auto& record : chunks) {
        if (record.live) result.free += record.freeSlots;
    }
    return result;
}

}
