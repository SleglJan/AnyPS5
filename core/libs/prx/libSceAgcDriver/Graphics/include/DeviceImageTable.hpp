#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEIMAGETABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEIMAGETABLE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageKeys.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics {

class Buffer;
class Recorder;
class Sampler;
class StorageTexture;
class Texture;

class DeviceImageTable {
public:
    static constexpr std::uint32_t RingCapacity = 1024u;

    explicit DeviceImageTable(const Context& context);
    ~DeviceImageTable();
    DeviceImageTable(const DeviceImageTable&) = delete;
    DeviceImageTable& operator=(const DeviceImageTable&) = delete;

    std::uint64_t HeaderAddress() const;
    void Expect(std::span<const ShaderRecompiler::DeviceImageTableAbi::Expectation> images, bool complete);
    void BeforeDispatch(const Context& context, Recorder& recorder);
    void AfterDispatch(const Context& context, Recorder& recorder, std::uint64_t storeRecord = 0);
    std::vector<std::shared_ptr<StorageTexture>> DispatchStorage() const;
    std::vector<std::shared_ptr<Texture>> DispatchTextures() const;

    struct Rejection {
        std::array<std::uint32_t, 8> words{};
        std::uint32_t reason = 0;
        std::uint32_t pc = 0;
        std::string message;
    };
    std::optional<Rejection> FindRejection(std::uint32_t index) const;

    struct Counts {
        std::uint64_t misses = 0;
        std::uint64_t missWaits = 0;
        std::uint64_t missWaitNanoseconds = 0;
        std::uint64_t distinct = 0;
        std::uint64_t expected = 0;
        std::uint64_t dropped = 0;
        std::uint64_t inserted = 0;
        std::uint64_t rejected = 0;
        std::uint64_t refreshed = 0;
        std::uint64_t replaced = 0;
        std::uint64_t clears = 0;
        std::uint64_t evicted = 0;
        std::uint64_t memory = 0;
        std::uint32_t entries = 0;
        std::uint32_t buckets = 0;
    };
    Counts Totals() const;
    static std::uint64_t MemoryEntries();

private:
    struct ValidationState {
        std::uint64_t epoch = 0;
        std::uint64_t unwatch = 0;
        std::uint64_t depthSurfaces = 0;
        std::uint64_t pending = 0;
        std::uint64_t ownWrites = 0;
        bool operator==(const ValidationState&) const = default;
    };
    struct Entry {
        std::shared_ptr<Texture> texture;
        std::shared_ptr<Sampler> sampler;
        std::uint64_t address = 0;
        std::uint64_t guestBytes = 0;
        std::uint64_t generation = 0;
        bool snapshot = false;
        bool perUse = false;
        bool parked = false;
        std::shared_ptr<StorageTexture> storage;
        bool written = false;
        std::uint64_t used = 0;
        ValidationState checked;
        bool memory = false;
        bool overwritten = false;
    };
    static ValidationState currentState();

    void resolve(const Context& build, Recorder& recorder);
    void refresh(const Context& build, Recorder& recorder, const std::vector<DeviceImageKeys::Key>* scope);
    void record(Recorder& recorder);
    DeviceImageValue lookup(const Context& build, const DeviceImageKeys::Key& key, std::uint32_t pc, Entry& entry, bool perUse, bool report);
    void refreshPerUse(const Context& build, Recorder& recorder, const std::vector<DeviceImageKeys::Key>& resolved);
    void evict(const Context& build, Recorder& recorder);
    std::uint32_t reject(const DeviceImageKeys::Key& key, std::uint32_t pc, std::uint32_t reason, const std::string& message, bool report = true);
    void insert(const DeviceImageKeys::Key& key, const DeviceImageValue& value, Recorder& recorder);
    void retire(const Entry& entry, Recorder& recorder);
    void drain(const std::shared_ptr<Buffer>& ring);
    void report();
    std::string rejectionText(const DeviceImageKeys::Key& key, std::uint32_t fault) const;

    Context context;
    mutable std::recursive_mutex mutex;
    DeviceImageKeys keys;
    std::map<DeviceImageKeys::Key, Entry> entries;
    std::shared_ptr<Buffer> header;
    std::shared_ptr<Buffer> table;
    std::uint32_t tableBuckets = 0;
    std::shared_ptr<Buffer> ring;
    std::vector<std::shared_ptr<Buffer>> freeRings;
    std::array<std::uint64_t, MaxRecorderStreams> ringBatch{};
    std::vector<DeviceImageMiss> pending;
    std::vector<DeviceImageMiss> expected;
    std::vector<DeviceImageMiss> candidates;
    bool incomplete = false;
    std::optional<std::array<std::uint64_t, MaxRecorderStreams>> unnamedBatch;
    std::vector<DeviceImageKeys::Key> current;
    std::vector<DeviceImageKeys::Key> perUseKeys;
    std::uint64_t evictionCheck = 0;
    struct DispatchStorageUse {
        DeviceImageKeys::Key key{};
        std::shared_ptr<StorageTexture> image;
        bool written = false;
    };
    std::vector<DispatchStorageUse> dispatchStorage;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> dispatchMemoryWrites;
    std::vector<std::shared_ptr<Texture>> dispatchTextures;
    std::vector<Rejection> rejections;
    std::map<std::tuple<DeviceImageKeys::Key, std::uint32_t, std::string>, std::uint32_t> rejectionIndex;
    Counts totals;
    Counts reported;
    std::uint64_t reportedPresents = 0;
    std::uint64_t reportedWaitReads = 0;
    std::uint64_t reportedWaitNanoseconds = 0;
    long long lastReport = 0;
};

bool UsesDeviceImageTable(const ShaderRecompiler::RecompileResult& program);

}

#endif
