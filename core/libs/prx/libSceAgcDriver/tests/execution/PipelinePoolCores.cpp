#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Spin(std::uint64_t rounds) {
    std::uint64_t value = 0x9e3779b97f4a7c15ull;
    for (std::uint64_t round = 0; round < rounds; ++round) value = value * 6364136223846793005ull + 1442695040888963407ull;
    return value;
}

std::atomic<std::uint64_t> sink{0};

double GuestThreadsMilliseconds(unsigned threads, std::uint64_t rounds) {
    std::vector<double> spent(threads);
    std::vector<std::thread> running;
    for (unsigned index = 0; index < threads; ++index) {
        running.emplace_back([&, index] {
            const auto start = std::chrono::steady_clock::now();
            sink.fetch_add(Spin(rounds), std::memory_order_relaxed);
            spent[index] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        });
    }
    for (auto& thread : running) thread.join();
    return *std::max_element(spent.begin(), spent.end());
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void CheckPoolRunsTogether() {
    const auto tasks = std::min(std::max(2u, std::thread::hardware_concurrency()), 8u);
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrived = 0;
    unsigned met = 0;
    unsigned finished = 0;
    for (unsigned task = 0; task < tasks; ++task) {
        AgcDriver::DriverDetail::SubmitLibraryPrebuild([&] {
            std::unique_lock lock(mutex);
            ++arrived;
            changed.notify_all();
            if (changed.wait_for(lock, std::chrono::seconds(2), [&] { return arrived == tasks; })) ++met;
            ++finished;
            changed.notify_all();
        });
    }
    std::unique_lock lock(mutex);
    Require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return finished == tasks; }), "the pipeline pool did not finish " + std::to_string(tasks) + " tasks");
    Require(met == tasks, std::to_string(tasks) + " pipeline pool tasks submitted together did not all run at the same time (" + std::to_string(met) + " met)");
}

void CheckGuestThreadsKeepTheirCores() {
    const auto cores = std::max(2u, std::thread::hardware_concurrency());
    const auto guests = std::max(1u, cores / 2u);
    std::uint64_t rounds = 1u << 20u;
    while (GuestThreadsMilliseconds(1, rounds) < 40.0) rounds *= 2;
    std::vector<double> idle;
    for (int sample = 0; sample < 3; ++sample) idle.push_back(GuestThreadsMilliseconds(guests, rounds));

    const auto busy = 2u * cores;
    std::atomic<bool> stop{false};
    std::atomic<unsigned> spinning{0};
    std::atomic<unsigned> done{0};
    const auto spinner = [&] {
        spinning.fetch_add(1);
        std::uint64_t value = 1;
        while (!stop.load(std::memory_order_relaxed)) value = Spin(value | 1u << 12u) & 0xffffu;
        sink.fetch_add(value, std::memory_order_relaxed);
        done.fetch_add(1);
    };
    for (unsigned task = 0; task < busy; ++task) {
        AgcDriver::DriverDetail::SubmitLibraryPrebuild(spinner);
        AgcDriver::DriverDetail::SubmitRegistrationPreparation(spinner);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    const auto poolThreads = cores + std::max(2u, cores / 2u);
    while (spinning.load() < poolThreads && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto started = spinning.load();
    std::vector<double> loaded;
    for (int sample = 0; sample < 3; ++sample) loaded.push_back(GuestThreadsMilliseconds(guests, rounds));
    stop.store(true);
    while (done.load() < 2u * busy && std::chrono::steady_clock::now() < deadline + std::chrono::seconds(20)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Require(started >= poolThreads, "the shader compile pools ran " + std::to_string(started) + " workers at once, expected " + std::to_string(poolThreads));
    Require(done.load() == 2u * busy, "the shader compile pools did not drain after the busy tasks stopped");
    const auto before = Median(idle);
    const auto after = Median(loaded);
    std::printf("%u guest threads on %u cores: %.1f ms with the compile pools idle, %.1f ms with %u compile workers busy\n", guests, cores, before, after, started);
    Require(after < 1.5 * before, std::to_string(guests) + " guest threads took " + std::to_string(after) + " ms against " + std::to_string(before) + " ms while the shader compile pools were busy: the pools take the cores guest threads need");
}

}

int main() {
    try {
        CheckPoolRunsTogether();
        CheckGuestThreadsKeepTheirCores();
        AgcDriver::DriverDetail::CloseRegistrationPreparation();
        std::puts("pipeline pool tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
