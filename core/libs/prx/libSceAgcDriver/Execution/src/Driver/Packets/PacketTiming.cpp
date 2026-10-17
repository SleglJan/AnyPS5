#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/PacketTiming.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"

namespace AgcDriver::DriverDetail {

PacketTimer::~PacketTimer() {
    if (!enabled) return;
    const auto now = std::chrono::steady_clock::now();
    auto& entry = profile.byOpcode[key];
    ++entry.first;
    entry.second += std::chrono::duration<double, std::milli>(now - start).count();
    if (now - profile.lastReport < std::chrono::seconds(10)) return;
    profile.lastReport = now;
    std::vector<std::pair<std::uint32_t, std::pair<std::uint64_t, double>>> hot(profile.byOpcode.begin(), profile.byOpcode.end());
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    std::string report;
    for (std::size_t i = 0; i < hot.size() && i < 10; ++i) {
        char text[96];
        std::snprintf(text, sizeof(text), " %s x%llu %.0fms", hot[i].first == 0xffffu ? "flip" : Pm4::Name(hot[i].first << 8u).c_str(), static_cast<unsigned long long>(hot[i].second.first), hot[i].second.second);
        report += text;
    }
    auto& waits = Driver::waitOutcomes();
    auto& epochs = Driver::epochBumps();


    auto& costs = submissionCosts(queue);
    const auto submitted = costs.submissions.exchange(0);
    const auto perSubmission = [&](std::atomic<std::uint64_t>& ns) { const auto total = ns.exchange(0); return submitted != 0 ? static_cast<double>(total) / 1000.0 / static_cast<double>(submitted) : 0.0; };
    const auto validateUs = perSubmission(costs.validateNs), copyUs = perSubmission(costs.copyNs), dequeueUs = perSubmission(costs.dequeueNs), completeUs = perSubmission(costs.completeNs);
    const auto flipRoomUs = perSubmission(costs.flipRoomNs), flipHoldUs = perSubmission(costs.flipHoldNs), preparationUs = perSubmission(costs.preparationNs);
    const auto perSubmissionCount = [&](std::atomic<std::uint64_t>& count) { const auto total = count.exchange(0); return submitted != 0 ? static_cast<double>(total) / static_cast<double>(submitted) : 0.0; };
    const auto packetsChecked = perSubmissionCount(costs.validatedPackets), walks = perSubmissionCount(costs.commandWalks), registerLists = perSubmissionCount(costs.registerLists), registerBytes = perSubmissionCount(costs.registerBytes);
    const auto suspends = costs.suspends.exchange(0);
    const auto suspendUs = suspends != 0 ? static_cast<double>(costs.suspendNs.exchange(0)) / 1000.0 / static_cast<double>(suspends) : 0.0;
    std::string outcomes;
    for (std::size_t i = 0; i < profile.dispatchOutcomes.size(); ++i) {
        char text[64];
        std::snprintf(text, sizeof(text), " %s %llu %.0fms", DispatchOutcomeNames[i], static_cast<unsigned long long>(profile.dispatchOutcomes[i].first), profile.dispatchOutcomes[i].second);
        outcomes += text;
    }
    auto& poll = Driver::pollStats();
    AgcDriver::ProfilePrint_nid_no_patch( "[poll] queue 0x%x (10 s): ticks %llu, table hits %llu, tries %llu / failed %llu, submits %llu, reaps %llu (%llu with work; by reason: behind completion %llu / other %llu); longest try hold %.1f us, try holds %.1f ms\n", queue, static_cast<unsigned long long>(poll.ticks), static_cast<unsigned long long>(poll.tableHits), static_cast<unsigned long long>(poll.tries), static_cast<unsigned long long>(poll.triesFailed), static_cast<unsigned long long>(poll.submits), static_cast<unsigned long long>(poll.reaps), static_cast<unsigned long long>(poll.reapsWithWork), static_cast<unsigned long long>(poll.reapsBehindCompletion), static_cast<unsigned long long>(poll.reapsOther), poll.longestTryHoldUs, poll.tryHoldUs / 1000.0);
    poll = PollStats{};
    AgcDriver::ProfilePrint_nid_no_patch( "[packets] queue 0x%x %llu submissions, time by packet (10 s):%s, flush %.0fms; DISPATCH_DIRECT by outcome:%s; per submission (%llu submitted): validate %.1f us (%.1f packets, %.1f command walks, %.1f register lists of %.0f bytes), waits for flip room %.1f us, behind the worker at a flip %.1f us, for registration preparation %.1f us, copy %.1f us, dequeue %.1f us, complete %.1f us; suspend points %llu x %.1f us, %llu skipped (nothing open); end submits %llu made, %llu skipped, %llu notifies skipped; waits satisfied: at entry %llu, from recorder %llu (%llu while polling, same queue %llu, %llu without the GPU mutex), polled %llu (%llu entered without the GPU mutex, %llu entry tries failed), timed out %llu; poll submits %llu, poll reaps %llu; epoch bumps: submissions %llu, waits %llu, drains %llu, reaps %llu, packets %llu\n", queue, static_cast<unsigned long long>(profile.submissions), report.c_str(), profile.flushMs, outcomes.c_str(), static_cast<unsigned long long>(submitted), validateUs, packetsChecked, walks, registerLists, registerBytes, flipRoomUs, flipHoldUs, preparationUs, copyUs, dequeueUs, completeUs, static_cast<unsigned long long>(suspends), suspendUs, static_cast<unsigned long long>(costs.suspendsSkipped.exchange(0)), static_cast<unsigned long long>(costs.endSubmits.exchange(0)), static_cast<unsigned long long>(costs.endSkipped.exchange(0)), static_cast<unsigned long long>(costs.notifiesSkipped.exchange(0)), static_cast<unsigned long long>(waits.atEntry), static_cast<unsigned long long>(waits.fromRecorder + waits.fromRecorderPolling), static_cast<unsigned long long>(waits.fromRecorderPolling), static_cast<unsigned long long>(waits.fromRecorderSameQueue), static_cast<unsigned long long>(waits.fromRecorderUnlocked), static_cast<unsigned long long>(waits.polled), static_cast<unsigned long long>(waits.entriesUnlocked), static_cast<unsigned long long>(waits.entryTriesFailed), static_cast<unsigned long long>(waits.timedOut), static_cast<unsigned long long>(waits.pollSubmits), static_cast<unsigned long long>(waits.pollReaps), static_cast<unsigned long long>(epochs.submissions), static_cast<unsigned long long>(epochs.waits), static_cast<unsigned long long>(epochs.drains), static_cast<unsigned long long>(epochs.reaps), static_cast<unsigned long long>(epochs.packets));
    waits = WaitOutcomes{};
    epochs = EpochBumps{};
    profile.byOpcode.clear();
    profile.submissions = 0;
    profile.flushMs = 0;
    profile.dispatchOutcomes = {};
}

}
