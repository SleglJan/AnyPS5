#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchTiming.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"

namespace AgcDriver::DriverDetail {

void Driver::addDriverPhases(DispatchClass which, const std::array<double, DriverPhaseCount>& ms, bool hit) {
    std::lock_guard lock(driverPhasesMutex);
    auto& totals = driverPhaseTotals[which];
    for (std::size_t i = 0; i < DriverPhaseCount; ++i) totals.ms[i] += ms[i];
    ++totals.dispatches;
    if (hit) ++totals.hits;
    const auto now = std::chrono::steady_clock::now();
    if (now - driverPhasesReport < std::chrono::seconds(10)) return;
    driverPhasesReport = now;
    for (std::size_t cls = 0; cls < DispatchClassCount; ++cls) {
        auto& line = driverPhaseTotals[cls];
        if (line.dispatches == 0) continue;
        std::string report;
        double total = 0;
        for (std::size_t i = 0; i < DriverPhaseCount; ++i) {
            char text[64];
            std::snprintf(text, sizeof(text), " %s %.1f", DriverPhaseNames[i], line.ms[i] * 1000 / static_cast<double>(line.dispatches));
            report += text;
            total += line.ms[i];
        }
        AgcDriver::ProfilePrint_nid_no_patch("[dispatch] driver phases %s (10 s, %llu dispatches, %llu cache hits), us per dispatch:%s, total %.1f (%.1f ms)\n", DispatchClassNames[cls], static_cast<unsigned long long>(line.dispatches), static_cast<unsigned long long>(line.hits), report.c_str(), total * 1000 / static_cast<double>(line.dispatches), total);
        line = {};
    }
    if (const auto* residency = Graphics::ResidencyDriver::Active(); residency != nullptr) {
        static Graphics::ResidencyDriverCounts previous;
        const auto counts = residency->Counts();
        AgcDriver::ProfilePrint_nid_no_patch("[dispatch] segment programs (10 s): %llu settled, %llu unsettled on the walk program, %llu on the segment program, %llu segment pending on the walk program, %llu out-of-segment errors\n", static_cast<unsigned long long>(counts.settledDispatches - previous.settledDispatches), static_cast<unsigned long long>(counts.unsettledDispatches - previous.unsettledDispatches), static_cast<unsigned long long>(counts.segmentDispatches - previous.segmentDispatches), static_cast<unsigned long long>(counts.segmentPending - previous.segmentPending), static_cast<unsigned long long>(counts.segmentFaults - previous.segmentFaults));
        previous = counts;
    }
}

PendingDispatchPhases& Driver::pendingDispatchPhases() {
    static thread_local PendingDispatchPhases pending;
    return pending;
}

std::chrono::steady_clock::time_point& Driver::packetStartedAt() {
    static thread_local std::chrono::steady_clock::time_point started{};
    return started;
}

double DispatchPhaseTiming::Elapsed() {
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<double, std::milli>(now - lap).count();
    lap = now;
    return ms;
}

void DispatchPhaseTiming::Phase(DriverPhase which) {
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    phaseMs[which] += std::chrono::duration<double, std::milli>(now - phaseLap).count();
    phaseLap = now;
}

}
