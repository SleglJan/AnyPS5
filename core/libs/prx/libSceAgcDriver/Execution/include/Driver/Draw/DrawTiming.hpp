#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWTIMING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWTIMING_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <list>

namespace AgcDriver::DriverDetail {

enum DrawDriverPhase { DrawRowPrologue, DrawRowPrecheck, DrawRowDecode, DrawRowProgramPrepare, DrawRowRecompile, DrawRowRectList, DrawRowVectors, DrawRowKeyLookupValidate, DrawRowLockWait, DrawRowLabels, DrawRowGraphics, DrawRowSkipped, DrawRowEpilogue, DrawDriverPhaseCount };

inline constexpr const char* DrawDriverPhaseNames[DrawDriverPhaseCount] = {"prologue", "precheck", "decode", "program prepare", "recompile", "rect-list", "vectors", "key/lookup/validate", "lock wait", "labels", "Graphics::Draw", "skipped", "epilogue"};

struct DrawPhaseTotals {
    std::array<double, DrawDriverPhaseCount> ms{};
    std::uint64_t packets = 0, drawn = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct PendingDrawPhases {
    bool phases = false;
    std::array<double, DrawDriverPhaseCount> ms{};
    std::chrono::steady_clock::time_point tailAt{};
};

struct DrawPhaseTiming {
    bool profile;
    std::array<double, DrawDriverPhaseCount>& phaseMs;
    std::chrono::steady_clock::time_point& phaseLap;
    void Phase(DrawDriverPhase which);
};

}

#endif
