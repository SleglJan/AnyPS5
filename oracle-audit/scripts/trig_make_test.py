"""Generate core/libs/prx/libSceAgcDriver/tests/execution/TrigReduction.cpp: v_sin/v_cos f32 and f16 rows pinned on the oracle.

Usage: python3 -s scripts/oracle/trig_make_test.py out-dir AnyPS5/core/libs/prx/libSceAgcDriver/tests/execution/TrigReduction.cpp
Each lane: f32 x in v4, f16 x in the low half of v5; results v10 sin_f32, v11 cos_f32, v12 sin_f16, v13 cos_f16.
The expected table is the oracle's output in the titles' mode (f32 denormals flushed, f16 kept).
"""
import importlib.util
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep", HERE / "transcendental_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
spec2 = importlib.util.spec_from_file_location("hosts", HERE / "lds64_hosts.py")
hosts = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(hosts)


def f32(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def f16(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]


ROWS32 = [f32(-2.0 ** -126), f32(-2.0 ** -100), f32(-2.0 ** -30), f32(-1e-6), f32(2.0 ** -126), f32(1e-6), f32(-6.1e-5), f32(6.1e-5),
          0xbeffffff, 0x3effffff, f32(0.5), f32(-0.5), 0x3e7fffff, 0xbe7fffff, f32(0.25), f32(0.75),
          f32(255.5), f32(256.25), f32(-1000.5), f32(3.4e38), 0x00000000, 0x80000000, 0x7f800000, 0x7fc00000,
          f32(0.1), f32(-0.1), f32(0.3), f32(-0.7), f32(1.3), f32(-2.6), f32(10.25), f32(-100.125)]
ROWS16 = [0x8001, 0x8400, 0x9000, f16(-1e-4), 0x0001, 0x0400, f16(-6.1e-5), f16(6.1e-5),
          0xb7ff, 0x37ff, 0x3800, 0xb800, 0x33ff, 0xb3ff, 0x3400, 0x3a00,
          f16(255.5), f16(256.25), f16(-1000.5), 0x7bff, 0x0000, 0x8000, 0x7c00, 0x7e00,
          f16(0.1), f16(-0.1), f16(0.3), f16(-0.7), f16(1.3), f16(-2.6), f16(10.25), f16(-100.125)]
NAMES = ["v_sin_f32", "v_cos_f32", "v_sin_f16", "v_cos_f16"]
BODY = "v_sin_f32 v10, v4\nv_cos_f32 v11, v4\nv_sin_f16 v12, v5\nv_cos_f16 v13, v5\n"


def main():
    out = Path(sys.argv[1]).resolve()
    rows = [[a, b, 0, 0] for a, b in zip(ROWS32, ROWS16)]
    (out / "trig-test-body.s").write_text(BODY)
    (out / "trig-test-rows.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in rows))
    import subprocess
    p = subprocess.run([sys.executable, "-s", str(sweep.ORACLE)] + sweep.flags(0) + ["--outs", "4", str(out / "trig-test-body.s"), str(out / "trig-test-rows.txt")], cwd=sweep.ROOT / "AnyPS5", capture_output=True, text=True, check=True)
    (out / "trig-test.txt").write_text(p.stdout)
    expected = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]
    kernel = "v_lshlrev_b32 v1, 4, v0\nv_lshlrev_b32 v2, 6, v0\nbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen\ns_waitcnt vmcnt(0)\n" + "".join(f"v_mov_b32 v{r}, 0\n" for r in range(10, 26)) + BODY
    kernel += "buffer_store_dwordx4 v[10:13], v2, s[4:7], 0 offen\ns_endpgm\n"
    words = hosts.assemble(kernel, False)
    code = ",\n    ".join(", ".join(f"0x{w:08x}u" for w in words[k:k + 8]) for k in range(0, len(words), 8))
    rows_text = ",\n".join("    {" + ", ".join(f"0x{v:08x}u" for v in r) + "}" for r in rows)
    expected_text = ",\n".join("    {" + ", ".join(f"0x{v:08x}u" for v in e[:4]) + "}" for e in expected)
    src = f'''#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {{

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 16;
constexpr std::uint32_t Columns = 4;
alignas(256) std::array<std::uint32_t, Threads * Inputs> Input{{}};
alignas(256) std::array<std::uint32_t, Threads * Results> Output{{}};

alignas(256) constexpr std::array<std::uint32_t, {len(words)}> Code{{
    {code},
}};

constexpr std::uint32_t Rows[Threads][4] = {{
{rows_text}
}};

constexpr std::uint32_t Expected[Threads][Columns] = {{
{expected_text}
}};

constexpr const char* Names[Columns] = {{"v_sin_f32", "v_cos_f32", "v_sin_f16", "v_cos_f16"}};
constexpr std::uint32_t Tolerance[Threads] = {{TOLERANCES}};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {{
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu}};
}}

std::string Hex(std::uint32_t value) {{
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}}

std::uint32_t Ordered(std::uint32_t bits, std::uint32_t width) {{
    const std::uint32_t sign = 1u << (width - 1u);
    return (bits & sign) != 0u ? sign - (bits & ~sign) : bits | sign;
}}

std::uint32_t Distance(std::uint32_t actual, std::uint32_t expected, std::uint32_t width) {{
    const std::uint32_t a = Ordered(actual, width);
    const std::uint32_t e = Ordered(expected, width);
    return a > e ? a - e : e - a;
}}

bool IsNan(std::uint32_t bits, std::uint32_t width) {{
    return width == 16u ? (bits & 0x7fffu) > 0x7c00u : (bits & 0x7fffffffu) > 0x7f800000u;
}}

void Run(AgcDriver::VulkanDevice& device, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode) {{
    for (std::uint32_t tid = 0; tid < Threads; ++tid) std::copy(std::begin(Rows[tid]), std::end(Rows[tid]), &Input[tid * Inputs]);
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{{{Threads, 1, 1}}, 0u, {{false, false, false}}, false, 1}};
    ShaderRecompiler::RecompileRequest request{{
        {{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {{}}}},
        {{32, 0, userData, compute, std::nullopt, std::nullopt, memory}},
        device.Target(),
        {{0, 0, 0, 128}}
    }};
    request.useCache = false;
    request.context.floatMode = floatMode;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {{}}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}}

void Check(const char* mode) {{
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {{
        const std::uint32_t* out = &Output[tid * Results];
        for (std::uint32_t column = 0; column < Columns; ++column) {{
            const std::uint32_t width = column < 2u ? 32u : 16u;
            const std::uint32_t mask = width == 16u ? 0xffffu : 0xffffffffu;
            const std::uint32_t actual = out[column] & mask;
            const std::uint32_t expected = Expected[tid][column] & mask;
            const std::string name = std::string("trig reduction: ") + mode + " lane " + std::to_string(tid) + " " + Names[column] + " is " + Hex(actual) + ", expected " + Hex(expected);
            if (IsNan(expected, width)) {{
                Require(IsNan(actual, width), name);
                continue;
            }}
            Require(Distance(actual, expected, width) <= Tolerance[tid], name + " (tolerance " + std::to_string(Tolerance[tid]) + " ulp)");
        }}
    }}
}}

}}

int main() {{
    try {{
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, std::nullopt);
        Check("no float mode");
        Run(*device, ShaderRecompiler::ShaderFloatMode{{0xc0u, true, false, false}});
        Check("FLOAT_MODE 0xc0");
        std::puts("trig reduction tests passed");
        return 0;
    }} catch (const std::exception& error) {{
        std::cerr << error.what() << '\\n';
        return 1;
    }}
}}
'''
    tolerances = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 12, 12, 12, 12]
    Path(sys.argv[2]).write_text(src.replace("TOLERANCES", ", ".join(f"{t}u" for t in tolerances)))
    print("wrote", sys.argv[2], "words", len(words), "rows", len(rows))
    for i, (r, e) in enumerate(zip(rows, expected)):
        print(f"  lane {i:2d} x32={r[0]:08x} x16={r[1]:04x} -> {' '.join(f'{v:08x}' for v in e[:4])}")


if __name__ == "__main__":
    main()
