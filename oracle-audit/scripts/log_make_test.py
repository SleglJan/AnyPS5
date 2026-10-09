"""Generate core/libs/prx/libSceAgcDriver/tests/execution/LogNearOne.cpp: v_log_f32 and v_log_f16 rows pinned on the oracle.

Usage: python3 -s scripts/oracle/log_make_test.py out-dir <LogNearOne.cpp> [tolerances: 64 comma-separated ULP counts]
Each lane: f32 x in v4, f16 x in the low half of v5; results v10 log_f32, v11 log_f16. Zero, infinite and NaN results must match the
bits; the others are compared in ULPs with a per-lane tolerance. The expected table is the oracle's output in the titles' mode
(f32 denormals flushed, f16 kept); the test runs in FLOAT_MODE 0xc0.
"""
import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep", HERE / "transcendental_sweep.py"); sweep = importlib.util.module_from_spec(spec); spec.loader.exec_module(sweep)
spec2 = importlib.util.spec_from_file_location("hosts", HERE / "lds64_hosts.py"); hosts = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(hosts)
f32 = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
f16 = lambda v: struct.unpack("<H", struct.pack("<e", v))[0]

ROWS32 = [f32(1.0 + s * 2.0 ** -k) for k in range(1, 25) for s in (1.0, -1.0)] + [f32(v) for v in (1.0, 0.875, 1.125, 0.9, 1.1, 0.5, 2.0, 3.0, 10.0, 0.1, 1e-3, 1e3)] + [0x00800000, 0x7f7fffff, 0x00000000, 0xbf800000]
ROWS16 = [f16(1.0 + s * 2.0 ** -k) if k <= 11 else f16(1.0 + s * 2.0 ** -11) for k in range(1, 25) for s in (1.0, -1.0)] + [f16(v) for v in (1.0, 0.875, 1.125, 0.9, 1.1, 0.5, 2.0, 3.0, 10.0, 0.1, 1e-3, 1e3)] + [0x0400, 0x7bff, 0x0000, 0xbc00]
assert len(ROWS32) == 64 and len(ROWS16) == 64
NAMES = ["v_log_f32", "v_log_f16"]
BODY = "v_log_f32 v10, v4\nv_log_f16 v11, v5\n"


def main():
    out = Path(sys.argv[1]).resolve()
    rows = [[a, b, 0, 0] for a, b in zip(ROWS32, ROWS16)]
    (out / "log-test-body.s").write_text(BODY)
    (out / "log-test-rows.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in rows))
    p = subprocess.run([sys.executable, "-s", str(sweep.ORACLE)] + sweep.flags(0) + ["--outs", "2", str(out / "log-test-body.s"), str(out / "log-test-rows.txt")], cwd=sweep.ROOT / "AnyPS5", capture_output=True, text=True, check=True)
    (out / "log-test.txt").write_text(p.stdout)
    expected = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]
    kernel = "v_lshlrev_b32 v1, 4, v0\nbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen\ns_waitcnt vmcnt(0)\n" + "".join(f"v_mov_b32 v{r}, 0\n" for r in range(10, 14)) + BODY + "buffer_store_dwordx4 v[10:13], v1, s[4:7], 0 offen\ns_endpgm\n"
    words = hosts.assemble(kernel, False)
    tolerances = [int(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0] * 64
    assert len(tolerances) == 64
    code = ",\n    ".join(", ".join(f"0x{w:08x}u" for w in words[k:k + 8]) for k in range(0, len(words), 8))
    rows_text = ",\n".join("    {" + ", ".join(f"0x{v:08x}u" for v in r) + "}" for r in rows)
    expected_text = ",\n".join("    {" + ", ".join(f"0x{v:08x}u" for v in e[:2]) + "}" for e in expected)
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
#include <span>
#include <string>
#include <vector>

namespace {{

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 4;
constexpr std::uint32_t Columns = 2;
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

constexpr const char* Names[Columns] = {{"v_log_f32", "v_log_f16"}};
constexpr std::uint32_t Tolerance[Threads] = {{{", ".join(f"{t}u" for t in tolerances)}}};

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

bool IsPinned(std::uint32_t bits, std::uint32_t width) {{
    const std::uint32_t magnitude = width == 16u ? bits & 0x7fffu : bits & 0x7fffffffu;
    const std::uint32_t infinity = width == 16u ? 0x7c00u : 0x7f800000u;
    return magnitude == 0u || magnitude >= infinity;
}}

void Run(AgcDriver::VulkanDevice& device) {{
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
    request.context.floatMode = ShaderRecompiler::ShaderFloatMode{{0xc0u, true, false, false}};
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {{}}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}}

void Check() {{
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {{
        const std::uint32_t* out = &Output[tid * Results];
        for (std::uint32_t column = 0; column < Columns; ++column) {{
            const std::uint32_t width = column == 0u ? 32u : 16u;
            const std::uint32_t mask = width == 16u ? 0xffffu : 0xffffffffu;
            const std::uint32_t actual = out[column] & mask;
            const std::uint32_t expected = Expected[tid][column] & mask;
            const std::string name = "log near one: lane " + std::to_string(tid) + " " + Names[column] + " of " + Hex(Rows[tid][column]) + " is " + Hex(actual) + ", expected " + Hex(expected);
            if (IsPinned(expected, width)) {{
                Require(actual == expected, name);
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
        Run(*device);
        Check();
        std::puts("log near one tests passed");
        return 0;
    }} catch (const std::exception& error) {{
        std::cerr << error.what() << '\\n';
        return 1;
    }}
}}
'''
    Path(sys.argv[2]).write_text(src)
    print("wrote", sys.argv[2], "words", len(words), "rows", len(rows))
    for i, (r, e) in enumerate(zip(rows, expected)):
        print(f"  lane {i:2d} x32={r[0]:08x} x16={r[1]:04x} -> {e[0]:08x} {e[1]:08x}")


if __name__ == "__main__":
    main()
