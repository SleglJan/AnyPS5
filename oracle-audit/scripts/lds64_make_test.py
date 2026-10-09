"""Generate core/libs/prx/libSceAgcDriver/tests/execution/LdsFloatAtomicModes.cpp from the measured matrices.

Usage: python3 -s scripts/oracle/lds64_make_test.py notes/oracle-lds64/matrix.json AnyPS5/core/libs/prx/libSceAgcDriver/tests/execution/LdsFloatAtomicModes.cpp
32 lanes, each with an f32 pair, an f64 pair and the compare-and-store markers; the kernel runs the six f32 and six f64
forms in isolation on the lane's LDS slot. Expected tables: the titles' mode (f32 flushed, f64 kept), 0xf0 (both kept),
0x00 (both flushed), from the oracle runs in ieee0_* modes.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hosts", HERE / "lds64_hosts.py")
hosts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosts)

PAIRS = [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 0), (0, 6), (7, 1), (6, 7), (2, 6), (8, 2), (2, 8), (8, 9), (9, 8),
         (10, 8), (8, 8), (11, 2), (2, 11), (11, 8), (8, 11), (11, 12), (12, 11), (13, 11), (11, 13), (6, 11), (11, 6), (4, 11), (11, 4), (0, 11), (11, 0)]
MARKER = 0x4142434445464748
NAMES = ["ds_min_rtn_f32 memory", "ds_min_rtn_f32 returned", "ds_max_rtn_f32 memory", "ds_max_rtn_f32 returned", "ds_cmpst_rtn_f32 memory",
         "ds_cmpst_rtn_f32 returned", "ds_min_f32 memory", "ds_max_f32 memory", "ds_cmpst_f32 memory",
         "ds_min_rtn_f64 memory", "ds_min_rtn_f64 returned", "ds_max_rtn_f64 memory", "ds_max_rtn_f64 returned", "ds_min_f64 memory", "ds_max_f64 memory",
         "ds_cmpst_rtn_f64 memory", "ds_cmpst_rtn_f64 returned", "ds_cmpst_f64 memory"]


def f32_body(mem, rtn, op, three):
    operands = "v3, v5" + (", v10" if three else "")
    use = f"{op} {rtn}, {operands}" if rtn else f"{op} {operands}"
    return f"ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\n{use}\ns_waitcnt lgkmcnt(0)\nds_read_b32 {mem}, v3\ns_waitcnt lgkmcnt(0)\n"


def f64_body(mem, rtn, op, three):
    operands = "v3, v[8:9]" + (", v[40:41]" if three else "")
    use = f"{op} {rtn}, {operands}" if rtn else f"{op} {operands}"
    return f"ds_write_b64 v3, v[6:7]\ns_waitcnt lgkmcnt(0)\n{use}\ns_waitcnt lgkmcnt(0)\nds_read_b64 {mem}, v3\ns_waitcnt lgkmcnt(0)\n"


def kernel():
    text = "v_lshlrev_b32 v1, 5, v0\nv_lshlrev_b32 v2, 7, v0\nbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen\nbuffer_load_dwordx4 v[8:11], v1, s[0:3], 0 offen offset:16\ns_waitcnt vmcnt(0)\n"
    text += "".join(f"v_mov_b32 v{r}, 0\n" for r in range(12, 40))
    text += f"v_mov_b32 v40, 0x{MARKER & 0xffffffff:x}\nv_mov_b32 v41, 0x{MARKER >> 32:x}\nv_lshlrev_b32 v3, 3, v0\n"
    text += f32_body("v12", "v13", "ds_min_rtn_f32", False) + f32_body("v14", "v15", "ds_max_rtn_f32", False) + f32_body("v16", "v17", "ds_cmpst_rtn_f32", True)
    text += f32_body("v18", "", "ds_min_f32", False) + f32_body("v19", "", "ds_max_f32", False) + f32_body("v20", "", "ds_cmpst_f32", True)
    text += f64_body("v[22:23]", "v[24:25]", "ds_min_rtn_f64", False) + f64_body("v[26:27]", "v[28:29]", "ds_max_rtn_f64", False)
    text += f64_body("v[30:31]", "", "ds_min_f64", False) + f64_body("v[32:33]", "", "ds_max_f64", False)
    text += f64_body("v[34:35]", "v[36:37]", "ds_cmpst_rtn_f64", True) + f64_body("v[38:39]", "", "ds_cmpst_f64", True)
    for i, r in enumerate(range(12, 40, 4)):
        text += f"buffer_store_dwordx4 v[{r}:{r + 3}], v2, s[4:7], 0 offen" + (f" offset:{i * 16}" if i else "") + "\n"
    return text + "s_endpgm\n"


def table(m, f32_mode, f64_mode):
    rows = []
    for i, j in PAIRS:
        r = i * 14 + j
        f32 = m["f32"]["modes"][f32_mode][r][:9]
        mm = m["f64_minmax"]["modes"][f64_mode][r][:12]
        cs = m["f64_cmpst"]["modes"][f64_mode][r][:6]
        assert m["f32"]["rows"][r][:2] == [hosts.matrix.F32[i], hosts.matrix.F32[j]] and m["f64_minmax"]["rows"][r] == [hosts.matrix.F64[i], hosts.matrix.F64[j]]
        assert m["f64_cmpst"]["rows"][r] == [hosts.matrix.F64[i], hosts.matrix.F64[j], MARKER]
        rows.append(f32 + mm + cs)
    return rows


def fmt_rows(rows, per_line):
    out = []
    for r in rows:
        words = [f"0x{v:08x}u" for v in r]
        lines = [", ".join(words[k:k + per_line]) for k in range(0, len(words), per_line)]
        out.append("    {" + (",\n     ".join(lines)) + "}")
    return ",\n".join(out)


def main():
    m = json.load(open(sys.argv[1]))
    words = hosts.assemble(kernel(), False)
    inputs = []
    for i, j in PAIRS:
        a32, b32, a64, b64 = hosts.matrix.F32[i], hosts.matrix.F32[j], hosts.matrix.F64[i], hosts.matrix.F64[j]
        inputs.append([a32, b32, a64 & 0xffffffff, a64 >> 32, b64 & 0xffffffff, b64 >> 32, MARKER & 0xffffffff, 0])
    titles = table(m, "ieee0_flushed", "ieee0_kept")
    kept = table(m, "ieee0_kept", "ieee0_kept")
    flushed = table(m, "ieee0_flushed", "ieee0_flushed")
    code_lines = [", ".join(f"0x{w:08x}u" for w in words[k:k + 8]) for k in range(0, len(words), 8)]
    names = ",\n".join(f'    "{n}"' for n in NAMES)
    src = f'''#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {{

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Inputs = 8;
constexpr std::uint32_t Results = 32;
constexpr std::uint32_t Columns = 27;
alignas(256) std::array<std::uint32_t, Threads * Inputs> Input{{}};
alignas(256) std::array<std::uint32_t, Threads * Results> Output{{}};

alignas(256) constexpr std::array<std::uint32_t, {len(words)}> Code{{
    {(",\n    ".join(code_lines))},
}};

constexpr std::uint32_t Rows[{len(PAIRS)}][8] = {{
{fmt_rows(inputs, 8)}
}};

constexpr std::uint32_t ExpectedTitles[{len(PAIRS)}][Columns] = {{
{fmt_rows(titles, 9)}
}};

constexpr std::uint32_t ExpectedKept[{len(PAIRS)}][Columns] = {{
{fmt_rows(kept, 9)}
}};

constexpr std::uint32_t ExpectedFlushed[{len(PAIRS)}][Columns] = {{
{fmt_rows(flushed, 9)}
}};

constexpr const char* Names[18] = {{
{names}
}};

constexpr std::uint32_t Columns64 = 9;

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {{
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu}};
}}

std::string Hex(std::uint32_t value) {{
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}}

const char* ColumnName(std::uint32_t column) {{
    return column < Columns64 ? Names[column] : Names[Columns64 + (column - Columns64) / 2u];
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
    const ShaderRecompiler::ShaderComputeStageInfo compute{{{{Threads, 1, 1}}, 64u, {{false, false, false}}, false, 1}};
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

void Check(const std::uint32_t (&expected)[{len(PAIRS)}][Columns], const char* mode) {{
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {{
        const std::uint32_t* out = &Output[tid * Results];
        for (std::uint32_t column = 0; column < Columns; ++column) {{
            const std::uint32_t actual = out[column < Columns64 ? column : column + 1u];
            Require(actual == expected[tid][column], std::string("lds float atomic modes: ") + mode + " lane " + std::to_string(tid) + " " + ColumnName(column)
                + (column >= Columns64 && (column - Columns64) % 2u ? " high" : column >= Columns64 ? " low" : "") + " is " + Hex(actual) + ", expected " + Hex(expected[tid][column]));
        }}
    }}
}}

}}

int main() {{
    try {{
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, std::nullopt);
        Check(ExpectedTitles, "no float mode");
        Run(*device, ShaderRecompiler::ShaderFloatMode{{0xc0u, true, false, false}});
        Check(ExpectedTitles, "FLOAT_MODE 0xc0");
        Run(*device, ShaderRecompiler::ShaderFloatMode{{0xf0u, true, false, false}});
        Check(ExpectedKept, "FLOAT_MODE 0xf0");
        Run(*device, ShaderRecompiler::ShaderFloatMode{{0x00u, true, false, false}});
        Check(ExpectedFlushed, "FLOAT_MODE 0x00");
        std::puts("lds float atomic modes tests passed");
        return 0;
    }} catch (const std::exception& error) {{
        std::cerr << error.what() << '\\n';
        return 1;
    }}
}}
'''
    Path(sys.argv[2]).write_text(src)
    print("wrote", sys.argv[2], "words", len(words))


if __name__ == "__main__":
    main()
