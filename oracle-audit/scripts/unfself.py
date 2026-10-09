"""Unwrap a fake-signed PS5 SELF (plaintext, uncompressed segments) into a plain ELF.

Usage: python3 -I unfself.py <input.self> <output.elf>

Layout of a fake-signed SELF (as written by make_fself):
  0x00  SELF header (magic 4F 15 3D 1D, header_size, file_size, num_entries, ...)
  0x20  num_entries * 0x20 segment entries {props, offset, filesz, memsz}
  then  the original ELF header + program headers, verbatim
  then  segment data at entry.offset, for entries whose props have the BLOCKED bit (1 << 11);
        bits 20..31 of props give the program-header index the data belongs to.
Entries with the HAS_DIGESTS bit (1 << 16) are signature metadata and carry no ELF data.
"""
import struct
import sys

SELF_MAGICS = (b"\x4f\x15\x3d\x1d", b"\x54\x14\xf5\xee")  # executable and library variants
PROPS_BLOCKED = 1 << 11
PROPS_HAS_DIGESTS = 1 << 16
PROPS_ENCRYPTED = 1 << 1
PROPS_DEFLATED = 1 << 3


def main(src: str, dst: str) -> int:
    with open(src, "rb") as f:
        data = f.read()
    if data[:4] not in SELF_MAGICS:
        print(f"{src}: not a SELF container (magic {data[:4].hex()})", file=sys.stderr)
        return 2

    (_magic, _ver, _mode, _endian, _attr, _key_type, header_size, _meta_size,
     file_size, num_entries, _flags) = struct.unpack_from("<IBBBBIHHQHH", data, 0)
    entries = [struct.unpack_from("<QQQQ", data, 0x20 + 0x20 * i) for i in range(num_entries)]
    elf_off = 0x20 + 0x20 * num_entries
    if data[elf_off:elf_off + 4] != b"\x7fELF":
        print(f"{src}: no ELF header at {elf_off:#x}", file=sys.stderr)
        return 2

    e_phoff, = struct.unpack_from("<Q", data, elf_off + 0x20)
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, elf_off + 0x36)
    phdrs = [struct.unpack_from("<IIQQQQQQ", data, elf_off + e_phoff + e_phentsize * i)
             for i in range(e_phnum)]
    headers_len = e_phoff + e_phentsize * e_phnum

    # Size the output from the program headers, then lay the data back at each p_offset.
    out_len = headers_len
    for (_t, _f, p_offset, _va, _pa, p_filesz, _ms, _al) in phdrs:
        out_len = max(out_len, p_offset + p_filesz)
    out = bytearray(out_len)
    out[:headers_len] = data[elf_off:elf_off + headers_len]
    # Drop section-header references: the SELF carries none.
    struct.pack_into("<Q", out, 0x28, 0)
    struct.pack_into("<HHH", out, 0x3a, 0, 0, 0)

    restored = set()
    for props, offset, filesz, _memsz in entries:
        if props & PROPS_HAS_DIGESTS or not props & PROPS_BLOCKED:
            continue
        if props & (PROPS_ENCRYPTED | PROPS_DEFLATED):
            print(f"{src}: encrypted or deflated segment, cannot unwrap", file=sys.stderr)
            return 2
        idx = (props >> 20) & 0xFFF
        p_offset, p_filesz = phdrs[idx][2], phdrs[idx][5]
        if filesz != p_filesz:
            print(f"{src}: entry for phdr {idx} has {filesz:#x} bytes, phdr says {p_filesz:#x}",
                  file=sys.stderr)
            return 2
        out[p_offset:p_offset + filesz] = data[offset:offset + filesz]
        restored.add(idx)

    # Segments with file data that no entry restored. Anything the SELF appends after
    # file_size is used for the first unrestored one (SCE version/comment data), in order.
    tail = data[file_size:]
    for idx, (_t, _f, p_offset, _va, _pa, p_filesz, _ms, _al) in enumerate(phdrs):
        if p_filesz == 0 or idx in restored:
            continue
        covered = any(phdrs[j][2] <= p_offset and p_offset + p_filesz <= phdrs[j][2] + phdrs[j][5]
                      for j in restored)
        if covered:
            continue
        chunk = tail[:p_filesz]
        tail = tail[p_filesz:]
        out[p_offset:p_offset + len(chunk)] = chunk
        print(f"phdr {idx} (type {phdrs[idx][0]:#x}, {p_filesz:#x} bytes): "
              f"{len(chunk):#x} bytes taken from the SELF tail", file=sys.stderr)

    with open(dst, "wb") as f:
        f.write(out)
    print(f"{dst}: {len(out):#x} bytes, {len(restored)} segments restored from {num_entries} entries")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
