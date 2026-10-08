"""Parse constant tables out of the execution tests' C++ sources (no compilation).

table(src, name)  -> list of rows (each a flat list of ints) for `Name[..][..] = {{..},..}`, `std::array<T,N> Name{{..}}`
                     and `T Name[] = {{..},..}` declarations; a 1-D table gives a flat list of ints.
names(src, name)  -> list of str-or-None for `const char* Name[N] = {"..", nullptr, ..}`.
kernel_words(src, name) -> the kernel's words; raises if the array holds anything but hex literals (named constants).
"""
import re

NUM = re.compile(r"\b(0x[0-9a-fA-F]+|\d+)(?:u|U|ull|ULL|ul|UL)?\b")


def strip_comments(text):
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def _body(src, name):
    src = strip_comments(src)
    m = re.search(r"\b" + re.escape(name) + r"\s*((?:\[[^\]]*\])*)\s*=?\s*\{", src)
    if not m:
        raise KeyError(f"table {name} not found")
    start = m.end() - 1
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1:i]
    raise ValueError(f"table {name}: unbalanced braces")


def _elements(body):
    """Split a brace body into top-level comma-separated elements."""
    out, depth, cur = [], 0, []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [e.strip() for e in out if e.strip()]


def _num(tok):
    return int(tok, 16) if tok.lower().startswith("0x") else int(tok)


def table(src, name):
    body = _body(src, name).strip()
    # std::array<T, N> Name{{ ... }} has one extra brace level
    if body.startswith("{") and len(_elements(body)) == 1 and _elements(body)[0].startswith("{"):
        inner = _elements(body)[0].strip()
        if _elements(inner[1:-1]) and _elements(inner[1:-1])[0].startswith("{"):
            body = inner[1:-1]
    elems = _elements(body)
    if all(not e.startswith("{") for e in elems):
        return [_num(m.group(1)) for e in elems for m in [NUM.search(e)] if m]
    rows = []
    for e in elems:
        rows.append([_num(m.group(1)) for m in NUM.finditer(e)])
    return rows


def names(src, name):
    body = _body(src, name)
    return [None if t == "nullptr" else t[1:-1] for t in re.findall(r'"[^"]*"|nullptr', body)]


def kernel_words(src, name):
    m = re.search(r"std::array<std::uint32_t,\s*\d+>\s*" + re.escape(name) + r"\s*\{(.*?)\};", strip_comments(src), re.S)
    if not m:
        raise KeyError(f"kernel {name} not found")
    toks = [t.strip() for t in m.group(1).split(",") if t.strip()]
    bad = [t for t in toks if not re.fullmatch(r"0x[0-9a-fA-F]{8}u?", t)]
    if bad:
        raise ValueError(f"kernel {name}: non-literal words {bad[:3]} (named constants); use the adapter's words_file")
    return [int(t.rstrip("u"), 16) for t in toks]
