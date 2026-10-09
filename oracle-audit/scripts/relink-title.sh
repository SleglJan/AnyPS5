#!/usr/bin/env bash
# Turn an extracted PS5 homebrew title folder into a runnable AnyPS5 layout.
#
# Usage: relink-title.sh <title-dir> <run-dir> [--host-libc]
#   <title-dir>   folder with eboot.bin, sce_sys/, optional sce_module/ (as shipped in a catalog ZIP)
#   <run-dir>     output folder: app.elf, libs/, app0/, download0/
#   --host-libc   leave the title's bundled sce_module/libc.prx out and use AnyPS5's own libc
#
# Steps: unwrap the fake-signed SELF containers, relink, copy the built system libraries,
# copy the title's resources into app0/, run the import audit.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
RELINKER="$ROOT/build/core/relinker/relinker"
LIBS="$ROOT/build/core/libs/libs"

TITLE=$(realpath "$1")
RUN=$(realpath -m "$2")
HOST_LIBC=${3:-}

[ -x "$RELINKER" ] || { echo "no relinker at $RELINKER; build first" >&2; exit 2; }
[ -f "$TITLE/eboot.bin" ] || { echo "no eboot.bin in $TITLE" >&2; exit 2; }

UNW="$RUN/unwrapped"
rm -rf "$RUN"
mkdir -p "$UNW" "$RUN/libs" "$RUN/app0" "$RUN/download0"

python3 -I "$HERE/unfself.py" "$TITLE/eboot.bin" "$UNW/input.elf"
if [ -d "$TITLE/sce_module" ]; then
    mkdir -p "$UNW/sce_module"
    for m in "$TITLE"/sce_module/*; do
        python3 -I "$HERE/unfself.py" "$m" "$UNW/sce_module/$(basename "$m")"
    done
fi

opts=(--registry)
[ "$HOST_LIBC" = "--host-libc" ] && opts+=(--exclude-sce-module libc.prx)
"$RELINKER" "${opts[@]}" "$UNW/input.elf" "$RUN/app.elf"

rmdir "$RUN/libs" && ln -s "$(realpath --relative-to="$RUN" "$LIBS")" "$RUN/libs"
for entry in "$TITLE"/*; do
    name=$(basename "$entry")
    case "$name" in eboot.bin|sce_module|sce_modules|prx) continue ;; esac
    cp -r "$entry" "$RUN/app0/"
done
chmod +x "$RUN/app.elf"

echo
echo "== import audit =="
mods=()
[ -d "$UNW/sce_module" ] && [ "$HOST_LIBC" != "--host-libc" ] && mods=(--modules "$UNW/sce_module")
python3 -s "$ROOT/AnyPS5/tools/import_audit.py" "$RUN/app.registry.json" --libs "$RUN/libs" "${mods[@]}" || true
echo
echo "run it with: scripts/run-sandboxed.sh $RUN [seconds]"
