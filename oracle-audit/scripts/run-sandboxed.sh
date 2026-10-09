#!/usr/bin/env bash
# Run a relinked AnyPS5 executable inside bubblewrap: whole filesystem read-only, $HOME hidden,
# no network, only the run directory writable. X11 (XWayland), GPU devices and the audio sockets
# are passed through so the title can draw and play sound.
#
# Usage: run-sandboxed.sh <run-dir> [seconds]   (seconds: how long to keep it running, default 25)
set -u
RUN_DIR=$(realpath "$1")
SECONDS_TO_RUN=${2:-25}
LOG="$RUN_DIR/run.log"
LIBS_REAL=$(realpath "$RUN_DIR/libs" 2>/dev/null || true)
SHOT="$RUN_DIR/screenshot.png"
XAUTH=${XAUTHORITY:-}
RT=/run/user/$(id -u)

args=(--ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /home --tmpfs /root --tmpfs /tmp
      --bind "$RUN_DIR" "$RUN_DIR" --bind /tmp/.X11-unix /tmp/.X11-unix
      --unshare-net --unshare-pid --die-with-parent
      --setenv HOME /home/sandbox --setenv DISPLAY "${DISPLAY:-:0}" --chdir "$RUN_DIR")
[ -n "$XAUTH" ] && args+=(--ro-bind "$XAUTH" "$XAUTH" --setenv XAUTHORITY "$XAUTH")
[ -n "$LIBS_REAL" ] && args+=(--ro-bind "$LIBS_REAL" "$LIBS_REAL")
for l in "$RUN_DIR"/app0/* "$RUN_DIR"/download0/*; do
    [ -L "$l" ] && t=$(realpath "$l" 2>/dev/null) && [ -e "$t" ] && args+=(--ro-bind "$t" "$t")
done
for s in "$RT/pipewire-0" "$RT/pulse/native"; do
    [ -S "$s" ] && args+=(--bind "$s" "$s")
done
args+=(--setenv XDG_RUNTIME_DIR "$RT")

chmod +x "$RUN_DIR/app.elf"
bwrap "${args[@]}" -- "$RUN_DIR/app.elf" > "$LOG" 2>&1 &
BW=$!
echo "started pid $BW, log: $LOG"

sleep "$SECONDS_TO_RUN"
if kill -0 "$BW" 2>/dev/null; then
    echo "still running after ${SECONDS_TO_RUN}s"
    for id in $(xprop -root _NET_CLIENT_LIST 2>/dev/null | grep -oE '0x[0-9a-f]+'); do
        name=$(xprop -id "$id" WM_NAME 2>/dev/null | cut -d'"' -f2)
        cls=$(xprop -id "$id" WM_CLASS 2>/dev/null | cut -d'"' -f2)
        echo "window $id: name='$name' class='$cls'"
        import -window "$id" "$SHOT" 2>/dev/null && echo "screenshot: $SHOT"
    done
    kill "$BW" 2>/dev/null; sleep 1; kill -9 "$BW" 2>/dev/null
    echo "stopped"
else
    wait "$BW"; echo "exited early with code $?"
fi
