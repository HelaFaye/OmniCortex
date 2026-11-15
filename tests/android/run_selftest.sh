#!/usr/bin/env bash
# Runs the client's self-test (clients/kivy/omnicortex_client/selftest.py) in a booted Android emulator against
# tests/android/fake_studio.py, and collects the evidence: logcat, the app's own screenshots, and screen captures
# of what the app cannot photograph itself (Google's QR scanner, the final screen).
#
#   tests/android/run_selftest.sh APK OUTDIR        # adb must see one emulator; run from the repository root
#
# Exit status: 0 when the self-test ends with fail=0 and the app never crashed; 1 otherwise.
set -uo pipefail
APK=$1
OUT=$2
PKG=org.omnicortex.omnicortex
mkdir -p "$OUT"

python3 tests/android/fake_studio.py --host 0.0.0.0 --port 8790 --link-host 10.0.2.2 --out "$OUT/link.txt" \
  > "$OUT/studio.log" 2>&1 &
STUDIO=$!
trap 'kill $STUDIO 2>/dev/null' EXIT
for _ in $(seq 60); do [ -s "$OUT/link.txt" ] && break; sleep 1; done
[ -s "$OUT/link.txt" ] || { echo "the fake Studio did not start"; cat "$OUT/studio.log"; exit 1; }

adb install -r -g "$APK" || exit 1
adb shell getprop ro.build.version.release | sed 's/^/Android /'

# The self-test file goes into the app's private storage: only run-as can write there, and only on a debug build.
printf '{"link": "%s", "name": "emulator", "shots": true}\n' "$(cat "$OUT/link.txt")" > "$OUT/oc_selftest.json"
adb push "$OUT/oc_selftest.json" /data/local/tmp/oc_selftest.json >/dev/null
adb shell chmod 644 /data/local/tmp/oc_selftest.json
adb shell "run-as $PKG sh -c 'mkdir -p files && cp /data/local/tmp/oc_selftest.json files/oc_selftest.json'"
adb shell "run-as $PKG ls -l files/oc_selftest.json" || { echo "could not place the self-test file"; exit 1; }

adb logcat -c
adb logcat -v time > "$OUT/logcat.txt" 2>&1 &
LOGCAT=$!
trap 'kill $STUDIO $LOGCAT 2>/dev/null' EXIT
adb shell am start -W -n "$PKG/org.kivy.android.PythonActivity"

ours_on_top() { adb shell dumpsys activity activities | grep -E 'mResumedActivity|topResumedActivity' | grep -q "$PKG"; }
DONE=""
SHOT_SCAN=0
for _ in $(seq 480); do                       # up to 8 minutes: the first start unpacks Python
  sleep 1
  if grep -q "\[OC-SELFTEST.*scan-open" "$OUT/logcat.txt" && [ $SHOT_SCAN = 0 ]; then
    SHOT_SCAN=1
    sleep 5
    adb exec-out screencap -p > "$OUT/scan-screen.png"
    adb shell dumpsys activity activities | grep -E 'mResumedActivity|topResumedActivity' > "$OUT/scan-activity.txt"
    ours_on_top || adb shell input keyevent KEYCODE_BACK       # close Google's scanner, never the app
  fi
  DONE=$(grep -o "\[OC-SELFTEST.*DONE pass=[0-9]* fail=[0-9]*" "$OUT/logcat.txt" | tail -1)
  [ -n "$DONE" ] && break
  if grep -qE "FATAL EXCEPTION|Python for android ended|SIGSEGV" "$OUT/logcat.txt"; then
    echo "the app crashed"; break
  fi
done
sleep 2
adb exec-out screencap -p > "$OUT/final-screen.png"

# the app's own screenshots, from its private storage
for f in $(adb shell "run-as $PKG ls files" | tr -d '\r' | grep '^selftest-.*\.png$'); do
  adb exec-out "run-as $PKG cat files/$f" > "$OUT/$f"
done
grep -E "\[OC-SELFTEST|Traceback|Error|FATAL" "$OUT/logcat.txt" | grep -v "OpenGL" > "$OUT/selftest.txt" || true
echo "----- self-test -----"
cat "$OUT/selftest.txt"
[ -s "$OUT/scan-activity.txt" ] && { echo "----- screen while scanning -----"; cat "$OUT/scan-activity.txt"; }
echo "----- result: ${DONE:-no result} -----"
case "$DONE" in *"fail=0") exit 0;; *) exit 1;; esac
