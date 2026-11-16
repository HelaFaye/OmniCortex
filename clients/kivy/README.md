# OmniCortex client (Kivy)

A Python and [Kivy](https://kivy.org) app for an OmniCortex Studio, on the desktop and Android (iOS later;
docs/MOBILE_CLIENTS.md).

## Run it (desktop)

```bash
pip install -r clients/kivy/requirements.txt      # Kivy 2.3 or newer
python clients/kivy/main.py                       # or, from clients/kivy: python -m omnicortex_client
```

1. **Pair.** In Studio: Connect, then Pair a device. Either:
   - **Scan QR code** and point the camera at the code Studio shows (Android: Google's code scanner; desktop: a
     webcam, with `pip install opencv-python-headless`);
   - or paste the link it shows.

   Give this device a name first. The pairing is saved where only you can read it:
   - desktop: `~/.config/omnicortex/client.json` (`OC_CLIENT_DIR` puts it somewhere else);
   - Android: the app's private storage.

   The app opens straight to the chat next time.
2. **Chat.** Pick a model at the top (`auto` is whatever Studio is serving) and send. The reply streams in;
   **Stop** ends it early, **New** starts a fresh conversation.
3. **Check a reply.** Under each reply:
   - **Risk** colours the tokens the model's H-neuron classifier flags: yellow for some risk, orange for elevated,
     red for high. It needs a classifier set for that model in Studio.
   - **Truth…** asks whether the model's own activations agree with its reply. You can add other answers to
     compare with. It needs a truth probe set for that model in Studio (docs/TRUTH_PROBE.md).
   - **Emotions** draws the emotion jewel for the reply: it plays the reply token by token, then holds the reply's
     mean (tap to replay), with the strongest emotions beside it. It needs emotion vectors set for that model in
     Studio (docs/FEATURES.md, "In the clients").
   - **Fix…** sends what the reply should have said. Studio keeps the pair for training (Fix This Mistake). It
     takes pairs from a device only when its owner allows that (Settings → Integrated options).
4. **Review.** Looks up a model id or Hugging Face repo in the unified review: accuracy, speed per hardware and
   features, from the review boards and Studio's own records.

**Unpair** forgets the saved pairing on this device. Studio keeps listing the device until you revoke it there.

## Security

- **Tokens:** pairing gives this device its own token, which Studio can revoke. The owner's token is never used.
- **Certificates:** an `https` link carries Studio's certificate fingerprint, and every connection checks it. A
  self-signed Studio is trusted that way, and any other certificate is refused.
- **Plain `http` to another computer:** allowed only for use inside a VPN, and the app says so after pairing.

## Code

| File | What it does |
|---|---|
| `omnicortex_client/api.py` | Studio's API: pairing with certificate pinning, models, chat and streaming, the reply checks, the review lookup, the saved pairing |
| `omnicortex_client/controller.py` | The app's logic without Kivy: the conversation, streaming on a worker thread, the checks, and the text the screens show |
| `omnicortex_client/app.py` | The Kivy screens: pair, chat, review |
| `omnicortex_client/jewel.py` | The emotion jewel, drawn with Kivy's canvas |
| `omnicortex_client/qr.py` | QR pairing: Google's code scanner on Android, OpenCV on the desktop |
| `buildozer.spec` | The Android build |
| `main.py` | Entry point (also what Buildozer runs on Android) |

Tests: `tests/test_kivy_client.py`. They cover the API and controller against a fake Studio over TLS. If Kivy and
`xvfb-run` are installed, they also build the screens on a virtual display.

## Android

```bash
pip install buildozer==1.5.0 cython==3.0.11
cd clients/kivy && buildozer android debug          # bin/omnicortex-0.1.0-*-debug.apk
```

CI builds the same APK on every change to the client (`.github/workflows/android.yml`; download it from the run's
artifacts). The app asks only for network access. Google's code scanner needs Play services; without them,
paste the link.

## Testing

- **Unit tests:** `tests/test_kivy_client.py`. They cover the API and controller against a fake Studio over
  pinned TLS, the display text, the jewel's colours and the QR round trip.
- **The whole app:** `omnicortex_client/selftest.py` drives every screen in order:
  - pair over pinned TLS;
  - list models;
  - stream a chat;
  - Risk, Truth, Emotions and Fix;
  - Review;
  - start the QR scanner;
  - reload the saved pairing.

  It runs only when `oc_selftest.json` is in the app's data folder, which on Android only `adb run-as` can write,
  on a debug build. It logs `[OC-SELFTEST] <step> ok|FAIL` and saves screenshots.
  - **Desktop:** `tests/test_kivy_client.py` runs it under Xvfb against `tests/android/fake_studio.py`.
  - **Android:** CI's `emulator` job (`.github/workflows/android.yml`) builds an x86_64 APK and boots an
    Android 13 emulator with Play services. `tests/android/run_selftest.sh` then runs the self-test there. The
    logcat, the app's screenshots and a capture of Google's scanner screen are kept as the run's artifact.

## Not yet

- Tested on a real phone: CI runs it in an emulator, but nobody has installed it on a phone yet.
- iOS (kivy-ios, which needs a Mac and Xcode).
