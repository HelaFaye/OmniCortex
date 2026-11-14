# Plan: phones and other cross-platform clients

This is a plan, not a feature yet. OmniCortex's own client will use Python and Kivy, not Flutter.

It was prompted by [orailnoor/cross-platform-llm-client](https://github.com/orailnoor/cross-platform-llm-client) ("PrivateLM",
MIT): a Flutter app for Android and iOS (web is cloud-only) that does two things:
- runs GGUF models on the device through a llama.cpp plugin (Vulkan on Android, Metal on iOS);
- calls OpenAI, Anthropic, Gemini and Kimi.

It has no custom server address yet, so it cannot talk to an OmniCortex Studio.

## What OmniCortex already offers a client

- **Studio's OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`): JIT loading, auto routing, per-subject
  models, linked hosts' models.
- **Device pairing** (Connect → Pair a device): a QR code or link gives a phone its own revocable token.
- **The pages work on a phone screen.** The chat page and Settings already adapt to narrow screens.

## Steps, smallest first

1. **Studio as a web app (S).** Add a web app manifest and icons, so "Add to home screen" gives a full-screen
   Studio on any phone, with nothing to install. The chat already works on a phone.
2. **A custom OpenAI-compatible provider in PrivateLM-style clients (S, upstream).** Propose a "Custom
   (OpenAI-compatible)" provider to the project: base URL plus key. Pairing then gives the phone its Studio URL and
   device token. Many mobile clients already have this; a short doc for each would do.
3. **Pairing by QR into those clients (S).** Add an `omnicortex://pair?url=…&token=…` link, or plain JSON in the
   QR code, that such a client can read. The pairing page shows it beside the browser link.
4. **Phone hardware profiles (M).** Hardware support profiles for mobile GPUs, telling an on-device client which
   quant and context fit:
   - Adreno and Mali through Vulkan;
   - Apple A- and M-series through Metal.

   The unified review's speed reports per hardware can then include phones.
5. **An OmniCortex client in Kivy (M)** *(in progress: the API layer is in `clients/kivy`, the interface is next)*. Python and [Kivy](https://kivy.org), so the client shares code with the
   rest of OmniCortex (`scripts/plugin_api.py`, the TestQA grader, the review lookup) instead of rewriting it in
   another language:
   - desktop (Linux, Windows, macOS) runs it with `pip install kivy`;
   - Android builds with Buildozer (python-for-android);
   - iOS builds with kivy-ios, which needs a Mac and Xcode.

   It covers chat, the reply checks (risk strip, Fix This Mistake), the emotion jewel and the review lookup, all
   through Studio's API and a paired device token. The Godot client still covers 3D and VR.
6. **On-device models that report back (L).** The Kivy client can also run a GGUF on the phone through
   llama-cpp-python, which python-for-android can build (a recipe for it, with Vulkan off at first). It sends its
   TestQA results and speed to the review board. Because llama-cpp-python exposes llama.cpp's eval callback
   (`cb_eval`), a Python client could in principle capture activations on the device too, which the Flutter
   wrappers do not allow. Whether that is fast enough on a phone is untested; start with results only.

## What to watch for

- **Tokens on phones:** use paired device tokens (revocable, per device), never the master token.
- **Over the internet:** Studio behind its TLS plus pairing, or reached over WireGuard or Tailscale. Do not expose
  llama-server directly.
- **On-device models and OmniCortex's interpretability tools:** those tools need activations. Flutter's llama.cpp
  wrappers do not expose them; a Kivy client with llama-cpp-python might, but treat the phone as a client of the
  analysis until that is measured.
- **Kivy builds:** pin the python-for-android and kivy-ios versions, and keep native code (llama.cpp) in recipes,
  so a build does not depend on what happens to be on the build machine.
