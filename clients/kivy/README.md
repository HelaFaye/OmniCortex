# OmniCortex client (Kivy)

A Python and [Kivy](https://kivy.org) app for an OmniCortex Studio, on the desktop now and Android and iOS later
(docs/MOBILE_CLIENTS.md).

## Run it (desktop)

```bash
pip install -r clients/kivy/requirements.txt      # Kivy 2.3 or newer
python clients/kivy/main.py                       # or, from clients/kivy: python -m omnicortex_client
```

1. **Pair.** In Studio: Connect, then Pair a device. Paste the link it shows into the app, and give this device a
   name. The pairing is saved in `~/.config/omnicortex/client.json` (readable by you only), so the app opens
   straight to the chat next time. `OC_CLIENT_DIR` puts it somewhere else.
2. **Chat.** Pick a model at the top (`auto` is whatever Studio is serving) and send. The reply streams in;
   **Stop** ends it early, **New** starts a fresh conversation.
3. **Check a reply.** Under each reply:
   - **Check (risk strip)** colours the tokens the model's H-neuron classifier flags: yellow for some risk,
     orange for elevated, red for high. It needs a classifier set for that model in Studio.
   - **Truth probe…** asks whether the model's own activations agree with its reply. You can add other answers
     to compare with. It needs a truth probe set for that model in Studio (docs/TRUTH_PROBE.md).
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
| `main.py` | Entry point (also what Buildozer runs on Android) |

Tests: `tests/test_kivy_client.py`. They cover the API and controller against a fake Studio over TLS. If Kivy and
`xvfb-run` are installed, they also build the screens on a virtual display.

## Not yet

- Pairing by QR code: paste the link for now. On Android, a camera scan comes with the Buildozer build.
- The Android build (Buildozer), then iOS (kivy-ios, which needs a Mac).
- The emotion jewel and Fix This Mistake from Studio's chat.
