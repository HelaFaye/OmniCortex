# OmniCortex client (Kivy)

A Python and [Kivy](https://kivy.org) app for an OmniCortex Studio, on the desktop and, later, Android and iOS
(docs/MOBILE_CLIENTS.md).

**Status:** the API layer only. `omnicortex_client/api.py` pairs with a Studio and uses it: models, chat
(streaming too), the reply checks and the review lookup. The interface is next (docs/STATUS.md).

```python
from omnicortex_client import api
p = api.Studio.pair("https://studio:7870/pair#c=ABCD-EFGH-JKMN&fp=…", name="my phone")  # from Connect → Pair a device
api.save(p)                       # ~/.config/omnicortex/client.json, readable by you only
st = api.Studio(api.load())
for piece in st.chat_stream([{"role": "user", "content": "Capital of Australia?"}]):
    print(piece, end="")
```

- **Tokens:** pairing gives this device its own token, which Studio can revoke. The owner's token is never used.
- **Certificates:** an `https` link carries Studio's certificate fingerprint, and every connection checks it. A
  self-signed Studio is trusted that way, and any other certificate is refused.
- **Plain `http` to another computer:** allowed, with a warning (`Studio.warning`), only for use inside a VPN.

Tests: `tests/test_kivy_client.py`.
