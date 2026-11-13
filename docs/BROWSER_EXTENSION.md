# Browser extension

The OmniCortex extension shows what the [unified review](UNIFIED_REVIEW.md) knows about the model on
the page you are reading: Hugging Face (also hf.co and hf-mirror.com), ModelScope, and LM Studio's model
pages. Its toolbar icon, a small blade, glows blue when there is data, and the badge shows how many
statistics there are.

Click it for a dense view: one line per statistic with its headline value (sample-weighted mean of
rates, median of speeds). Each line opens into a table of every source's value with its setup and
sample size. You can filter by text, group, and source, and include or exclude related re-packagings
and self-reported extras. Filters are remembered. The links open the model in your Studio, ready to
download a file, or open the neuron review.

## Install

```bash
python extension/build.py        # dist/omnicortex-extension-chromium.zip and -firefox.zip
```

CI builds both packages and attaches them to tagged releases.

| Browser | How | Status |
|---|---|---|
| Chrome, Edge, Brave, Vivaldi, Opera (desktop) | `chrome://extensions` (Opera: `opera://extensions`), Developer mode, **Load unpacked** → `build/extension-chromium` | Tested in Chromium |
| Firefox (desktop) | `about:debugging` → **Load Temporary Add-on** → the Firefox zip. To keep it installed, it must be signed through addons.mozilla.org | Not tested yet |
| Firefox for Android (128+) | Needs a signed build from addons.mozilla.org (or a Nightly custom collection). Host access is granted in the add-on's settings | Supported by the manifest; not tested |
| Safari (macOS, iOS) | `xcrun safari-web-extension-converter build/extension-chromium`, then build in Xcode (needs an Apple developer account to distribute) | Not tested |
| Edge for Android | Only installs from Microsoft's store | Not published |
| Chrome for Android, Opera mobile, Samsung Internet | These do not run extensions | Not possible |

No store listing has been published yet. Publishing to addons.mozilla.org, the Chrome Web Store, and
Opera add-ons is a maintainer step.

## Sources

Open the extension's settings (⚙) and add one or both kinds of source:

- **Review boards**: `github:owner/repo`, `owner/repo`, or a scoreboard server's URL. A board is
  downloaded whole and cached (an hour by default). It never learns which model you looked at.
- **Studios**: `URL token`. Your Studio is asked about the model on the page. It adds your own
  observations and SAELens availability, and its answer includes its board.
  - Open Studio in the browser once to accept its certificate if it uses its own.
  - The token is kept in the browser's extension storage.

The browser asks you to allow each Studio or scoreboard host when you save. GitHub boards need no extra
permission.

## Privacy

The extension sends nothing about you. A page's model id goes only to the Studios you configured.
Board downloads are requests for a public file.

## Next

A later version can ask your Studio, or a remote host running it, to do contextual work from the page:
- test this model with TestQA;
- fetch it to a worker;
- compare it with the model you use now.

The `open.hub` link is the first such action.
