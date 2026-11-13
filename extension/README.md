# OmniCortex browser extension

See [docs/BROWSER_EXTENSION.md](../docs/BROWSER_EXTENSION.md). `python extension/build.py` packages it;
`python extension/make_icons.py` redraws the icons.

- `src/sites.js`: which pages show which model.
- `src/normalize.js`: the JS twin of `scripts/unified_review.py`. `tests/test_unified_review.py` checks they agree.
- `src/background.js`: fetches, caches, and lights the icon.
- `src/content.js`: reports page changes.
- `src/popup.*`: the popup.
- `src/options.*`: the sources settings page.
