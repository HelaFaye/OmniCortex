# Widgets

Small pieces of UI that take data and settings from outside, so any view can
use them. The web pages share one script (`viz/widgets.py`, `WIDGETS_JS`); the
Godot client has the same widgets in `plugins/godot-client/client/widgets/`, drawn the same way
and reading the same settings keys.

| widget | where | what it shows |
|---|---|---|
| **jewel** | 3D replay, live viewer, Godot | The model's emotion vectors at the current token. Click it to hide or show the emotion graph (the jewel stays); right-click for the next style. |
| **emotion graph** | 3D replay, live viewer, Godot | One lane per emotion over the reply, as z against the emotion stories. Click to jump to a token. |
| **nav gizmo** | Godot | Blender-style viewport navigation (below). |
| **chat box** | Godot | Search the trace's tokens, features and emotions; `>` asks the model. |

In the Godot client every widget sits on the UI layer, not in the 3D scene, so
zooming, panning or orbiting cannot move them off screen.

## The jewel

What it reads is the emotion track the decoding lab records
([FEATURES.md](FEATURES.md#watch-while-it-generates)): each emotion's z at
each token.

- **Colour** is the lane colours mixed by |z|²; an emotion below its mean is mixed in darker.
- **Size and glow** follow the strongest |z|, full at 3.
- **Motion** follows each emotion separately (below).
- Values glide toward each token's numbers rather than jumping, so stepping or streaming reads as one motion.

| style | how the values move it |
|---|---|
| `core` (default) | A reactor core with one ribbon per emotion on its own tilted orbit. A ribbon's length, width and speed follow its \|z\|. Positive z turns one way, negative the other. The core pulses with the strongest emotion. |
| `icosphere` (the jewel-styles add-on) | A sphere pushed out (positive z) or in (negative) around a direction for each emotion. Faces take the colours of the emotions shaping them. |
| `tesseract` (the jewel-styles add-on) | A 4D cube. It turns in one 4D plane as positive emotions rise and in another as negative ones do. The inner cube takes the mixed colour; each outer edge takes one emotion's colour and weight. |
| `off` | A faint ring. |

## Navigating the Godot 3D view

| | |
|---|---|
| orbit | drag in the view, or drag the gizmo |
| pan | Shift + drag, middle-drag, or drag the ✋ button |
| zoom | the wheel, `+` / `-`, or drag the ⌕ button |
| preset views | click an axis ball (front, back, left, right, top, bottom); numpad 1, 3, 7, with Ctrl for the opposite side |
| frame everything | ⛶, Home or numpad `.` |
| perspective / orthographic | ◇ / ▣, or numpad 5 |
| sway | ∿ or S. The slow automatic sway is on by default and pauses when you move the view. Turning it on or off is remembered. |

The web replay has its own camera bar (Front, Side, Top, Reset, Orbit, Follow flags).

## Settings

Later sources win:

1. the defaults in `viz/widgets.py` (`SPEC`, which also lists every setting's range);
2. the 3D theme's own `"widgets"` block in `viz/themes.json`;
3. the owner's choices in **Settings → Appearance → Widgets**, which has a live preview;
4. a viewer's own picks: the jewel's right-click style and whether the graph is hidden. A browser keeps these in local storage; Godot keeps them in `user://widgets.json`.

Studio serves the merged block with every 3D theme (`api/theme` → `widgets`).
It saves the owner's choices with `POST /api/settings/widgets {"widgets": {...}}`.
Values outside `SPEC` are refused.

| widget | setting | |
|---|---|---|
| jewel | `style` | `core`, `off`, and styles from add-ons (`icosphere`, `tesseract`) |
| | `speed` | 0.1 to 4 |
| | `glow` | 0 to 2 |
| | `scale` | 0.5 to 1, its size within its box |
| emotion_graph | `lanes` | 1 to 12, the strongest first |
| | `labels` | names in their lane colours |
| nav_gizmo | `visible`, `sway` | |
| chat_box | `max_hits` | 3 to 30 |

## Writing a widget

- **Web.** `NSWidgets.register(name, (element, opts) => api)`; a page creates one with `NSWidgets.create(name, element, {settings, ...})`. Keep per-viewer state in `NSWidgets.store`.
- **Godot.** Extend `res://widgets/widget.gd`, set `widget_id`, draw in `_draw()`, and read `setting(key, default)`. The host calls `set_theme_data(theme)` and `set_data(dict)`; `save_state()` and `save_setting()` persist what the viewer changes.
- **Both.** Add the settings to `SPEC` so Settings can draw them and Studio can check them.

Jewel styles beyond the reactor core come from add-ons ([ADDONS.md](ADDONS.md#default-packages)). A style
whose add-on is off falls back to the core.

