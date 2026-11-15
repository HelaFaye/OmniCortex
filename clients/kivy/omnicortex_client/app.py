"""
The OmniCortex client app (Kivy): pair with a Studio, chat with a model of your choice (streamed), check a reply
(the H-neuron risk strip and the truth probe), and look a model up in the unified review.

    python -m omnicortex_client          # from clients/kivy (or: python clients/kivy/main.py)

The work happens in controller.py; this file only draws it.
"""
from __future__ import annotations

import os
import platform

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen

from .controller import Client

KV = """
#:import dp kivy.metrics.dp

<Small@Label>:
    font_size: '13sp'
    color: 0.75, 0.75, 0.8, 1
    text_size: self.width, None
    size_hint_y: None
    height: self.texture_size[1]

<Btn@Button>:
    size_hint_y: None
    height: dp(44)

<PairScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(10)
        Label:
            text: '[b]Pair with OmniCortex Studio[/b]'
            markup: True
            font_size: '20sp'
            size_hint_y: None
            height: dp(40)
        Small:
            text: "In Studio: Connect, then Pair a device. Paste the link it shows (or type host:port#c=CODE)."
        TextInput:
            id: link
            hint_text: 'https://studio:7870/pair#c=ABCD-EFGH-JKMN&fp=…'
            multiline: False
            size_hint_y: None
            height: dp(44)
        TextInput:
            id: name
            hint_text: 'Name for this device in Studio'
            text: root.default_name
            multiline: False
            size_hint_y: None
            height: dp(44)
        Btn:
            text: 'Pair'
            on_release: root.pair(link.text, name.text)
        Small:
            id: status
            text: root.status
            markup: True
        Widget:

<MessageView>:
    orientation: 'vertical'
    size_hint_y: None
    height: self.minimum_height
    padding: dp(8)
    spacing: dp(4)
    canvas.before:
        Color:
            rgba: (0.16, 0.22, 0.32, 1) if root.role == 'user' else (0.14, 0.14, 0.17, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(8)]
    Small:
        text: root.header
    Label:
        text: root.body
        markup: True
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]
    Small:
        text: root.note
        opacity: 1 if root.note else 0
        height: self.texture_size[1] if root.note else 0
    BoxLayout:
        size_hint_y: None
        height: dp(36) if root.role == 'assistant' else 0
        opacity: 1 if root.role == 'assistant' else 0
        disabled: root.role != 'assistant'
        spacing: dp(6)
        Button:
            text: 'Check (risk strip)'
            on_release: app.check(root)
        Button:
            text: 'Truth probe…'
            on_release: app.ask_truth(root)

<ChatScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(8)
        spacing: dp(6)
        BoxLayout:
            size_hint_y: None
            height: dp(44)
            spacing: dp(6)
            Spinner:
                id: model
                text: 'auto'
                values: ['auto']
                on_text: app.choose_model(self.text)
            Button:
                text: 'Refresh'
                size_hint_x: None
                width: dp(80)
                on_release: app.refresh_models()
            Button:
                text: 'Review'
                size_hint_x: None
                width: dp(80)
                on_release: app.root.current = 'lookup'
            Button:
                text: 'New'
                size_hint_x: None
                width: dp(60)
                on_release: app.new_chat()
            Button:
                text: 'Unpair'
                size_hint_x: None
                width: dp(80)
                on_release: app.unpair()
        Small:
            id: status
            text: root.status
            markup: True
        ScrollView:
            id: scroll
            do_scroll_x: False
            BoxLayout:
                id: log
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(8)
                padding: dp(2)
        BoxLayout:
            size_hint_y: None
            height: dp(88)
            spacing: dp(6)
            TextInput:
                id: entry
                hint_text: 'Message'
            BoxLayout:
                orientation: 'vertical'
                size_hint_x: None
                width: dp(80)
                spacing: dp(4)
                Button:
                    text: 'Send'
                    on_release: app.send(entry.text)
                Button:
                    text: 'Stop'
                    on_release: app.client.stop()

<LookupScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(12)
        spacing: dp(8)
        BoxLayout:
            size_hint_y: None
            height: dp(44)
            spacing: dp(6)
            Button:
                text: 'Chat'
                size_hint_x: None
                width: dp(90)
                on_release: app.root.current = 'chat'
            TextInput:
                id: model
                hint_text: 'Model id or Hugging Face repo (org/name)'
                multiline: False
                on_text_validate: app.lookup(self.text)
            Button:
                text: 'Look up'
                size_hint_x: None
                width: dp(90)
                on_release: app.lookup(model.text)
        ScrollView:
            Label:
                id: result
                text: root.result
                text_size: self.width, None
                size_hint_y: None
                height: self.texture_size[1]
                halign: 'left'
"""


class PairScreen(Screen):
    status = StringProperty("")
    default_name = StringProperty(f"{platform.node() or 'device'}")

    def pair(self, link, name):
        App.get_running_app().pair(link, name)


class ChatScreen(Screen):
    status = StringProperty("")


class LookupScreen(Screen):
    result = StringProperty("")


class MessageView(BoxLayout):
    role = StringProperty("")
    header = StringProperty("")
    body = StringProperty("")
    note = StringProperty("")
    message = ObjectProperty(None, allownone=True)

    def show(self, m):
        from .controller import escape_markup
        self.message, self.role = m, m.role
        self.header = "You" if m.role == "user" else (m.model if m.model and m.model != "auto" else "Model")
        self.body = m.check or escape_markup(m.content) or "…"
        self.note = m.note


class OmniCortexApp(App):
    title = "OmniCortex"

    def build(self):
        Builder.load_string(KV)
        from kivy.uix.screenmanager import ScreenManager
        self.client = Client(data_dir=os.environ.get("OC_CLIENT_DIR"), post=lambda fn: Clock.schedule_once(lambda dt: fn()))
        sm = ScreenManager()
        for s in (PairScreen(name="pair"), ChatScreen(name="chat"), LookupScreen(name="lookup")):
            sm.add_widget(s)
        self.views = {}
        if self.client.load():
            sm.current = "chat"
            Clock.schedule_once(lambda dt: self.after_pair(), 0)
        else:
            sm.current = "pair"
        from kivy.core.window import Window          # needs a display: imported only when the app runs
        Window.softinput_mode = "below_target"
        return sm

    # ------------------------------------------------------------- helpers
    def screen(self, name):
        return self.root.get_screen(name)

    def say(self, text, error=False, where="chat"):
        from .controller import escape_markup
        t = escape_markup(text)
        self.screen(where).status = f"[color=ff7070]{t}[/color]" if error else t

    # ------------------------------------------------------------- pairing
    def pair(self, link, name):
        from . import api
        try:
            self.client.pair(link, name)
        except (api.StudioError, OSError) as e:
            return self.say(str(e), error=True, where="pair")
        self.root.current = "chat"
        self.after_pair()

    def after_pair(self):
        w = self.client.warning()
        self.say(w or f"Paired with {self.client.pairing.base}.", error=bool(w))
        self.refresh_models()
        self.redraw()

    def unpair(self):
        self.client.forget()
        self.screen("chat").ids.log.clear_widgets()
        self.views = {}
        self.screen("pair").status = "Unpaired. Studio still lists this device until you revoke it there."
        self.root.current = "pair"

    # ------------------------------------------------------------- models
    def refresh_models(self):
        from . import api
        try:
            models = self.client.refresh_models()
        except api.StudioError as e:
            return self.say(str(e), error=True)
        sp = self.screen("chat").ids.model
        sp.values = ["auto"] + models
        sp.text = self.client.pairing.model

    def choose_model(self, model):
        if self.client.pairing and model != self.client.pairing.model:
            self.client.choose(model)

    # ------------------------------------------------------------- chat
    def view_for(self, m):
        v = self.views.get(id(m))
        if v is None:
            v = self.views[id(m)] = MessageView()
            self.screen("chat").ids.log.add_widget(v)
        v.show(m)
        return v

    def redraw(self):
        log = self.screen("chat").ids.log
        log.clear_widgets()
        self.views = {}
        for m in self.client.messages:
            self.view_for(m)

    def scroll_end(self):
        Clock.schedule_once(lambda dt: setattr(self.screen("chat").ids.scroll, "scroll_y", 0), 0)

    def send(self, text):
        if self.client.busy:
            return self.say("Still answering; Stop first, or wait.")
        entry = self.screen("chat").ids.entry
        reply = self.client.send(text, on_piece=self.view_for, on_done=self.reply_done, on_error=self.reply_error)
        if reply is None:
            return
        entry.text = ""
        for m in self.client.messages[-2:]:
            self.view_for(m)
        self.say("Answering…")
        self.scroll_end()

    def reply_done(self, m):
        self.view_for(m)
        self.say("")
        self.scroll_end()

    def reply_error(self, msg):
        self.say(msg, error=True)

    def new_chat(self):
        self.client.clear()
        self.redraw()

    # ------------------------------------------------------------- reply checks
    def check(self, view):
        self.say("Checking the reply…")
        self.client.check(view.message, on_done=lambda m: (self.view_for(m), self.say("")),
                          on_error=lambda e: self.say(e, error=True))

    def ask_truth(self, view):
        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.textinput import TextInput
        box = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))
        box.add_widget(Label(text="Other answers to compare with, one per line (optional):", size_hint_y=None,
                             height=dp(30)))
        alts = TextInput()
        box.add_widget(alts)
        go = Button(text="Check", size_hint_y=None, height=dp(44))
        box.add_widget(go)
        pop = Popup(title="Truth probe", content=box, size_hint=(0.9, 0.6))

        def run(*_):
            pop.dismiss()
            self.say("Asking the truth probe…")
            self.client.truth(view.message, alts.text.splitlines(), on_done=lambda m: (self.view_for(m), self.say("")),
                              on_error=lambda e: self.say(e, error=True))
        go.bind(on_release=run)
        pop.open()

    # ------------------------------------------------------------- review lookup
    def lookup(self, model):
        scr = self.screen("lookup")
        scr.result = "Looking up…"
        self.client.lookup(model, on_done=lambda text: setattr(scr, "result", text),
                           on_error=lambda e: setattr(scr, "result", e))


def main():
    OmniCortexApp().run()
