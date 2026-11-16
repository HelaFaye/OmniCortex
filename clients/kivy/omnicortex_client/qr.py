"""
Pairing by QR code: Studio's Connect -> Pair a device shows the link as a QR code (segno), and the app reads it.

- Android: Google's code scanner (ML Kit, through Play services). It shows its own camera screen, so the app needs
  no camera permission, and the image never reaches the app: only the decoded text does.
- Desktop: the webcam through OpenCV (`pip install opencv-python-headless`), for up to 30 seconds; or a picture of
  the code (a screenshot works).
- Anywhere: pasting the link works without any of these.

Whatever is read goes through api.parse_pair_link before it is used, so a code that is not a Studio pairing link
(a web address, a Wi-Fi code) is refused rather than followed.
"""
from __future__ import annotations

import threading

from . import api


class ScanError(RuntimeError):
    pass


DOWNLOADING = "Google Play services is still downloading the QR scanner"
CODE_FAILED = "the app's scanner code failed"     # an error in this file, as opposed to a device without a scanner


def on_android() -> bool:
    try:
        from kivy.utils import platform
    except ImportError:
        return False
    return platform == "android"


def check_link(text: str) -> str:
    """The text if it is a Studio pairing link, else ScanError."""
    text = (text or "").strip()
    try:
        api.parse_pair_link(text)
    except api.StudioError as e:
        raise ScanError(f"that code is not a Studio pairing link ({e})") from e
    return text


def decode_image(path: str) -> str:
    """The pairing link in a picture of the QR code (desktop)."""
    try:
        import cv2
    except ImportError as e:
        raise ScanError("reading QR codes on the desktop needs OpenCV: pip install opencv-python-headless; "
                        "or paste the link") from e
    img = cv2.imread(str(path))
    if img is None:
        raise ScanError(f"cannot read the picture {path}")
    text, _pts, _ = cv2.QRCodeDetector().detectAndDecode(img)
    if not text:
        raise ScanError("no QR code found in the picture")
    return check_link(text)


def scan_webcam(on_result, on_error, seconds: float = 30.0, device: int = 0):
    """Reads the webcam on a worker thread until a QR code shows a pairing link (desktop)."""
    try:
        import cv2
    except ImportError:
        return on_error("scanning on the desktop needs OpenCV: pip install opencv-python-headless; or paste the link")

    def work():
        import time
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return on_error("no camera found; paste the link instead")
        det, end, last = cv2.QRCodeDetector(), time.time() + seconds, ""
        try:
            while time.time() < end:
                ok, frame = cap.read()
                if not ok:
                    continue
                text, _pts, _ = det.detectAndDecode(frame)
                if text and text != last:
                    last = text
                    try:
                        return on_result(check_link(text))
                    except ScanError as e:
                        on_error(str(e))
            on_error("no pairing code seen; hold Studio's QR code in front of the camera, or paste the link")
        finally:
            cap.release()
    threading.Thread(target=work, daemon=True).start()


def scan_android(on_result, on_error):
    """Google's code scanner (com.google.android.gms:play-services-code-scanner, in buildozer.spec)."""
    try:
        from jnius import PythonJavaClass, autoclass, java_method
    except ImportError:
        return on_error("QR scanning needs the Android build")

    class Success(PythonJavaClass):
        __javainterfaces__ = ["com/google/android/gms/tasks/OnSuccessListener"]
        __javacontext__ = "app"

        @java_method("(Ljava/lang/Object;)V")
        def onSuccess(self, barcode):
            text = barcode.getRawValue() if barcode is not None else ""
            try:
                on_result(check_link(text))
            except ScanError as e:
                on_error(str(e))

    class Failure(PythonJavaClass):
        __javainterfaces__ = ["com/google/android/gms/tasks/OnFailureListener"]
        __javacontext__ = "app"

        @java_method("(Ljava/lang/Exception;)V")
        def onFailure(self, e):
            msg = str(e.getMessage() or "")
            if "download" in msg.lower():         # Play services is still fetching the scanner (first use)
                on_error(f"{DOWNLOADING}: try again in a minute, or paste the link")
            else:
                on_error(f"the scanner could not run ({msg}); paste the link instead")

    from jnius import JavaException
    try:
        activity = autoclass("org.kivy.android.PythonActivity").mActivity
        Scanning = autoclass("com.google.mlkit.vision.codescanner.GmsBarcodeScanning")
        # every format: whatever is read must still pass check_link, so a QR-only filter would add nothing
        task = Scanning.getClient(activity).startScan()
        # keep the listeners alive as long as the task (pyjnius would otherwise collect them)
        scan_android.listeners = (Success(), Failure())
        task.addOnSuccessListener(scan_android.listeners[0])
        task.addOnFailureListener(scan_android.listeners[1])
    except JavaException as e:                                # no Play services, an old device
        on_error(f"the scanner is not available on this device ({e}); paste the link instead")
    except Exception as e:                                    # a bug here, not the device: say so plainly
        on_error(f"{CODE_FAILED} ({type(e).__name__}: {e}); paste the link instead")


def scan(on_result, on_error):
    """The platform's scanner: results and errors come back on another thread; the app posts them to its own."""
    if on_android():
        return scan_android(on_result, on_error)
    return scan_webcam(on_result, on_error)
