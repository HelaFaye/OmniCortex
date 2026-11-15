# Android build of the OmniCortex client (docs/MOBILE_CLIENTS.md):
#   pip install buildozer==1.5.0 cython==3.0.11
#   cd clients/kivy && buildozer android debug        # bin/omnicortex-*.apk
# CI builds the same APK (.github/workflows/android.yml). Versions are pinned so a build does not depend on what
# happens to be on the build machine.

[app]
title = OmniCortex
package.name = omnicortex
package.domain = org.omnicortex
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
source.exclude_dirs = bin,.buildozer,__pycache__
version = 0.1.0
# python3 brings OpenSSL (TLS with certificate pinning); Kivy 2.3 needs filetype to load images; pyjnius
# reaches Google's code scanner for QR pairing
requirements = python3,kivy==2.3.1,filetype,pyjnius
orientation = portrait
fullscreen = 0

# Network only. QR pairing uses Google's code scanner, which runs its own camera screen: no camera permission.
android.permissions = INTERNET
android.api = 34
android.minapi = 24
android.archs = arm64-v8a, armeabi-v7a
android.accept_sdk_license = True
android.enable_androidx = True
android.gradle_dependencies = com.google.android.gms:play-services-code-scanner:16.1.0
android.allow_backup = False
p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 0
