"""Entry point PyInstaller freezes into the omnicortex-remote executable (scripts/ci/)."""
from neuronscope_remote.agent import main

raise SystemExit(main())
