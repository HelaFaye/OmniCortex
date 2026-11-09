#!/usr/bin/env python3
"""Single entry point showing all OmniCortex transfer modes."""
import argparse
import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(prog='ns-transfer'); sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('cli-http',help='reliable resumable CLI sender/receiver')
    sub.add_parser('webrtc',help='launch the existing WebRTC transfer GUI')
    sub.add_parser('worker',help='remote adaptive tuning worker mode')
    sub.add_parser('mcp',help='MCP control plane')
    a,rest=p.parse_known_args()
    if a.mode=='cli-http': return subprocess.call([sys.executable,str(ROOT/'oc_transfer.py'),*rest])
    if a.mode=='webrtc': return subprocess.call([sys.executable,str(ROOT.parent/'viz'/'transfer_lab.py'),*rest])
    if a.mode=='worker': return subprocess.call([sys.executable,str(ROOT/'tuning_worker.py'),*rest])
    if a.mode=='mcp': return subprocess.call([sys.executable,str(ROOT/'oc_transfer_mcp.py'),*rest])
    return 2
if __name__=='__main__': raise SystemExit(main())
