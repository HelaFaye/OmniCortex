"""The Proxmox backend (scripts/proxmox_lxc.py), the MI50 VBIOS check (scripts/amd_vbios.py) and the ARM hardware
profiles, on fake sysfs trees: an RK3588 board with an MI50, and an x86 server with a Tesla M10."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amd_vbios as VB  # noqa: E402
import hw_profiles as HW  # noqa: E402
import proxmox_lxc as PX  # noqa: E402


def w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def mi50_board(root: Path, vbios="113-D1631711-100", bar=0, resize=False, kfd=True):
    w(root / "proc/device-tree/compatible", "xunlong,orangepi-5-plus\x00rockchip,rk3588\x00")
    for i, cap in enumerate([414] * 4 + [1024] * 4):
        w(root / f"sys/devices/system/cpu/cpu{i}/cpu_capacity", str(cap))
    dev = root / "sys/bus/pci/devices/0000:01:00.0"
    w(dev / "vendor", "0x1002")
    w(dev / "device", "0x66a1")
    w(dev / "class", "0x038000")
    start, end = (0x900000000, 0x900000000 + bar - 1) if bar else (0, 0)
    w(dev / "resource", f"0x{start:016x} 0x{end:016x} 0x000000000014220c\n")
    if resize:
        w(dev / "resource0_resize", "0000000000007f00")
    w(dev / "drm/card1/device/vbios_version", vbios)
    (dev / "drm/renderD128").mkdir(parents=True)
    (root / "drv/amdgpu").mkdir(parents=True)
    os.symlink(root / "drv/amdgpu", dev / "driver")
    if kfd:
        w(root / "dev/kfd", "")
    return root


def m10_server(root: Path, uvm=True):
    w(root / "proc/device-tree/compatible", "")
    (root / "sys/kernel/iommu_groups/0").mkdir(parents=True)
    for i, bus in enumerate(["05", "06", "07", "08"]):
        dev = root / f"sys/bus/pci/devices/0000:{bus}:00.0"
        w(dev / "vendor", "0x10de")
        w(dev / "device", "0x13bd")
        w(dev / "class", "0x030200")
        w(root / f"dev/nvidia{i}", "")
    w(root / "dev/nvidiactl", "")
    if uvm:
        w(root / "dev/nvidia-uvm", "")
        w(root / "dev/nvidia-uvm-tools", "")
    w(root / "proc/driver/nvidia/version", "NVRM version: NVIDIA UNIX aarch64 Kernel Module  580.95.05  Release Build")
    return root


def test_rk3588_mi50_host_facts(tmp_path):
    f = PX.host_facts(str(mi50_board(tmp_path)), machine="aarch64")
    assert f["board"] == "rockchip,rk3588" and not f["coherent_pcie"] and not f["vfio_possible"]
    assert f["pcie_window"] == 1 << 30 and f["big_cores"] == "4-7" and f["kfd"]
    (g,) = f["gpus"]
    assert g["model"] == "Instinct MI50" and g["driver"] == "amdgpu" and g["render"] == ["/dev/dri/renderD128"]


def test_rk3588_mi50_plan(tmp_path):
    f = PX.host_facts(str(mi50_board(tmp_path)), machine="aarch64")
    pl = PX.plan(f, 101, "agent", render_gid=993, rocm="/opt/rocm", models="/srv/models")
    assert ["pct", "set", "101", "-dev0", "/dev/kfd,gid=993"] in pl["pct"]
    assert ["pct", "set", "101", "-dev1", "/dev/dri/renderD128,gid=993"] in pl["pct"]
    assert ["pct", "set", "101", "-mp0", "/opt/rocm,mp=/opt/rocm,ro=1"] in pl["pct"]
    assert ["pct", "set", "101", "-mp1", "/srv/models,mp=/models"] in pl["pct"]
    assert pl["conf"] == ["lxc.cgroup2.cpuset.cpus: 4-7"]
    assert pl["env"]["HSA_USERPTR_FOR_PAGED_MEM"] == "0"
    assert any("VFIO" in x for x in pl["warnings"])
    assert pl["port"] == 47871 and any("omnicortex-remote.sh serve" in x for x in pl["container"])
    text = PX.render(pl)
    assert "pct set 101 -dev0 /dev/kfd,gid=993" in text and "/etc/pve/lxc/101.conf" in text
    # without the container's render gid, the plan says how to find it
    assert any("getent group render" in x for x in PX.plan(f, 101)["warnings"])
    with pytest.raises(ValueError):
        PX.plan(f, 5)


def test_m10_server_plan(tmp_path):
    f = PX.host_facts(str(m10_server(tmp_path)), machine="x86_64")
    assert f["coherent_pcie"] and f["vfio_possible"] and len(f["gpus"]) == 4
    assert f["nvidia_driver"] == "580.95.05"
    pl = PX.plan(f, 102, "studio", cpus=None)
    devs = [c[4] for c in pl["pct"] if c[3].startswith("-dev")]
    assert devs == ["/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2",
                    "/dev/nvidia3", "/dev/nvidiactl"]
    assert any("580.95.05.run --no-kernel-module" in x for x in pl["container"])
    assert any("Tesla M10" in x for x in pl["warnings"]) and not pl["conf"]
    assert any("viz/studio.py --host 0.0.0.0 --port 7870" in x for x in pl["container"])
    # the uvm nodes appear only after CUDA runs on the host: the plan passes them anyway and says so
    f2 = PX.host_facts(str(m10_server(tmp_path / "b", uvm=False)), machine="x86_64")
    pl2 = PX.plan(f2, 102)
    assert any("nvidia-modprobe -u" in x for x in pl2["warnings"])
    assert "/dev/nvidia-uvm" in [c[4] for c in pl2["pct"]]


def test_check_inside_container(tmp_path):
    root = mi50_board(tmp_path)
    w(root / "dev/dri/renderD128", "")
    res = PX.check(str(root), env={})
    assert {"pass"} <= {x["level"] for x in res}
    assert any(x["level"] == "warn" and "HSA_USERPTR_FOR_PAGED_MEM" in x["text"] for x in res)
    assert not any("HSA_USERPTR" in x["text"] for x in PX.check(str(root), env={"HSA_USERPTR_FOR_PAGED_MEM": "0"}))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert PX.check(str(empty), env={})[0]["level"] == "fail"
    nv = m10_server(tmp_path / "nv", uvm=False)
    assert any("nvidia-uvm missing" in x["text"] for x in PX.check(str(nv), env={}))


@pytest.mark.parametrize("vbios,resize,bar,level", [
    ("113-D1631711-100", False, 0, "fail"),        # 16 GB fixed BAR, no resize: cannot fit RK3588's 1 GiB window
    ("113-D1631700-111", True, 0, "warn"),         # resizable: shrink to 256 MiB first
    ("113-D1640200-043", True, 256 << 20, "ok"),   # already shrunk: fits
])
def test_vbios_against_a_small_window(tmp_path, vbios, resize, bar, level):
    root = mi50_board(tmp_path, vbios=vbios, bar=bar, resize=resize)
    cs = VB.cards(str(root))
    assert cs[0]["vbios"] == vbios and cs[0]["model"] == "Instinct MI50"
    bar_v = [v for v in VB.verdicts(cs, 1 << 30) if "BAR0" in v["text"]]
    assert bar_v[0]["level"] == level


def test_vbios_notes_and_mixed_versions(tmp_path):
    root = mi50_board(tmp_path)
    v = VB.verdicts(VB.cards(str(root)))
    assert v[0]["level"] == "warn" and "16 GB" in v[0]["text"]           # the Vulkan 16 GB limit
    other = dict(VB.cards(str(root))[0], pci="0000:02:00.0", vbios="113-D1631700-111")
    v = VB.verdicts(VB.cards(str(root)) + [other])
    assert any(x["pci"] == "*" and "different VBIOS" in x["text"] for x in v)
    unknown = dict(other, vbios="113-XXXX", known=None)
    assert "not in the table" in VB.verdicts([unknown])[0]["text"]


def test_vbios_cli_exit_code(tmp_path, capsys):
    root = mi50_board(tmp_path)
    assert VB.main(["--root", str(root), "--window-mib", "1024"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_arm_profiles_win_on_arm():
    cat = HW.catalogue()
    mi50 = [{"name": "MI50", "vendor": "amd", "backend": "vulkan", "gfx": "gfx906"}]
    assert HW.match(cat, mi50, "aarch64")[0]["profile"]["id"] == "amd-gfx906-arm"
    assert HW.match(cat, mi50, "x86_64")[0]["profile"]["id"] != "amd-gfx906-arm"
    m10 = [{"name": "Tesla M10", "vendor": "nvidia", "backend": "cuda", "compute": 5.0}]
    assert HW.match(cat, m10, "aarch64")[0]["profile"]["id"] == "nvidia-maxwell-arm"
    assert HW.match(cat, m10, "x86_64")[0]["profile"]["id"] == "nvidia-legacy-sm50-70"
    pl = HW.plan(HW.get("nvidia-maxwell-arm"), "inference")
    assert "50-real;52-real" in pl["steps"][0]["cmd"]
