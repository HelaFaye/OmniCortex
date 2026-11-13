# Proxmox: OmniCortex in a container, with the host's GPUs

The Proxmox backend runs OmniCortex on a Proxmox VE host, inside an LXC container that has the host's GPUs. The
container can run either the remote agent (Studio elsewhere uses this machine) or a whole Studio.

The harness supports legacy and current hardware alike. The examples below are an AMD Instinct MI50 and an NVIDIA
Tesla M10 on an RK3588 board, the hardest case. The same steps work for any AMD or NVIDIA GPU on any Proxmox host.

> **Status:** the plans are tested against simulated hosts, not on real hardware. The MI50 and M10 guides it
> builds on (mi50-arm and m10-arm) are themselves untested on their cards.

## Why a container and not a VM

To give a VM a GPU, Proxmox hands the card to VFIO. The kernel allows that only when the IOMMU reports cache
coherency (`IOMMU_CAP_CACHE_COHERENCY`), and refuses it otherwise. Boards whose PCIe is not cache-coherent, such as
RK3588 and the Raspberry Pi 5, can never pass a GPU to a VM, whatever the Proxmox build or VM option.

A container shares the host's kernel, so it needs no VFIO and has no virtualisation overhead. The GPU driver runs
on the host, and the container gets the device nodes:

| GPU | The host runs | The container gets | The container installs |
|---|---|---|---|
| AMD | amdgpu (and KFD for ROCm) | `/dev/kfd`, `/dev/dri/renderD*` | Mesa Vulkan; ROCm mounted read-only from the host |
| NVIDIA | NVIDIA's kernel module | `/dev/nvidia0…N`, `nvidiactl`, `nvidia-uvm`, `nvidia-uvm-tools` | the same driver version's user space, `--no-kernel-module` |

On x86 servers with the IOMMU on, a VM with VFIO also works. The container is still simpler, and lets the host and
other containers share the card.

## Use

On the Proxmox host:

```bash
python scripts/proxmox_lxc.py host                       # GPUs, device nodes, coherency, fast cores
python scripts/amd_vbios.py                              # MI50/MI60: what the VBIOS allows
python scripts/proxmox_lxc.py plan --ctid 101 --role agent --models /srv/models --rocm /opt/rocm
python scripts/proxmox_lxc.py apply --ctid 101 --role agent --models /srv/models --rocm /opt/rocm --yes
```

`plan` prints everything and changes nothing:
- the `pct set` commands for the device nodes, the ROCm mount and the models folder;
- a cpuset line for `/etc/pve/lxc/<id>.conf`, which keeps the container on the fastest cores (on RK3588, the A76
  cores 4–7, found from each core's capacity);
- the environment the container needs;
- the commands to run inside it.

`apply --yes` runs the `pct` commands and adds the cpuset line. It looks up the container's `render` group id itself
(`pct exec <id> -- getent group render`). Device passthrough needs `root@pam`.

The same plan is in Studio under Jobs → Setup → **Proxmox: run OmniCortex in a container with the GPU**.

Inside the container, after installing OmniCortex:

```bash
python scripts/proxmox_lxc.py check          # can this user open the GPU, is the environment right
python scripts/hw_profiles.py match          # the hardware profile for this GPU
```

Then start the remote agent and pair it from Studio (Remote), or start Studio itself.

### Roles

- **agent** (default): the remote agent on port 47871. Studio on another computer uses this machine's GPU through
  pairing, like any other remote host. This is the usual choice for a headless Proxmox box.
- **studio**: a whole Studio on port 7870, bound to all interfaces. A non-loopback bind requires Studio's token and
  TLS.

## AMD: the MI50 and other Vega 20 cards

- **amdgpu must bind on the host.** Don't set `vfio-pci.ids=`, and don't blacklist amdgpu. `plan` warns when an AMD
  GPU is bound to another driver.
- **ROCm on a non-coherent host** needs `HSA_USERPTR_FOR_PAGED_MEM=0`. The plan sets it, and `check` warns when it is
  missing.
- **Vulkan first.** llama.cpp's Vulkan backend runs on Mesa and needs no ROCm. The hardware profile
  `amd-gfx906-arm` uses it for inference on ARM hosts.

### VBIOS and the BAR

`scripts/amd_vbios.py` reads each card's VBIOS from sysfs and compares it with a community comparison of MI50
32 GB VBIOS ROMs:

| VBIOS | Card | Resizable BAR | BAR | What it means |
|---|---|---|---|---|
| 113-D1631711-100 | AMD MI50 32 GB (2021) | no | 16 GB | Vulkan sees only 16 GB; on RK3588 the BAR cannot be placed at all |
| 113-D1631700-111 | AMD MI50 32 GB (2020) | yes | 32 GB | recommended without display output |
| 113-D1640200-043 | Radeon Pro V420 32 GB | yes | 32 GB | display output, 178 W cap, lowest idle power |
| 113-D163A1XT-045 | Apple Radeon Pro VII 32 GB | yes | 32 GB | PCIe Gen 3 only, higher idle power |

That comparison was tested on x86, with AMD's original PCB. Some cards use a simplified PCB, with 240 TMUs and 60 ROPs
instead of 256 and 64 (`amdgpu_top -d` shows which). The VBIOSes were not tested on it.

RK3588's PCIe 3.0 x4 controller has a 1 GiB 64-bit window (768 MiB in edk2-rk3588's ACPI mode, with at most 512 MiB
for one BAR). The MI50's BAR0 has to be shrunk to 256 MiB before amdgpu binds, which needs resizable BAR. The GPU
still uses all its VRAM through a small BAR. So on RK3588:
- a card on `113-D1631711-100` cannot work as it is. The check says **FAIL** and names the VBIOSes that can.
- a card with resizable BAR gets a **WARN** with the resize command.
- a BAR that already fits is **OK**.

Cards running different VBIOS versions in one machine have caused ROCm failures in that testing, so the check flags
a mix.

The check only reads. Reflashing a VBIOS can brick a card. If you decide to, do it on an x86 machine with
`amdvbflash`, keep a backup of the original ROM, and match the memory size.

## NVIDIA: the Tesla M10 and other cards

- **Matching versions.** The container's user-space driver must be exactly the host's version. The plan reads the
  version from `/proc/driver/nvidia/version` and prints the installer line with `--no-kernel-module`.
- **The uvm nodes.** `/dev/nvidia-uvm` appears only after something on the host uses CUDA. Run
  `nvidia-modprobe -u -c=0` on the host at boot (a systemd unit) before the container starts. The plan passes the
  nodes anyway and says so.
- **The M10 is four GPUs on one card.** Every `/dev/nvidiaN` is passed. llama.cpp splits a model across them with
  `--split-mode layer`; vLLM does not support Maxwell.
- **On RK3588**, Maxwell needs NVIDIA's proprietary module, patched for non-coherent PCIe (the m10-arm guide:
  driver 580.95.05 aarch64, CUDA 12.9). The hardware profile `nvidia-maxwell-arm` builds llama.cpp for compute 5.0
  and 5.2.

## Which Proxmox on RK3588

- **Official Proxmox VE 9.2 arm64:**
  - It needs UEFI, so edk2-rk3588 firmware.
  - Single-board computers are not supported, and the installer has failed on some boards. Installing Debian 13
    first and adding Proxmox on top is the fallback.
  - Its kernel has to be rebuilt with the non-coherent GTT patch.
- **The community port (PXVirt) on Armbian:** keeps your own patched Armbian kernel.
- **For the M10:** its patched module has only been built against Linux 6.18, and the official arm64 kernel is
  7.0, so PXVirt on Armbian with a 6.18 kernel is the safer route for now. m10-arm's `docs/PROXMOX.md` has the
  host and container steps.

The mi50-arm guide (`docs/PROXMOX.md` there) has the steps for both. OmniCortex starts once the host's driver works.

## What is not done

- Real-hardware runs: none yet, on either card.
- VMs: this backend does not create VMs or set up VFIO. Where VFIO works, Proxmox's own GPU passthrough guide
  applies, and OmniCortex installs in the VM as on any computer.
- Creating the container: make it in Proxmox first (an arm64 or amd64 Debian or Ubuntu template, unprivileged).
  The plan configures an existing container.
