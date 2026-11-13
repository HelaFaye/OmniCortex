"""Inventory of a Proxmox VE cluster: nodes, guests and GPUs (read-only API calls with an API token)."""
import argparse
import json
import ssl
import urllib.request


def get(base, path, token, ctx):
    req = urllib.request.Request(base.rstrip("/") + "/api2/json" + path, headers={"Authorization": f"PVEAPIToken={token}"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
        return json.loads(r.read())["data"]


def inventory(base, token, insecure=False, fetch=get):
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    res = fetch(base, "/cluster/resources", token, ctx)
    nodes = []
    for n in (r for r in res if r.get("type") == "node"):
        name = n["node"]
        try:
            pci = fetch(base, f"/nodes/{name}/hardware/pci", token, ctx)
        except Exception as e:
            pci, err = [], str(e)
        else:
            err = None
        gpus = [{"id": d.get("id"), "name": d.get("device_name") or d.get("device"), "vendor": d.get("vendor_name"),
                 "iommu_group": d.get("iommugroup")} for d in pci if str(d.get("class", "")).startswith("0x03")]
        guests = [{"vmid": g.get("vmid"), "name": g.get("name"), "type": g.get("type"), "status": g.get("status"),
                   "maxmem_gb": round((g.get("maxmem") or 0) / 2**30, 1)}
                  for g in res if g.get("node") == name and g.get("type") in ("qemu", "lxc")]
        nodes.append({"node": name, "status": n.get("status"), "cpu": n.get("cpu"), "maxcpu": n.get("maxcpu"),
                      "mem_free_gb": round(((n.get("maxmem") or 0) - (n.get("mem") or 0)) / 2**30, 1),
                      "gpus": gpus, "guests": guests, **({"pci_error": err} if err else {})})
    return {"nodes": nodes,
            "note": "GPUs shared into LXC containers can serve several guests; one passed through to a VM serves that VM only. "
                    "Run an OmniCortex agent in the guest that has the GPU so Studio can read its free memory."}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    inv = inventory(a.url, a.token, a.insecure)
    with open(a.out, "w") as f:
        json.dump(inv, f, indent=1)
    for n in inv["nodes"]:
        print(f"{n['node']}: {n['status']}, {n['mem_free_gb']} GB free, GPUs: {', '.join(g['name'] or '?' for g in n['gpus']) or 'none'}, "
              f"{len(n['guests'])} guests")


if __name__ == "__main__":
    main()
