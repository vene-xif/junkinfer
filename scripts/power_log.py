#!/usr/bin/env python3
"""System power / state logger for diagnosing sudden power-offs (systemd: scripts/ji-power-log.service, root).

Every INTERVAL s (0.5) one line to logs/power.log, written with fsync so the last lines survive a power loss:
  per GPU: power (W, hwmon power1_input), temperature, shader / memory clock, vddgfx (mV), fan rpm
  CPU: package power (W, RAPL energy delta), Tctl, mean core clock, load average
  sum of GPU + CPU package power (the rest of the box - board, RAM, disks, fans - is not measured)
  board sensor chip voltages (it87 / it8792 if loaded), available RAM, model process alive + RSS
The previous boot's log is kept as logs/power.prev.log; the current one is rotated at 200 MB (power.old.log).
A sudden power-off shows as the log ending mid-load with no shutdown lines in `journalctl -b -1`.
"""
import glob, os, time

INTERVAL = float(os.environ.get("JI_POWER_LOG_INTERVAL", "0.5"))
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG = os.path.join(LOGDIR, "power.log")
MAX_BYTES = 200 << 20


def rd(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def gpus():
    out = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]")):
        hw = sorted(glob.glob(card + "/device/hwmon/hwmon*"))
        if hw:
            out.append((os.path.basename(card), hw[0], os.path.basename(os.path.realpath(card + "/device"))))
    return out


def board_chips():
    chips = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = rd(h + "/name", "")
        if name.startswith("it8") or name.startswith("nct"):
            ins = sorted(glob.glob(h + "/in*_input"), key=lambda p: int(os.path.basename(p)[2:-6]))
            chips.append((name, ins))
    return chips


def k10temp():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(h + "/name") == "k10temp":
            return h + "/temp1_input"
    return None


def model_proc():
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(p, "rb") as f:
                a0 = f.read().split(b"\0")[0]
        except OSError:
            continue
        if a0.endswith((b"bin/llama-server", b"bin/llama-junk-ref", b"bin/llama-completion")):
            pid = p.split("/")[2]
            for line in (rd("/proc/%s/status" % pid, "") or "").splitlines():
                if line.startswith("VmRSS:"):
                    return "%s:%dM" % (pid, int(line.split()[1]) // 1024)
            return pid
    return "-"


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    if os.path.exists(LOG):
        os.replace(LOG, os.path.join(LOGDIR, "power.prev.log"))
    G = gpus()
    chips = board_chips()
    tctl = k10temp()
    rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"
    rapl_max = int(rd("/sys/class/powercap/intel-rapl:0/max_energy_range_uj", "0") or 0)
    e0, t0 = int(rd(rapl, "0") or 0), time.monotonic()
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    hdr = "# fields: time | per GPU name(pci): W C sclk/mclk MHz vddgfx mV fan | cpu: pkgW Tctl MHz load | gpu+cpu W | board volts (mV) | MemAvail | model\n"
    os.write(fd, hdr.encode() + ("# GPUs: %s chips: %s\n" % ([g[2] for g in G], [c[0] for c in chips])).encode())
    os.fsync(fd)
    n = 0
    while True:
        time.sleep(INTERVAL)
        parts = [time.strftime("%F %T") + ".%d" % (int(time.time() * 10) % 10)]
        gsum = 0.0
        for name, h, pci in G:
            w = int(rd(h + "/power1_input", "0") or 0) / 1e6
            gsum += w
            parts.append("%s %.0fW %sC %s/%sMHz %smV %srpm" % (
                name, w, int(rd(h + "/temp1_input", "0") or 0) // 1000,
                int(rd(h + "/freq1_input", "0") or 0) // 1000000, int(rd(h + "/freq2_input", "0") or 0) // 1000000,
                rd(h + "/in0_input", "?"), rd(h + "/fan1_input", "?")))
        e1, t1 = int(rd(rapl, "0") or 0), time.monotonic()
        de = e1 - e0 if e1 >= e0 else e1 + rapl_max - e0
        cpu_w = de / 1e6 / max(t1 - t0, 1e-3) if e1 else 0.0
        e0, t0 = e1, t1
        mhz = [int(rd(p, "0") or 0) for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")]
        parts.append("cpu %.0fW %sC %dMHz load %s" % (cpu_w, int(rd(tctl, "0") or 0) // 1000 if tctl else "?",
                                                      (sum(mhz) // len(mhz) // 1000) if mhz else 0,
                                                      (rd("/proc/loadavg", "?") or "?").split()[0]))
        parts.append("sum %.0fW" % (gsum + cpu_w))
        for cname, ins in chips:
            parts.append(cname + " " + ",".join(rd(p, "?") for p in ins))
        mem = [l for l in (rd("/proc/meminfo", "") or "").splitlines() if l.startswith("MemAvailable")]
        parts.append("memavail %dM" % (int(mem[0].split()[1]) // 1024 if mem else -1))
        parts.append("model " + model_proc())
        os.write(fd, (" | ".join(parts) + "\n").encode())
        os.fsync(fd)
        n += 1
        if n % 600 == 0 and os.fstat(fd).st_size > MAX_BYTES:
            os.close(fd)
            os.replace(LOG, os.path.join(LOGDIR, "power.old.log"))
            fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)


if __name__ == "__main__":
    main()
