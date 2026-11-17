# Clamp-on V-path ray viewer (desktop app).
# Change the numbers on the left, press "Update", and see the beam path.
# Needs: numpy, matplotlib (tkinter comes with Python).
# To make a Windows .exe:   pip install pyinstaller
#                           pyinstaller --onefile --windowed ray_app.py
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# wave speed that carries the signal through the wall [m/s]
MATERIALS = {"Stainless steel (shear)": 3100.0, "Carbon steel (shear)": 3230.0,
             "PVC (longitudinal)": 2380.0, "Custom": None}
# wall thickness [mm] by schedule. PVC pipe has no sanitary, Sch10 or Sch160 sizes.
SIZES = {'2"':   dict(od=60.3,  san=(50.8, 1.65),  metal={10: 2.77, 40: 3.91, 80: 5.54, 160: 8.74},  pvc={40: 3.91, 80: 5.54, 120: 6.35}),
         '2.5"': dict(od=73.0,  san=(63.5, 1.65),  metal={10: 3.05, 40: 5.16, 80: 7.01, 160: 9.53},  pvc={40: 5.16, 80: 7.01, 120: 7.62}),
         '3"':   dict(od=88.9,  san=(76.2, 1.65),  metal={10: 3.05, 40: 5.49, 80: 7.62, 160: 11.13}, pvc={40: 5.49, 80: 7.62, 120: 8.89}),
         '4"':   dict(od=114.3, san=(101.6, 2.11), metal={10: 3.05, 40: 6.02, 80: 8.56, 160: 13.49}, pvc={40: 6.02, 80: 8.56, 120: 11.10})}
MARK = {"san": "D", 10: "^", 40: "o", 80: "s", 120: "P", 160: "X"}      # marker shape per schedule


def presets(material):
    """Pipe presets (OD, wall) that exist for this wall material."""
    out = {}
    for name, z in SIZES.items():
        if "PVC" not in material:
            out[f"{name} sanitary"] = z["san"]
        for sch, wall in (z["pvc"] if "PVC" in material else z["metal"]).items():
            out[f"{name} Sch{sch}"] = (z["od"], wall)
    return out


# name, label, default
FIELDS = [("ang", "Wedge angle [deg]", 33.15), ("c_wedge", "Wedge speed [m/s]", 2500.0),
          ("c_water", "Water speed [m/s]", 1480.0), ("c_wall", "Wall wave speed [m/s]", 3100.0),
          ("OD", "Pipe OD [mm]", 114.3), ("wall", "Wall thickness [mm]", 6.02),
          ("h", "PZT centre distance from pipe [mm]", 10.0), ("L", "PZT length [mm]", 18.0),
          ("S_used", "Spacing used on pipe [mm] (stays fixed)", 78.5)]


def trace(v):
    """Ray path of the beam centre. x = along pipe, z = depth below the pipe outer surface."""
    p = np.sin(np.radians(v["ang"])) / v["c_wedge"]           # Snell: sin(angle)/speed is constant
    if p * v["c_wall"] >= 1 or p * v["c_water"] >= 1:
        return None                                            # total reflection: no beam gets through
    a, a_wall, a_wat = np.radians(v["ang"]), np.arcsin(p * v["c_wall"]), np.arcsin(p * v["c_water"])
    ID = v["OD"] - 2 * v["wall"]
    steps = [(v["h"] * np.tan(a), v["h"]), (v["wall"] * np.tan(a_wall), v["wall"]),
             (ID * np.tan(a_wat), ID), (ID * np.tan(a_wat), -ID),
             (v["wall"] * np.tan(a_wall), -v["wall"]), (v["h"] * np.tan(a), -v["h"])]
    pts = [(0.0, -v["h"])]
    for dx, dz in steps:
        pts.append((pts[-1][0] + dx, pts[-1][1] + dz))
    pts = [(px - pts[1][0], pz) for px, pz in pts]             # origin: beam entry on the pipe outer surface
    names = ["Transmit PZT centre", "Enters pipe (outer surface)", "Enters water",
             "Reflects at far wall", "Leaves water", "Leaves pipe (outer surface)",
             "Reaches receive PZT height"]
    S_ideal = pts[5][0] - pts[1][0]
    S_used = v["S_used"] if v["S_used"] > 0 else S_ideal
    foot = v["L"] / np.cos(a)                                  # beam footprint on the pipe
    miss = S_ideal - S_used                                    # + = beam lands beyond the receiver
    time = 2e-3 * (v["h"] / np.cos(a) / v["c_wedge"] + v["wall"] / np.cos(a_wall) / v["c_wall"]
                   + ID / np.cos(a_wat) / v["c_water"])
    return dict(a=a, a_wall=a_wall, a_wat=a_wat, ID=ID, pts=pts, names=names, S_ideal=S_ideal,
                S_used=S_used, foot=foot, miss=miss, time=time,
                overlap=max(0.0, 1 - abs(miss) / foot), echo=2 * v["wall"] * np.tan(a_wall))


def spacing(OD, wall, c_wall, v):
    p = np.sin(np.radians(v["ang"])) / v["c_wedge"]
    return 2 * wall * np.tan(np.arcsin(p * c_wall)) + 2 * (OD - 2 * wall) * np.tan(np.arcsin(p * v["c_water"]))


def draw_path(fig, v):
    fig.clear()
    ax = fig.add_subplot(111)
    r = trace(v)
    if r is None:
        ax.text(0.5, 0.5, "Total reflection: no beam passes with these speeds and angle",
                ha="center", transform=ax.transAxes)
        return "Total reflection."
    x = [q[0] for q in r["pts"]]; z = [q[1] for q in r["pts"]]
    xt = x[0]                                                  # transmit PZT centre
    x0, x1 = xt - v["L"] - 5, x[1] + max(r["S_used"], r["S_ideal"]) + v["h"] + v["L"] + 5
    ax.axhspan(0, v["wall"], color="0.6"); ax.axhspan(v["wall"], v["wall"] + r["ID"], color="#cfe8ff")
    ax.axhspan(v["wall"] + r["ID"], v["OD"], color="0.6")
    rx_entry = x[1] + r["S_used"]                              # where the receiver expects the beam
    rx_c = rx_entry + v["h"] * np.tan(r["a"])                  # receive PZT centre
    hw = v["L"] / 2 * np.cos(r["a"]); hz = v["L"] / 2 * np.sin(r["a"])
    for xa, xb in ((xt - hw - 3, x[1] + r["foot"] / 2 + 3), (rx_entry - r["foot"] / 2 - 3, rx_c + hw + 3)):
        ax.fill([xa, xb, xb, xa], [0, 0, -v["h"] - hz - 3, -v["h"] - hz - 3], color="#ffe9b0")   # wedges
    ax.plot([xt - hw, xt + hw], [-v["h"] + hz, -v["h"] - hz], "k", lw=4)   # transmit PZT
    ax.plot(0, 0, "ko", ms=5); ax.annotate("0", (0, 0), textcoords="offset points", xytext=(-4, 6), fontsize=9)
    ax.plot([rx_c - hw, rx_c + hw], [-v["h"] - hz, -v["h"] + hz], "k", lw=4)   # receive PZT
    for dx, lw in ((0, 2), (-r["foot"] / 2, 0.7), (r["foot"] / 2, 0.7)):   # centre ray and beam edges
        ax.plot([q + dx for q in x[1:6]], z[1:6], "r", lw=lw)
    ax.plot(x[:2], z[:2], "r", lw=2); ax.plot(x[5:], z[5:], "r", lw=2)
    if "steel" in v["material"].lower():                       # first echo inside the wall leaves later
        ax.plot([x[5] + r["echo"], x[6] + r["echo"]], [0, -v["h"]], "r--", lw=1)
    ax.plot(x[1:6], z[1:6], "ro", ms=4)
    if v.get("show_ideal") and abs(r["miss"]) > 0.05:          # receiver position at the ideal spacing
        e = x[5]; c = e + v["h"] * np.tan(r["a"]); xa, xb, zt = e - r["foot"] / 2 - 3, c + hw + 3, -v["h"] - hz - 3
        ax.plot([xa, xb, xb, xa, xa], [0, 0, zt, zt, 0], color="0.4", ls="--", lw=1)
        ax.plot([c - hw, c + hw], [-v["h"] - hz, -v["h"] + hz], color="0.4", ls="--", lw=2)
        ax.text(xa, zt - 1, "ideal", fontsize=8, color="0.4")
    if v.get("show_angles"):                                   # angle from the surface normal in each layer
        ax.plot([x[1], x[1]], [0, -v["h"]], "k:", lw=1); ax.plot([x[2], x[2]], [v["wall"], v["wall"] + 15], "k:", lw=1)
        ax.text(x[1] - 2, -v["h"] / 2, f"wedge {v['ang']:.1f} deg", ha="right", fontsize=9, weight="bold")
        ax.text(x[2] + 3, v["wall"] / 2, f"wall {np.degrees(r['a_wall']):.1f} deg", va="center", fontsize=9, weight="bold")
        ax.text(x[2] + 3, v["wall"] + 12, f"water {np.degrees(r['a_wat']):.1f} deg", fontsize=9, weight="bold")
    if v.get("show_grid"):
        ax.minorticks_on(); ax.grid(True, which="major", color="k", alpha=0.35); ax.grid(True, which="minor", color="k", alpha=0.12)
    ax.axvline(rx_entry, color="g", ls=":"); ax.axvline(x[5], color="r", ls=":")
    ax.set_xlim(x0, x1); ax.set_ylim(v["OD"] + 3, -v["h"] - hz - 6); ax.set_aspect("equal")
    ax.set_xlabel("along pipe [mm]"); ax.set_ylabel("depth [mm]")
    ax.set_title(f"Beam lands at {x[5]:.1f} mm (red), receiver expects {rx_entry:.1f} mm (green)")
    out = [f"Angles:  wedge {v['ang']:.2f} deg | wall {np.degrees(r['a_wall']):.1f} deg | "
           f"water {np.degrees(r['a_wat']):.1f} deg",
           f"Ideal spacing on pipe {r['S_ideal']:.1f} mm | used {r['S_used']:.1f} mm | "
           f"miss {r['miss']:+.1f} mm",
           f"Beam footprint on pipe {r['foot']:.1f} mm | geometric overlap {100 * r['overlap']:.0f}% "
           f"(rough, ignores beam spreading)",
           f"PZT centre-to-centre {r['S_used'] + 2 * v['h'] * np.tan(r['a']):.1f} mm | "
           f"transit time {r['time'] * 1e6:.1f} us | wall echo shift {r['echo']:.1f} mm", ""]
    out += [f"{n:32s} x = {q[0]:7.1f}   depth = {q[1]:7.1f} mm" for n, q in zip(r["names"], r["pts"])]
    return "\n".join(out)


def draw_charts(fig, v):
    fig.clear()
    t = np.linspace(1.6, 14, 100)
    for k, (name, z) in enumerate(SIZES.items()):
        ax = fig.add_subplot(2, 2, k + 1)
        for label, c, table in (("Stainless steel", 3100.0, z["metal"]), ("PVC", 2380.0, z["pvc"])):
            line, = ax.plot(t, spacing(z["od"], t, c, v), label=label)
            for sch, wall in table.items():
                ax.plot(wall, spacing(z["od"], wall, c, v), MARK[sch], color=line.get_color())
                ax.annotate(f"Sch{sch}", (wall, spacing(z["od"], wall, c, v)), fontsize=6,
                            textcoords="offset points", xytext=(0, 5), ha="center")
        if abs(z["od"] - v["OD"]) < 0.5 and v["S_used"] > 0:
            ax.axhline(v["S_used"], color="g", ls=":", label="spacing used")
        ax.set_title(f"{name} (OD {z['od']} mm)"); ax.grid(True); ax.legend(fontsize=7)
        ax.set_xlabel("wall thickness [mm]"); ax.set_ylabel("ideal spacing [mm]")
    fig.tight_layout()


def draw_readout(fig, v):
    fig.clear()
    ax = fig.add_subplot(111)
    x = np.linspace(0.01, 0.2, 100)
    lines = {}
    for m, c in {"Stainless steel": 3100.0, "PVC": 2380.0, "Selected wall material": v["c_wall"]}.items():
        lines[m], = ax.plot(x, spacing(1.0, x, c, v), label=m, ls="--" if m.startswith("Sel") else "-")
    for name, z in SIZES.items():                             # one marker per standard pipe
        pipes = [("Stainless steel", 3100.0, "san", z["san"][1] / z["san"][0])]
        pipes += [("Stainless steel", 3100.0, sch, w / z["od"]) for sch, w in z["metal"].items()]
        pipes += [("PVC", 2380.0, sch, w / z["od"]) for sch, w in z["pvc"].items()]
        for m, c, sch, ratio in pipes:
            ax.plot(ratio, spacing(1.0, ratio, c, v), MARK[sch], color=lines[m].get_color())
            ax.annotate(name, (ratio, spacing(1.0, ratio, c, v)), fontsize=6, ha="center",
                        textcoords="offset points", xytext=(0, 6 if m != "PVC" else -11))
    for sch, mk in MARK.items():                              # legend entries for the marker shapes
        ax.plot([], [], mk, color="0.3", label="Sanitary" if sch == "san" else f"Sch{sch}")
    ax.plot(v["wall"] / v["OD"], spacing(1.0, v["wall"] / v["OD"], v["c_wall"], v), "k*", ms=12, label="this pipe")
    ax.set_xlabel("wall thickness / OD"); ax.set_ylabel("ideal spacing / OD")
    ax.set_title("Read-out chart: spacing = OD x (value on the vertical axis)")
    ax.minorticks_on(); ax.grid(True, which="both", alpha=0.4); ax.legend(fontsize=7, ncol=2)


if __name__ == "__main__":
    root = tk.Tk(); root.title("Clamp-on V-path ray viewer_ Johar.P")
    left = ttk.Frame(root, padding=6); left.pack(side="left", fill="y")
    entries = {}
    mat = tk.StringVar(value="Stainless steel (shear)")
    preset = tk.StringVar(value='4" Sch40')
    ttk.Label(left, text="Pipe preset").pack(anchor="w")
    pbox = ttk.Combobox(left, textvariable=preset, values=list(presets(mat.get())), state="readonly"); pbox.pack(fill="x")
    ttk.Label(left, text="Wall material").pack(anchor="w")
    box = ttk.Combobox(left, textvariable=mat, values=list(MATERIALS), state="readonly"); box.pack(fill="x")
    for key, label, default in FIELDS:
        ttk.Label(left, text=label).pack(anchor="w")
        e = ttk.Entry(left); e.insert(0, str(default)); e.pack(fill="x"); entries[key] = e
    checks = {}
    for key, label, on in (("auto", "Always use the ideal spacing", False), ("show_angles", "Show angles", True),
                           ("show_grid", "Show grid", False), ("show_ideal", "Show ideal receiver position", True)):
        checks[key] = tk.BooleanVar(value=on)
        ttk.Checkbutton(left, text=label, variable=checks[key], command=lambda: update()).pack(anchor="w")
    info = tk.Text(left, width=62, height=14, font=("Courier", 8)); info.pack(pady=6)

    tabs = ttk.Notebook(root); tabs.pack(side="right", fill="both", expand=True)
    figs = []
    for title in ("Beam path", "Spacing vs wall thickness", "Read-out chart"):
        frame = ttk.Frame(tabs); tabs.add(frame, text=title)
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, frame); canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, frame)
        figs.append((fig, canvas))

    def set_material(_=None):
        if MATERIALS[mat.get()]:
            entries["c_wall"].delete(0, "end"); entries["c_wall"].insert(0, str(MATERIALS[mat.get()]))
        pbox["values"] = list(presets(mat.get()))             # PVC has no sanitary, Sch10 or Sch160
        if preset.get() not in pbox["values"]:
            preset.set("")
        update()

    def update(_=None):
        try:
            v = {k: float(e.get()) for k, e in entries.items()}
        except ValueError:
            return
        v["material"] = mat.get()
        v.update({k: c.get() for k, c in checks.items()})
        if v["auto"]:
            v["S_used"] = 0.0                                  # 0 means: use the ideal spacing
        text = draw_path(figs[0][0], v); draw_charts(figs[1][0], v); draw_readout(figs[2][0], v)
        info.delete("1.0", "end"); info.insert("1.0", text)
        for _, canvas in figs:
            canvas.draw()

    def set_preset(_=None):
        for key, val in zip(("OD", "wall"), presets(mat.get())[preset.get()]):
            entries[key].delete(0, "end"); entries[key].insert(0, str(val))
        update()

    pbox.bind("<<ComboboxSelected>>", set_preset)
    box.bind("<<ComboboxSelected>>", set_material)
    ttk.Button(left, text="Update", command=update).pack(fill="x")
    root.bind("<Return>", update)
    update()
    root.mainloop()
