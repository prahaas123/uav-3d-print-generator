import argparse
import math
import os
import time
import numpy as np
import cadquery as cq
from OCP.gp import gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

# CONFIGURATION
CONFIG = dict(
    wing=dict(
        root_chord=100.0, tip_chord=50.0,
        semispan=349.5,              # root (centreline) to tip, measured along the span
        le_sweep=3.27,               # deg
        dihedral=2.0,                # deg
        twist=0.0,                   # deg at tip, linear, about c/4 (+ = nose up)
        airfoil="naca2412",          # "nacaXXXX" or a Selig .dat path
        origin=(0.0, 0.0, 0.0),      # root LE
        splits=[0.0, 0.5, 1.0],      # panel breaks as fraction of semispan (2 or 3 panels)
        lattice_pitch=22.0,          # perpendicular spacing between parallel slots
        lattice_angle=45.0,          # slot angle to the span axis
    ),
    htail=dict(
        root_chord=45.0, tip_chord=45.0, semispan=80.0, le_sweep=0.0, dihedral=0.0,
        twist=0.0, airfoil="naca0008", origin=(367.0, 0.0, 0.0),
        lattice_pitch=15.0, lattice_angle=45.0,
    ),
    vtail=dict(
        root_chord=45.0, tip_chord=45.0, span=80.0, le_sweep=0.0, twist=0.0,
        airfoil="naca0008", origin=(367.0, 0.0, 0.0),
        lattice_pitch=15.0, lattice_angle=45.0,
    ),

    # ---- carbon rods (diameters are the rods, bores get ROD_CLEARANCE added) ----------
    wing_spar=dict(
        diameter=5.0,
        mode="continuous",           # "continuous": one straight rod through both wings
                                     # "per_side": one rod per side following the dihedral
        chord_fraction=None,         # None = root max-thickness position
        max_semi_length=None,        # cap on the rod's half length (None = as far as it fits)
    ),
    wing_pins=dict(diameter=2.0, length=30.0, rear_fraction=0.60),  # panel-joint alignment pins
    ht_rod=dict(diameter=1.5, chord_fraction=0.22),
    vt_rod=dict(diameter=1.5, chord_fraction=0.45),
    boom=dict(diameter=8.0, socket_fuselage=40.0, socket_mount=30.0),

    # ---- tail mount ---------------------------------------------------------------------
    mount=dict(width=14.0, nose_ahead=15.0, wall=1.6),

    # ---- fuselage (pod under the wing; nose + wingbox) -------------------------------------
    # stations: (x, width, height, z_top). Under the wing z_top sits 1 mm above the chord plane and the wing is subtracted, which leaves a saddle that matches the wing.
    # The last station is replaced by the boom-socket end automatically.
    fuselage=dict(
        stations=[(-110.0, 30.0, 30.0, -4.0),
                  (-70.0, 44.0, 44.0, 0.0),
                  (-20.0, 46.0, 46.0, 1.0),
                  (30.0, 46.0, 46.0, 1.0),
                  (90.0, 42.0, 40.0, 1.0),
                  (150.0, 0.0, 0.0, 0.0)],
        squareness=2.6,              # superellipse exponent (2 = ellipse, higher = boxier)
        wall=1.0,
        split_x=-20.0,               # nose | wingbox joint
        spigot_length=12.0, spigot_wall=1.0, spigot_clearance=0.15,
        firewall=3.0,
        motor_center_hole=8.0, motor_bolt=3.2, motor_bolt_spacing=(16.0, 19.0),
        dowel_diameter=3.2, dowel_x=(-8.0, 108.0), dowel_below_top=8.0,
        socket_wall=1.4,
    ),

    # ---- printing ---------------------------------------------------------------------------
    print=dict(
        line_width=0.45,             # vase-mode line width you will slice with
        slot_width=0.10,             # slot cut width (v5 used ~0.1)
        slot_gap=0.60,               # neck between top and bottom slots
        bore_keepout=0.60,           # material kept between lattice slots and any bore
        min_wall=0.6,                # skin material required above/below a rod bore
        te_thickness=0.8,            # blunt TE so it survives slicing
        rod_clearance=0.15,          # added to rod diameter for bores
        density_vase=0.55,           # g/cm^3, effective foamed LW-PLA in vase mode
        density_solid=0.75,          # g/cm^3, LW-PLA for normal-print parts
        stl_tolerance=0.02, stl_angular=0.15,
    ),
)

# helpers
WARNINGS = []
TIP_MARGIN = 5.0   # rods stop this far short of a free tip so the tip face stays closed

def warn(msg):
    WARNINGS.append(msg)
    print("  WARNING:", msg)

def V(p):
    return cq.Vector(float(p[0]), float(p[1]), float(p[2]))

def xform(shape, T):
    """Rigid transform of a CadQuery shape by a 4x4 numpy matrix."""
    t = gp_Trsf()
    t.SetValues(*[float(v) for v in np.asarray(T)[:3, :4].ravel()])
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, t, True).Shape())

def rot_x(deg):
    a = math.radians(deg)
    T = np.eye(4)
    T[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return T

def translate(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T

def apply(T, pts):
    pts = np.atleast_2d(pts)
    return (np.c_[pts, np.ones(len(pts))] @ np.asarray(T).T)[:, :3]

def oriented_box(origin, u, v, lu, lv, lw):
    """Box with one corner region: spans u in [0,lu], v in [0,lv], w=u x v in [-lw/2,lw/2]."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) - np.dot(v, u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = u, v, w, origin
    return xform(cq.Solid.makeBox(lu, lv, lw, cq.Vector(0, 0, -lw / 2)), T)

def cylinder(p0, p1, d):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    return cq.Solid.makeCylinder(d / 2, float(np.linalg.norm(ax)), V(p0), V(ax))

def biggest_solid(shape):
    s = shape.Solids()
    return max(s, key=lambda x: x.Volume()) if s else shape

# airfoils
N_AF = 70  # points per surface

def _cosine(n):
    return 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))

def naca4(code, n=N_AF):
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = _cosine(n)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m == 0 or p == 0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
    # resample both surfaces on the common cosine x grid
    return x, np.interp(x, xu, zu), np.interp(x, xl, zl)

def selig(path, n=N_AF):
    pts = []
    with open(path) as f:
        for line in f:
            try:
                a, b = map(float, line.split()[:2])
                pts.append((a, b))
            except ValueError:
                continue
    pts = np.array(pts)
    i = int(np.argmin(pts[:, 0]))
    up, lo = pts[:i + 1][::-1], pts[i:]
    x0, x1 = pts[i, 0], max(up[-1, 0], lo[-1, 0])
    x = _cosine(n)
    xs = x0 + x * (x1 - x0)
    zu = np.interp(xs, up[:, 0], up[:, 1])
    zl = np.interp(xs, lo[:, 0], lo[:, 1])
    c = x1 - x0
    return x, (zu - pts[i, 1]) / c, (zl - pts[i, 1]) / c

def load_airfoil(spec):
    s = spec.strip().lower()
    if s.startswith("naca") and len(s) == 8 and s[4:].isdigit():
        return naca4(s[4:])
    return selig(spec)

# lifting surface
class Surface:
    """Straight-tapered lifting surface built in a local frame (x chord, s span, z thickness)
    and placed by G = translate(origin) @ rot_x(dihedral). Vertical tail: dihedral = 90."""

    def __init__(self, name, root_chord, tip_chord, span, le_sweep, twist, airfoil, origin,
                 dihedral, te_thickness):
        self.name = name
        self.cr, self.ct, self.b = root_chord, tip_chord, span
        self.tan_le = math.tan(math.radians(le_sweep))
        self.twist_tip = twist
        self.xn, self.zu_n, self.zl_n = load_airfoil(airfoil)
        self.te = te_thickness
        self.G = translate(origin) @ rot_x(dihedral)
        self.Ginv = np.linalg.inv(self.G)
        self.R = self.G[:3, :3]

    # planform -----------------------------------------------------------------------
    def chord(self, s):
        return self.cr + (self.ct - self.cr) * s / self.b

    def x_le(self, s):
        return s * self.tan_le

    def twist(self, s):
        return self.twist_tip * s / self.b

    def section(self, s):
        """local (x, z) of upper and lower surface at station s (same x-ordering, LE first)."""
        c, xle = self.chord(s), self.x_le(s)
        te_n = self.te / c
        zu = self.zu_n + self.xn * te_n / 2
        zl = self.zl_n - self.xn * te_n / 2
        x = xle + self.xn * c
        U, L = np.c_[x, zu * c], np.c_[x, zl * c]
        a = math.radians(self.twist(s))
        if a:
            px = xle + 0.25 * c
            Rm = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
            U = (U - [px, 0]) @ Rm.T + [px, 0]
            L = (L - [px, 0]) @ Rm.T + [px, 0]
        return U, L

    def surfaces_at(self, s, x):
        """local z of (lower, upper) at local chordwise x (nan outside the chord)."""
        U, L = self.section(s)
        if x < U[0, 0] or x > U[-1, 0]:
            return float("nan"), float("nan")
        return float(np.interp(x, L[:, 0], L[:, 1])), float(np.interp(x, U[:, 0], U[:, 1]))

    def camber_at(self, s, x):
        lo, up = self.surfaces_at(s, x)
        return 0.5 * (lo + up)

    def max_thickness_fraction(self):
        t = self.zu_n - self.zl_n
        return float(self.xn[int(np.argmax(t))])

    # geometry ---------------------------------------------------------------------------
    def to_global(self, pts):
        return apply(self.G, pts)

    def _stations(self, s0, s1):
        if self.twist_tip == 0:
            return [s0, s1]
        n = max(2, int(math.ceil((s1 - s0) / 20.0)) + 1)
        return list(np.linspace(s0, s1, n))

    def section_wire(self, s):
        U, L = self.section(s)
        ring = np.r_[U[::-1], L[1:]]                    # TE upper -> LE -> TE lower
        pts = self.to_global(np.c_[ring[:, 0], np.full(len(ring), s), ring[:, 1]])
        spline = cq.Edge.makeSpline([V(p) for p in pts])
        close = cq.Edge.makeLine(V(pts[-1]), V(pts[0]))
        return cq.Wire.assembleEdges([spline, close])

    def solid(self, s0, s1):
        return cq.Solid.makeLoft([self.section_wire(s) for s in self._stations(s0, s1)], True)

    def lattice_lines(self, s0, s1, pitch, angle):
        """(point, direction) of the +/-angle slot lines in local (x, s); phase anchored at root LE
        so the slots line up across panel joints."""
        a = math.radians(angle)
        xs = [self.x_le(s0), self.x_le(s1), self.x_le(s0) + self.chord(s0), self.x_le(s1) + self.chord(s1)]
        cx, cs = 0.5 * (min(xs) + max(xs)), 0.5 * (s0 + s1)
        half = math.hypot(max(xs) - min(xs), s1 - s0)
        lines = []
        for sgn in (1, -1):
            d = np.array([sgn * math.sin(a), math.cos(a)])
            n = np.array([d[1], -d[0]])
            ref = np.dot([self.x_le(0), 0.0], n)
            k0 = int(math.floor((np.dot([cx, cs], n) - half - ref) / pitch))
            k1 = int(math.ceil((np.dot([cx, cs], n) + half - ref) / pitch))
            for k in range(k0, k1 + 1):
                foot = np.array([cx, cs]) + n * (ref + k * pitch - np.dot([cx, cs], n))
                lines.append((foot, d, n))
        return lines

    def slot_cutters(self, s0, s1, pitch, angle, width, gap, bores, keep, step=1.0):
        """Thin prisms (global) for the vase-mode slots of panel [s0, s1].
        Top slot: from outside the upper skin down to camber + gap/2; bottom slot: from outside the
        lower skin up to camber - gap/2. Around every bore the slot stops `keep` short of the bore
        (top slots above it, bottom slots below it), so no slot can ever seal off a void."""
        loc = []
        for b in bores:
            a0, a1 = apply(self.Ginv, [b.p0, b.p1])
            loc.append((a0, a1, b.d / 2 + keep, keep))
        cutters = []
        for foot, d, n in self.lattice_lines(s0, s1, pitch, angle):
            # t-range where the line is inside the (slightly grown) panel planform
            ts = np.arange(-600, 600, 0.5)
            P2 = foot[None, :] + ts[:, None] * d[None, :]
            ok = ((P2[:, 1] >= s0 - 0.3) & (P2[:, 1] <= s1 + 0.3) &
                  (P2[:, 0] >= self.x_le(P2[:, 1]) - 0.3) &
                  (P2[:, 0] <= self.x_le(P2[:, 1]) + self.chord(P2[:, 1]) + 0.3))
            if ok.sum() < 3:
                continue
            t_in = ts[ok]
            t_lo, t_hi = t_in[0], t_in[-1]
            n_pts = max(3, int(math.ceil((t_hi - t_lo) / step)) + 1)
            T = np.linspace(t_lo, t_hi, n_pts)
            # refine near bores
            extra = []
            for a0, a1, R, m in loc:
                for t in T:
                    q = foot + t * d
                    if self._bore_dist(q, a0, a1, m)[0] < R + step:
                        extra.extend(np.linspace(t - step, t + step, 9))
            if extra:
                T = np.unique(np.clip(np.r_[T, extra], t_lo, t_hi))
            zb_top, zb_bot, z_hi, z_lo = [], [], -1e9, 1e9
            for t in T:
                x, s = foot + t * d
                sc = min(max(s, 0.0), self.b)
                U, L = self.section(sc)
                xc = min(max(x, U[0, 0]), U[-1, 0])
                up = np.interp(xc, U[:, 0], U[:, 1])
                lo = np.interp(xc, L[:, 0], L[:, 1])
                cam = 0.5 * (up + lo)
                zt, zb = cam + gap / 2, cam - gap / 2
                for a0, a1, R, m in loc:
                    dist, z_ax = self._bore_dist(np.array([x, s]), a0, a1, m)
                    if dist < R:
                        h = math.sqrt(R * R - dist * dist)
                        zt = max(zt, z_ax + h)
                        zb = min(zb, z_ax - h)
                zb_top.append(zt)
                zb_bot.append(zb)
                z_hi, z_lo = max(z_hi, up), min(z_lo, lo)
            for bottom_edge, cap in ((zb_top, z_hi + 2.0), (zb_bot, z_lo - 2.0)):
                e = np.clip(bottom_edge, z_lo - 1.0, z_hi + 1.0)
                if cap > 0 and np.all(e >= z_hi - 1e-6):
                    continue
                if cap < 0 and np.all(e <= z_lo + 1e-6):
                    continue
                ring = [(t, z) for t, z in zip(T, e)] + [(T[-1], cap), (T[0], cap)]
                pts = np.array([[foot[0] + t * d[0], foot[1] + t * d[1], z] for t, z in ring])
                pts = pts - np.array([n[0], n[1], 0.0]) * width / 2
                g = apply(self.G, pts)
                normal = self.R @ np.array([n[0], n[1], 0.0]) * width
                wire = cq.Wire.makePolygon([V(p) for p in g], close=True)
                cutters.append(cq.Solid.extrudeLinear(cq.Face.makeFromWires(wire), V(normal)))
        return cutters

    @staticmethod
    def _bore_dist(q, a0, a1, m):
        """planform distance from local (x, s) point q to a bore axis, and the axis z there."""
        A, B = a0[[0, 1]], a1[[0, 1]]
        D = B - A
        L2 = float(np.dot(D, D))
        L = math.sqrt(L2)
        lam = float(np.dot(q - A, D) / L2)
        lam_c = min(max(lam, -m / L), 1 + m / L)
        p = A + lam_c * D
        return float(np.linalg.norm(q - p)), float(a0[2] + lam_c * (a1[2] - a0[2]))

# rods
class Bore:
    def __init__(self, name, p0, p1, rod_d, clearance, slit_down=None):
        self.name, self.p0, self.p1 = name, np.asarray(p0, float), np.asarray(p1, float)
        self.rod_d, self.d = rod_d, rod_d + clearance
        self.slit_down = slit_down

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    def solid(self):
        return cylinder(self.p0, self.p1, self.d)

    def keepout(self, margin):
        ax = (self.p1 - self.p0) / self.length
        return cylinder(self.p0 - ax * margin, self.p1 + ax * margin, self.d + 2 * margin)

    def slit(self, surf, width, depth=40.0, overrun=1.0):
        """Thin cut from the bore axis straight 'down' (panel -z) through the bottom skin.
        Built in the panel frame with its ends on constant-span planes (= print layers) and
        running `overrun` past the bore ends, so no layer can see a sealed sliver."""
        a0, a1 = apply(surf.Ginv, [self.p0, self.p1])
        if a1[1] < a0[1]:
            a0, a1 = a1, a0
        ds = a1[1] - a0[1]
        if ds < 1e-6:
            return None
        dirn = (a1 - a0) / ds                      # per unit span
        b0, b1 = a0 - dirn * overrun, a1 + dirn * overrun
        quad = [b0, b1, b1 - [0, 0, depth], b0 - [0, 0, depth]]
        nrm = np.cross(a1 - a0, [0, 0, 1.0])
        nrm /= np.linalg.norm(nrm)
        quad = [q - nrm * width / 2 for q in quad]
        g = apply(surf.G, np.array(quad))
        face = cq.Face.makeFromWires(cq.Wire.makePolygon([V(p) for p in g], close=True))
        return cq.Solid.extrudeLinear(face, V(surf.R @ nrm * width))

def fits(surf, s, x, z, r, m):
    """True if a bore of radius r centred at local (x, z) on station s keeps m of skin."""
    lows, ups = [], []
    for dx in (-r, 0.0, r):
        lo, up = surf.surfaces_at(s, x + dx)
        if math.isnan(lo):
            return False
        lows.append(lo)
        ups.append(up)
    return (z - r - m >= max(lows)) and (z + r + m <= min(ups))

def solve_continuous_rod(surf, rod_d, clearance, min_wall, fraction, cap, step=1.0):
    """Straight rod along global y at fixed (x, z), symmetric about the centreline.
    Picks z to maximise how far it reaches inside the (dihedralled) surface."""
    r = (rod_d + clearance) / 2
    f = surf.max_thickness_fraction() if fraction is None else fraction
    x_g = surf.to_global([[surf.x_le(0) + f * surf.cr, 0, 0]])[0, 0]
    lo0, up0 = surf.surfaces_at(0.0, surf.x_le(0) + f * surf.cr)
    z0 = surf.to_global([[0, 0, 0]])[0, 2]
    best = (-1, None)
    cands = np.arange(lo0 + r + min_wall, up0 - r - min_wall + 1e-9, 0.05)
    cands = sorted(cands, key=lambda z: abs(z - 0.5 * (lo0 + up0)))   # prefer centred on ties
    for zl in cands:
        z_g = z0 + zl
        y = 0.0
        while True:
            pl = apply(surf.Ginv, [[x_g, y + step, z_g]])[0]
            if pl[1] > surf.b - TIP_MARGIN or not fits(surf, pl[1], pl[0], pl[2], r, min_wall):
                break
            y += step
        if y > best[0] + 1e-9:
            best = (y, z_g)
    reach, z_g = best
    if z_g is None:
        return None, f
    semi = reach if cap is None else min(reach, cap)
    return (x_g, z_g, semi), f

def solve_local_rod(surf, rod_d, clearance, min_wall, fraction, s_start=0.0, cap=None, step=1.0):
    """Rod along a constant-chord-fraction line in the surface's own frame, on the camber line."""
    r = (rod_d + clearance) / 2
    f = surf.max_thickness_fraction() if fraction is None else fraction

    def pt(s):
        x = surf.x_le(s) + f * surf.chord(s)
        return np.array([x, s, surf.camber_at(max(s, 0.0), x)])

    s = max(s_start, 0.0)
    if not fits(surf, s, *pt(s)[[0, 2]], r, min_wall):
        return None, f
    while s + step <= surf.b - TIP_MARGIN:
        p = pt(s + step)
        if not fits(surf, s + step, p[0], p[2], r, min_wall):
            break
        s += step
    if cap is not None:
        s = min(s, s_start + cap)
    p0, p1 = pt(s_start), pt(s)
    return surf.to_global([p0, p1]), f

# vase-mode panel
def vase_panel(surf, s0, s1, pitch, angle, bores, P):
    body = surf.solid(s0, s1)
    tools = surf.slot_cutters(s0, s1, pitch, angle, P["slot_width"], P["slot_gap"], bores, P["bore_keepout"])
    for b in bores:
        tools.append(b.solid())
        sl = b.slit(surf, P["slot_width"])
        if sl is not None:
            tools.append(sl)
    body = body.cut(*tools)
    return biggest_solid(body.clean())

# STL cleanup + vase check (need trimesh; skipped without it)
def clean_stl(path):
    """Drop zero-area triangles and close tessellation cracks that OCC sometimes leaves."""
    try:
        import trimesh
        import networkx  # noqa: F401  trimesh's hole filling needs it
    except ImportError:
        return None
    m = trimesh.load(path)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: p.area)
    m.export(path)
    return m.is_watertight

# vase check
def vase_check(stl_path, layer=0.2):
    try:
        import trimesh
    except ImportError:
        return "skipped (pip install trimesh)"
    m = trimesh.load(stl_path)
    zmax = m.bounds[1, 2]
    bad = []
    for z in np.arange(layer * 3, zmax - layer * 2, layer):      # skip solid bottom layers
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            bad.append((z, "empty"))
            continue
        pl, _ = sec.to_2D()
        polys = pl.polygons_full
        holes = sum(len(p.interiors) for p in polys)
        if len(polys) != 1 or holes:
            bad.append((round(float(z), 2), f"{len(polys)} loops, {holes} holes"))
    if not bad:
        return "OK (single contour on every layer)"
    return f"FAIL on {len(bad)} layers, e.g. {bad[:4]}"

# build
def superellipse_wire(x, w, h, zc, n, k=40):
    t = np.linspace(0, 2 * math.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    y = (w / 2) * np.sign(c) * np.abs(c) ** (2 / n)
    z = zc + (h / 2) * np.sign(s) * np.abs(s) ** (2 / n)
    return cq.Wire.assembleEdges([cq.Edge.makeSpline([V((x, a, b)) for a, b in zip(y, z)], periodic=True)])


def build(cfg):
    P = cfg["print"]
    clr = P["rod_clearance"]
    parts = {}      # name -> dict(shape=global solid, print=4x4 or callable, kind="vase"/"solid")
    info = []

    # ---------------- wing ----------------
    W = cfg["wing"]
    wing = Surface("wing", W["root_chord"], W["tip_chord"], W["semispan"], W["le_sweep"],
                   W["twist"], W["airfoil"], W["origin"], W["dihedral"], P["te_thickness"])
    sp = cfg["wing_spar"]
    bores = []
    if sp["mode"] == "continuous":
        res, f = solve_continuous_rod(wing, sp["diameter"], clr, P["min_wall"], sp["chord_fraction"],
                                      sp["max_semi_length"])
        if res is None:
            raise SystemExit("Wing spar does not fit at the root. Use a smaller diameter.")
        x_r, z_r, semi = res
        bores.append(Bore("wing_spar", [x_r, 0, z_r], [x_r, semi, z_r], sp["diameter"], clr))
        rod_reach_s = semi
        info.append(f"Wing spar (continuous): {sp['diameter']:.1f} mm rod, length {2 * semi:.0f} mm, "
                    f"at x={x_r:.1f}, z={z_r:.2f} (root chord fraction {f:.2f}), centred on the centreline")
        if semi < 0.6 * wing.b and W["dihedral"]:
            info.append(f"  (dihedral {W['dihedral']} deg limits a straight rod to {semi:.0f} mm per side; "
                        "use per_side mode or a thinner rod to reach further)")
    else:
        res, f = solve_local_rod(wing, sp["diameter"], clr, P["min_wall"], sp["chord_fraction"],
                                 cap=sp["max_semi_length"])
        if res is None:
            raise SystemExit("Wing spar does not fit at the root. Use a smaller diameter.")
        bores.append(Bore("wing_spar", res[0], res[1], sp["diameter"], clr))
        rod_reach_s = float(np.linalg.norm(res[1] - res[0]))
        info.append(f"Wing spar (per side): 2 x {sp['diameter']:.1f} mm rods, {rod_reach_s:.0f} mm each, "
                    f"chord fraction {f:.2f}. They meet at the centreline (the wingbox saddle carries the joint)")

    # joint pins
    pins = cfg["wing_pins"]
    splits = [W["splits"][i] * wing.b for i in range(len(W["splits"]))]
    for j, sj in enumerate(splits[:-1]):
        want = [("rear", pins["rear_fraction"])]
        if rod_reach_s < sj + pins["length"] / 2 + 5:
            want.append(("front", wing.max_thickness_fraction()))
        for tag, fr in want:
            r = (pins["diameter"] + clr) / 2
            if sj == 0.0:
                if tag == "front":
                    continue
                x = wing.x_le(0) + fr * wing.cr
                z = wing.to_global([[0, 0, wing.camber_at(0, x)]])[0, 2]
                ok = fits(wing, 0.0, x, wing.camber_at(0, x), r, P["min_wall"])
                p0, p1 = [x + W["origin"][0], 0, z], [x + W["origin"][0], pins["length"] / 2, z]
            else:
                pts = []
                ok = True
                for s in (sj - pins["length"] / 2, sj + pins["length"] / 2):
                    xx = wing.x_le(s) + fr * wing.chord(s)
                    zz = wing.camber_at(s, xx)
                    ok &= fits(wing, s, xx, zz, r, P["min_wall"])
                    pts.append([xx, s, zz])
                p0, p1 = wing.to_global(pts)
            if not ok:
                warn(f"{tag} alignment pin at span {sj:.0f} mm does not fit ({pins['diameter']} mm), skipped")
                continue
            bores.append(Bore(f"pin_{tag}_{j}", p0, p1, pins["diameter"], clr))
            info.append(f"Alignment pin ({tag}) at wing joint {sj:.0f} mm: {pins['diameter']} x "
                        f"{pins['length']:.0f} mm rod")

    for i, (s0, s1) in enumerate(zip(splits[:-1], splits[1:])):
        t0 = time.time()
        name = f"wing_R{i + 1}"
        print(f"building {name} ...", flush=True)
        body = vase_panel(wing, s0, s1, W["lattice_pitch"], W["lattice_angle"], bores, P)
        tip_down = (i == len(splits) - 2)
        parts[name] = dict(shape=body, kind="vase", frame=wing, s0=s0, s1=s1, tip_down=tip_down)
        parts[f"wing_L{i + 1}"] = dict(shape=body.mirror("XZ"), kind="vase", mirror_of=name)
        print(f"   {time.time() - t0:.1f}s")

    # ---------------- tail ----------------
    H, Vt, Mt = cfg["htail"], cfg["vtail"], cfg["mount"]
    ht = Surface("ht", H["root_chord"], H["tip_chord"], H["semispan"], H["le_sweep"], H["twist"],
                 H["airfoil"], H["origin"], H["dihedral"], P["te_thickness"])
    vt = Surface("vt", Vt["root_chord"], Vt["tip_chord"], Vt["span"], Vt["le_sweep"], Vt["twist"],
                 Vt["airfoil"], Vt["origin"], 90.0, P["te_thickness"])
    half_w = Mt["width"] / 2
    ht_root_lo, ht_root_up = ht.surfaces_at(0, ht.x_le(0) + ht.max_thickness_fraction() * ht.cr)
    z_ht = H["origin"][2]
    boom = cfg["boom"]
    r_boom = (boom["diameter"] + clr) / 2
    z_boom = z_ht + ht_root_lo - Mt["wall"] - r_boom
    s_stub = (ht_root_up - 0.0) + 1.0                       # VT stub height above HT chord plane

    hr = cfg["ht_rod"]
    res, f = solve_continuous_rod(ht, hr["diameter"], clr, P["min_wall"], hr["chord_fraction"], None)
    tail_bores = []
    if res is None:
        warn("HT rod does not fit, HT has no rod")
    else:
        x_r, z_r, semi = res
        tail_bores.append(Bore("ht_rod", [x_r, -semi, z_r], [x_r, semi, z_r], hr["diameter"], clr))
        info.append(f"HT rod: {hr['diameter']} mm, length {2 * semi:.0f} mm, through mount and both halves "
                    f"(x={x_r:.1f}, z={z_r:.2f})")
    vr = cfg["vt_rod"]
    s_bottom = z_boom + r_boom + Mt["wall"] - z_ht           # VT rod bottom, stays above the boom bore
    s_bottom = max(s_bottom, ht_root_lo)
    res, f = solve_local_rod(vt, vr["diameter"], clr, P["min_wall"], vr["chord_fraction"], s_start=s_stub)
    if res is None:
        warn("VT rod does not fit, VT has no rod")
    else:
        p_top = res[1]
        p_bot = res[0].copy()
        p_bot[2] = z_ht + s_bottom
        tail_bores.append(Bore("vt_rod", p_bot, p_top, vr["diameter"], clr))
        info.append(f"VT rod: {vr['diameter']} mm, length {np.linalg.norm(p_top - p_bot):.0f} mm, "
                    f"from the mount up into the VT")

    print("building tail ...", flush=True)
    ht_body = vase_panel(ht, half_w, ht.b, H["lattice_pitch"], H["lattice_angle"],
                         [b for b in tail_bores if b.name == "ht_rod"], P)
    parts["ht_R"] = dict(shape=ht_body, kind="vase", frame=ht, s0=half_w, s1=ht.b, tip_down=True)
    parts["ht_L"] = dict(shape=ht_body.mirror("XZ"), kind="vase", mirror_of="ht_R")
    vt_body = vase_panel(vt, s_stub, vt.b, Vt["lattice_pitch"], Vt["lattice_angle"],
                         [b for b in tail_bores if b.name == "vt_rod"], P)
    parts["vt"] = dict(shape=vt_body, kind="vase", frame=vt, s0=s_stub, s1=vt.b, tip_down=True)

    # mount
    x_le_ht = H["origin"][0]
    c_ht = ht.cr
    x_front = x_le_ht - Mt["nose_ahead"]
    r_tube = r_boom + Mt["wall"]
    centre = ht.solid(0, half_w).fuse(ht.solid(0, half_w).mirror("XZ"))
    stub = vt.solid(0, s_stub)
    tube = cylinder([x_front, 0, z_boom], [x_le_ht + 0.8 * c_ht, 0, z_boom], 2 * r_tube)
    web = cq.Solid.makeBox(0.65 * c_ht, 2 * r_tube, z_ht - z_boom,
                           V((x_le_ht + 0.12 * c_ht, -r_tube, z_boom)))
    mount = centre.fuse(stub, tube, web).clean()
    boom_bore = cylinder([x_front - 1, 0, z_boom], [x_front + boom["socket_mount"], 0, z_boom], 2 * r_boom)
    mount = biggest_solid(mount.cut(boom_bore, *[b.solid() for b in tail_bores]).clean())
    parts["tail_mount"] = dict(shape=mount, kind="solid", print_axis="x")

    # ---------------- fuselage ----------------
    F = cfg["fuselage"]
    st = [list(s) for s in F["stations"]]
    tube_d = 2 * (r_boom + F["socket_wall"])
    st[-1] = [st[-1][0], tube_d, tube_d, z_boom + tube_d / 2]
    xs = np.array([s[0] for s in st])
    if not any(abs(xs - F["split_x"]) < 1e-6):
        row = [F["split_x"]] + [float(np.interp(F["split_x"], xs, [s[k] for s in st])) for k in (1, 2, 3)]
        st.append(row)
        st.sort(key=lambda r: r[0])
    t = F["wall"]
    n = F["squareness"]
    outer = cq.Solid.makeLoft([superellipse_wire(x, w, h, zt - h / 2, n) for x, w, h, zt in st], False)
    inner_st = [list(s) for s in st]
    inner_st[0][0] += F["firewall"]
    inner_st[-1][0] -= t
    inner = cq.Solid.makeLoft([superellipse_wire(x, max(w - 2 * t, 0.5), max(h - 2 * t, 0.5), zt - h / 2, n)
                               for x, w, h, zt in inner_st], False)
    # whole wing as ONE loft (tip-root-tip) so the cutter has no internal root faces
    w_r = [wing.section_wire(s) for s in wing._stations(0, wing.b)]
    w_l = [w.mirror("XZ") for w in reversed(w_r[1:])]
    wing_full = cq.Solid.makeLoft(w_l + w_r, True)
    inner = inner.cut(wing_full.translate(cq.Vector(0, 0, -t)))
    shell = outer.cut(wing_full).cut(inner)

    # boom socket: tube + 45 deg funnel so it prints with the wingbox nose-down
    x_end = st[-1][0]
    r_to = r_boom + F["socket_wall"]
    x_meet = x_end - boom["socket_fuselage"]
    Rb = 40.0
    x_a = x_meet - Rb + r_to
    cone = cq.Solid.makeCone(Rb, 0.0, Rb, V((x_a, 0, z_boom)), cq.Vector(1, 0, 0))
    funnel = cone.translate(cq.Vector(t * math.sqrt(2), 0, 0)).cut(cone)
    sock = cylinder([x_meet - 2, 0, z_boom], [x_end, 0, z_boom], 2 * r_to)
    outer_cut = outer.cut(wing_full)
    fus = shell.fuse(funnel.intersect(outer_cut), sock.intersect(outer_cut)).clean()
    fus = fus.cut(cylinder([x_end + 1, 0, z_boom], [x_meet, 0, z_boom], 2 * r_boom))
    # rubber-band dowels: transverse holes a quarter of the local height below the top
    xs_st = [s[0] for s in st]
    for xd in F["dowel_x"]:
        ztop = float(np.interp(xd, xs_st, [s[3] for s in st]))
        hgt = float(np.interp(xd, xs_st, [s[2] for s in st]))
        zd = ztop - max(F["dowel_below_top"], 0.25 * hgt)
        fus = fus.cut(cylinder([xd, -60, zd], [xd, 60, zd], F["dowel_diameter"]))
        info.append(f"Rubber-band dowel: {F['dowel_diameter'] - 0.2:.0f} mm rod across the wingbox at x={xd:.0f}, z={zd:.1f}")
    # firewall holes
    x0, w0, h0, zt0 = st[0]
    zc = zt0 - h0 / 2
    hole_tools = [cylinder([x0 - 1, 0, zc], [x0 + F["firewall"] + 1, 0, zc], F["motor_center_hole"])]
    a, b2 = F["motor_bolt_spacing"]
    for dy, dz in ((a / 2, 0), (-a / 2, 0), (0, b2 / 2), (0, -b2 / 2)):
        hole_tools.append(cylinder([x0 - 1, dy, zc + dz], [x0 + F["firewall"] + 1, dy, zc + dz], F["motor_bolt"]))
    fus = fus.cut(*hole_tools)

    xs_ = F["split_x"]
    big = 1000
    nose = fus.intersect(cq.Solid.makeBox(big, big, big, V((xs_ - big, -big / 2, -big / 2))))
    wbox = fus.intersect(cq.Solid.makeBox(big, big, big, V((xs_, -big / 2, -big / 2))))
    # spigot on the nose, slides into the wingbox
    row = [r for r in inner_st if abs(r[0] - xs_) < 1e-6][0]
    w_in, h_in = row[1] - 2 * t, row[2] - 2 * t
    zc_in = row[3] - row[2] / 2
    c_sp = F["spigot_clearance"]
    sw = F["spigot_wall"]
    ring_in = superellipse_wire(xs_ - 4.0, w_in - 2 * c_sp - 2 * sw, h_in - 2 * c_sp - 2 * sw, zc_in, n)
    attach = cq.Solid.extrudeLinear(   # overlaps the nose wall by 0.3 mm so it fuses
        cq.Face.makeFromWires(superellipse_wire(xs_ - 4.0, w_in + 0.6, h_in + 0.6, zc_in, n), [ring_in]),
        cq.Vector(4.0, 0, 0))
    ring_in2 = superellipse_wire(xs_, w_in - 2 * c_sp - 2 * sw, h_in - 2 * c_sp - 2 * sw, zc_in, n)
    spig = cq.Solid.extrudeLinear(
        cq.Face.makeFromWires(superellipse_wire(xs_, w_in - 2 * c_sp, h_in - 2 * c_sp, zc_in, n), [ring_in2]),
        cq.Vector(F["spigot_length"], 0, 0))
    nose = biggest_solid(nose.fuse(attach.intersect(outer), spig).clean())
    parts["nose"] = dict(shape=nose, kind="solid", print_axis="x")
    parts["wingbox"] = dict(shape=biggest_solid(wbox.clean()), kind="solid", print_axis="x")

    rods = dict(boom=(boom["diameter"],
                      (x_front + boom["socket_mount"]) - (x_end - boom["socket_fuselage"])))
    info.append(f"Tail boom: {boom['diameter']} mm rod, length {rods['boom'][1]:.0f} mm "
                f"(horizontal at z={z_boom:.1f}), {boom['socket_fuselage']:.0f} mm into the wingbox, "
                f"{boom['socket_mount']:.0f} mm into the tail mount")
    rod_list = []
    for b in bores:
        if abs(b.p0[1]) < 1e-6 and np.allclose(b.p0[[0, 2]], b.p1[[0, 2]]):   # straight across centreline
            rod_list.append((b.name, b.p1 * [1, -1, 1], b.p1, b.rod_d))
        else:
            rod_list.append((b.name + "_R", b.p0, b.p1, b.rod_d))
            rod_list.append((b.name + "_L", b.p0 * [1, -1, 1], b.p1 * [1, -1, 1], b.rod_d))
    for b in tail_bores:
        rod_list.append((b.name, b.p0, b.p1, b.rod_d))
    rod_list.append(("boom", [x_end - boom["socket_fuselage"], 0, z_boom],
                     [x_front + boom["socket_mount"], 0, z_boom], boom["diameter"]))
    for name, p0, p1, d in rod_list:
        parts["rod_" + name] = dict(shape=cylinder(p0, p1, d), kind="rod")
    return parts, info, dict(wing=wing, ht=ht, vt=vt, bores=bores + tail_bores, z_boom=z_boom)

# print orientation
def print_matrix(part, parts):
    """4x4 that maps the part from aircraft coordinates to its print pose."""
    if "frame" in part:
        surf, s0, s1 = part["frame"], part["s0"], part["s1"]
        if part["tip_down"]:     # local (x, s, z) -> (x, z, s1 - s)
            Pm = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, s1], [0, 0, 0, 1.0]])
        else:                    # local (x, s, z) -> (x, -z, s - s0)
            Pm = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, -s0], [0, 0, 0, 1.0]])
        return Pm @ surf.Ginv
    # x-axis parts: front face on the bed, x -> z
    return np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])

def to_print(name, parts):
    part = parts[name]
    if "mirror_of" in part:
        src = to_print(part["mirror_of"], parts)
        return src.mirror("XZ")
    shp = xform(part["shape"], print_matrix(part, parts))
    bb = shp.BoundingBox()
    return shp.translate(cq.Vector(-0.5 * (bb.xmin + bb.xmax), -0.5 * (bb.ymin + bb.ymax), -bb.zmin))

# preview
def preview(parts, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return
    tris = []
    for name, p in parts.items():
        v, f = p["shape"].tessellate(0.3, 0.3)
        v = np.array([[q.x, q.y, q.z] for q in v])
        if len(f):
            tris.append((name, v[np.array(f)]))
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for name, T in tris:
        col = "#222222" if name.startswith("rod_") else "#c8832a" if name.startswith(("nose", "wingbox", "tail_mount")) else "#4a7ab5"
        ax.add_collection3d(Poly3DCollection(T, facecolor=col, edgecolor="none", alpha=0.95))
    allv = np.concatenate([T.reshape(-1, 3) for _, T in tris])
    lo, hi = allv.min(0), allv.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo); ax.view_init(25, -125); ax.set_title("Isometric")
    for k, (i, j, title) in enumerate([(1, 0, "Top"), (1, 2, "Front"), (0, 2, "Side")]):
        a = fig.add_subplot(2, 2, k + 2)
        for name, T in tris:
            col = "#222222" if name.startswith("rod_") else "#c8832a" if name.startswith(("nose", "wingbox", "tail_mount")) else "#4a7ab5"
            for tri in T:
                a.fill(tri[:, i], tri[:, j], color=col, lw=0, alpha=0.5)
        a.set_aspect("equal"); a.set_title(title); a.grid(alpha=0.3)
        if title == "Top":
            a.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=110)

# main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rc_parts"))
    ap.add_argument("--check", action="store_true", help="slice every vase part and verify one contour per layer")
    ap.add_argument("--no-stl", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    P = CONFIG["print"]
    if not (P["line_width"] < P["slot_gap"] < 2 * P["line_width"]):
        warn("slot_gap should sit between line_width and 2*line_width or the slots will not fuse / will split the layer")

    t0 = time.time()
    parts, info, geo = build(CONFIG)
    os.makedirs(os.path.join(args.out, "step"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "stl"), exist_ok=True)

    lines = ["RC plane print parts", "=" * 60, ""]
    lines += info + [""]
    lines.append(f"{'part':<12}{'print':<34}{'mass g':>8}{'  vase check'}")
    total = 0.0
    assy = cq.Assembly(name="aircraft")
    for name, part in parts.items():
        shp = part["shape"]
        if part["kind"] == "rod":
            assy.add(shp, name=name, color=cq.Color(0.1, 0.1, 0.1))
            continue
        if not shp.isValid():
            warn(f"{name}: OCC reports the solid as invalid; check it in SolidWorks")
        cq.exporters.export(shp, os.path.join(args.out, "step", f"{name}.step"))
        col = cq.Color(0.29, 0.48, 0.71) if part["kind"] == "vase" else cq.Color(0.78, 0.51, 0.16)
        assy.add(shp, name=name, color=col)
        if part["kind"] == "vase":
            faces = shp.Faces()
            area = sum(f.Area() for f in faces)
            caps = sum(f.Area() for f in faces if f.geomType() == "PLANE" and f.Area() > 50)
            mass = (area - caps) * P["line_width"] * P["density_vase"] / 1000 + caps * 0.6 * P["density_vase"] / 1000
            how = "vase mode, " + ("tip on bed" if parts.get(part.get("mirror_of", name), part).get("tip_down") else "root on bed")
        else:
            mass = shp.Volume() * P["density_solid"] / 1000
            how = "normal, front face on bed"
        total += mass
        chk = ""
        if not args.no_stl:
            pth = os.path.join(args.out, "stl", f"{name}.stl")
            cq.exporters.export(to_print(name, parts), pth, tolerance=P["stl_tolerance"],
                                angularTolerance=P["stl_angular"])
            if clean_stl(pth) is False:
                warn(f"{name}.stl is not watertight after cleanup; let the slicer repair it")
            if args.check and part["kind"] == "vase":
                chk = vase_check(pth)
        lines.append(f"{name:<12}{how:<34}{mass:8.1f}  {chk}")
        print(f"  {name:<12}{how:<34}{mass:7.1f} g  {chk}", flush=True)
    assy.export(os.path.join(args.out, "step", "assembly.step"))
    lines.append(f"{'TOTAL printed':<46}{total:8.1f}  (rods, electronics not included)")
    lines += ["", "Slicer: spiral vase / spiralize for vase parts, line width "
              f"{P['line_width']} mm, 3-4 bottom layers, no top layers.",
              "Wing panels: inboard panels print root-down (closed face at the joint, open tip),",
              "the tip panel prints tip-down. Tail halves and VT print tip-down (open root against the mount)."]
    if WARNINGS:
        lines += ["", "WARNINGS"] + ["  " + w for w in WARNINGS]
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if args.preview:
        preview(parts, os.path.join(args.out, "preview.png"))
    print("\n".join(info))
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out}")

if __name__ == "__main__":
    main()