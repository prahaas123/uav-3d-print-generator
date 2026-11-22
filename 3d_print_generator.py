import argparse
import math
import os
import time
import numpy as np
import cadquery as cq
from OCP.gp import gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

CONFIG = dict(
    wing=dict(
        root_chord=102.1, tip_chord=102.1 * 0.5,
        semispan=700.0 / 2,          # VSP TotalSpan / 2, measured along the span
        le_sweep=2.89,               # deg
        dihedral=2.5,                # deg
        twist=0.0,                   # deg at tip, linear, about c/4 (+ = nose up)
        incidence=2.0,               # deg, VSP Y_Rel_Rotation about the root LE
        airfoil=r"C:\Users\kprah\Desktop\Prahaas\WatArrow\VSP Automation\Airfoils\goe322.dat",  # "nacaXXXX" or a Selig/Lednicer .dat (relative = next to this script)
        origin=(0.0, 0.0, 0.0),      # root LE
        splits=[0.0, 0.5, 1.0],      # panel breaks as fraction of semispan (2 or 3 panels)
        lattice_pitch=22.0,          # perpendicular spacing between parallel slots
        lattice_angle=45.0,          # slot angle to the span axis
    ),
    htail=dict(
        root_chord=45.66, tip_chord=45.66, semispan=159.81 / 2, le_sweep=0.0, dihedral=0.0,
        twist=0.0, incidence=1.1531, airfoil="naca0012", origin=(366.29, 0.0, 0.0),   # x = l_H
        lattice_pitch=15.0, lattice_angle=45.0,
    ),
    vtail=dict(
        root_chord=45.66, tip_chord=45.66 * 1.0, span=80.0, le_sweep=0.0, twist=0.0,
        airfoil="naca0012", origin=(366.29, 0.0, 0.0),
        lattice_pitch=15.0, lattice_angle=45.0,
    ),
    x_cg=32.71,                      # target CG aft of wing root LE (for the balance report)

    # ---- control surfaces (fixed layout) -----------------------------------------------------
    controls=dict(
        aileron=dict(hinge_fraction=0.75, hinge_side="top", throw=20.0),
        elevator=dict(hinge_fraction=0.70, hinge_side="top", throw=20.0),
        rudder=dict(hinge_fraction=0.70, hinge_side="top", throw=25.0),
        neck=0.60,                   # hinge skin left at the V apex (line_width < neck < 2*line_width)
        flat=1.0,                    # flat at the V apex so the toolpath U-turns instead of spiking
        margin=5.0,                  # deg added to the throw when sizing the V
        end_gap=1.0,                 # spanwise clearance at the surface end next to fixed structure
    ),
    # ---- aileron servo tray pockets (bottom skin of each aileron panel; trays not generated) ----
    servo_pocket=dict(
        span_offset=45.0,            # pocket centre, mm outboard of the aileron panel's inboard end
        chord_len=24.0,              # tray body footprint along the chord   <- set to your tray
        span_len=14.0,               # tray body footprint along the span    <- set to your tray
        depth=7.0,                   # tray body depth above the bottom skin <- set to your tray
        chord_fraction=None,         # None = centred in the free space between spar and hinge groove
    ),

    # ---- carbon rods: one wing spar + the tail boom (bores get rod_clearance added) ------------
    wing_spar=dict(
        diameter=5.0,                # one straight rod through both wings, centred on the centreline
        chord_fraction=None,         # None = root max-thickness position
        max_semi_length=None,        # cap on the rod's half length (None = as far as it fits)
    ),
    boom=dict(diameter=8.0, socket_fuselage=40.0, socket_mount=30.0),

    # ---- tail mount ---------------------------------------------------------------------
    mount=dict(width=14.0, nose_ahead=15.0, wall=1.6),
    tail_servos=dict(
        length=23.0,                 # case length along x (tab to tab direction)
        width=12.2,                  # case thickness across y
        tab_span=32.5,               # length over the mounting tabs
        tab_height=16.0,             # case bottom to the underside of the tabs (depth in the slot)
        clearance=0.3,               # added to length and width
        wall=1.6,                    # block walls (also between slot and boom bore)
    ),

    # ---- fuselage (pod under the wing; nose + wingbox) -------------------------------------
    # stations: (x, width, height, z_top). Under the wing z_top sits 1 mm above the chord
    # plane and the wing is subtracted, which leaves a saddle that matches the wing.
    # The last station is replaced by the boom-socket end automatically.
    fuselage=dict(
        stations=[(-110.0, 30.0, 30.0, -4.0),
                  (-70.0, 44.0, 44.0, 0.0),
                  (-20.0, 46.0, 46.0, 1.0),
                  (30.0, 46.0, 46.0, 1.0),
                  (90.0, 42.0, 40.0, 1.0),
                  (106.0, 36.0, 34.0, -3.0),   # just aft of the wing TE (incidence drops the TE)
                  (150.0, 0.0, 0.0, 0.0)],
        squareness=2.6,              # superellipse exponent (2 = ellipse, higher = boxier)
        wall=1.0,
        split_x=-20.0,               # nose | wingbox joint
        spigot_length=15.0, spigot_wall=1.0, spigot_clearance=0.15,
        firewall=3.0,
        motor_center_hole=8.0, motor_bolt=3.2, motor_bolt_spacing=(16.0, 19.0),
        dowel_diameter=3.2, dowel_x=(-8.0, 106.0), dowel_below_top=8.0,
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
        density_rod=1.55,            # g/cm^3, solid CF rod (boom: use a tube and the report over-estimates)
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

def rot_y(deg):
    """+deg = nose up (TE down), as VSP Y_Rel_Rotation."""
    a = math.radians(deg)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
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

def read_dat(path, n=N_AF):
    """Selig (TE->upper->LE->lower->TE) or Lednicer (counts line, upper LE->TE, lower LE->TE)."""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                a, b = map(float, line.split()[:2])
                rows.append((a, b))
            except ValueError:
                continue
    if rows and rows[0][0] > 1.5:                                  # Lednicer
        nu, nl = int(rows[0][0]), int(rows[0][1])
        up, lo = np.array(rows[1:1 + nu]), np.array(rows[1 + nu:1 + nu + nl])
    else:                                                          # Selig
        pts = np.array(rows)
        i = int(np.argmin(pts[:, 0]))
        up, lo = pts[:i + 1][::-1], pts[i:]
    x0 = min(up[0, 0], lo[0, 0])
    z0 = 0.5 * (up[0, 1] + lo[0, 1])
    c = max(up[-1, 0], lo[-1, 0]) - x0
    x = _cosine(n)
    zu = (np.interp(x0 + x * c, up[:, 0], up[:, 1]) - z0) / c
    zl = (np.interp(x0 + x * c, lo[:, 0], lo[:, 1]) - z0) / c
    return x, zu, zl

def load_airfoil(spec):
    s = spec.strip().lower()
    if s.startswith("naca") and len(s) == 8 and s[4:].isdigit():
        return naca4(s[4:])
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (spec, os.path.join(here, spec), os.path.join(os.getcwd(), spec)):
        if os.path.isfile(p):
            return read_dat(p)
    raise SystemExit(f"Airfoil file not found: {spec} (looked next to the script and in the working folder)")

# lifting surface
class Surface:
    """Straight-tapered lifting surface built in a local frame (x chord, s span, z thickness)
    and placed by G = translate(origin) @ rot_x(dihedral). Vertical tail: dihedral = 90."""

    def __init__(self, name, root_chord, tip_chord, span, le_sweep, twist, airfoil, origin,
                 dihedral, te_thickness, incidence=0.0):
        self.name = name
        self.cr, self.ct, self.b = root_chord, tip_chord, span
        self.tan_le = math.tan(math.radians(le_sweep))
        self.twist_tip = twist
        self.xn, self.zu_n, self.zl_n = load_airfoil(airfoil)
        self.te = te_thickness
        # VSP order: incidence (Y_Rel_Rotation) about the root LE, dihedral within the wing
        self.G = translate(origin) @ rot_y(incidence) @ rot_x(dihedral)
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
        # thicken the TE only up to te_thickness (the airfoil may already have a blunt TE)
        te_n = max(0.0, self.te / c - (self.zu_n[-1] - self.zl_n[-1]))
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

    def slot_cutters(self, s0, s1, pitch, angle, width, gap, bores, keep, blocks=(), step=1.0):
        """Thin prisms (global) for the vase-mode slots of panel [s0, s1].
        Top slot: from outside the upper skin down to camber + gap/2; bottom slot: from outside the
        lower skin up to camber - gap/2. Around every bore the slot stops `keep` short of the bore
        (top slots above it, bottom slots below it), so no slot can ever seal off a void.
        `blocks`: callables blocked(x, s) in local planform coords; no slot at all where any is True
        (hinge grooves, servo pockets: a slot there would cut the layer into two contours)."""
        def blk(q):
            return any(f(q[0], q[1]) for f in blocks)
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
            if blocks:   # put sample pairs right at every blocked-region boundary (near-vertical slot ends)
                flags = [blk(foot + t * d) for t in T]
                add = []
                for i in range(len(T) - 1):
                    if flags[i] != flags[i + 1]:
                        a, b = T[i], T[i + 1]
                        for _ in range(16):
                            m_ = 0.5 * (a + b)
                            if blk(foot + m_ * d) == flags[i]:
                                a = m_
                            else:
                                b = m_
                        add += [a, b]
                if add:
                    T = np.unique(np.r_[T, add])
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
                if blocks and blk((x, s)):
                    zt, zb = 1e9, -1e9          # clipped to just outside the skin below = no cut
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
    def __init__(self, name, p0, p1, rod_d, clearance):
        self.name, self.p0, self.p1 = name, np.asarray(p0, float), np.asarray(p1, float)
        self.rod_d, self.d = rod_d, rod_d + clearance

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    def solid(self):
        return cylinder(self.p0, self.p1, self.d)

    def keepout(self, margin):
        ax = (self.p1 - self.p0) / self.length
        return cylinder(self.p0 - ax * margin, self.p1 + ax * margin, self.d + 2 * margin)

    def slit(self, surf, width, overrun, depth=40.0):
        """Thin cut from the TOP of the bore straight 'down' (panel -z) through the bottom skin.
        Built in the panel frame with its ends on constant-span planes (= print layers), running
        `overrun` (= the bore keep-out) past the bore ends. Starting at the bore top means a layer
        that only clips the tilted end of a bore still sees that sliver joined to the outside."""
        a0, a1 = apply(surf.Ginv, [self.p0, self.p1])
        if a1[1] < a0[1]:
            a0, a1 = a1, a0
        ds = a1[1] - a0[1]
        if ds < 1e-6:
            return None
        dirn = (a1 - a0) / ds                      # per unit span
        b0, b1 = a0 - dirn * overrun, a1 + dirn * overrun
        top = np.array([0, 0, self.d / 2])
        quad = [b0 + top, b1 + top, b1 - [0, 0, depth], b0 - [0, 0, depth]]
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
    x_f = surf.x_le(0) + f * surf.cr
    lo0, up0 = surf.surfaces_at(0.0, x_f)
    best = (-1, None)
    cands = np.arange(lo0 + r + min_wall, up0 - r - min_wall + 1e-9, 0.05)
    cands = sorted(cands, key=lambda z: abs(z - 0.5 * (lo0 + up0)))   # prefer centred on ties
    for zl in cands:
        x_g, _, z_g = surf.to_global([[x_f, 0, zl]])[0]
        y = 0.0
        while True:
            pl = apply(surf.Ginv, [[x_g, y + step, z_g]])[0]
            if pl[1] > surf.b - TIP_MARGIN or not fits(surf, pl[1], pl[0], pl[2], r, min_wall):
                break
            y += step
        if y > best[0] + 1e-9:
            best = (y, (x_g, z_g))
    reach, xz = best
    if xz is None:
        return None, f
    x_g, z_g = xz
    semi = reach if cap is None else min(reach, cap)
    return (x_g, z_g, semi), f

# control surfaces
def local_loft(surf, stations, poly_fn):
    """Ruled loft through closed polygons given in the surface's local (x, z) at each span station."""
    wires = []
    for s in stations:
        pts = surf.to_global([[x, s, z] for x, z in poly_fn(s)])
        wires.append(cq.Wire.makePolygon([V(p) for p in pts], close=True))
    return cq.Solid.makeLoft(wires, True)


class Hinge:
    """Print-in-place skin hinge over the whole panel [s0, s1] at a constant chord fraction
    (straight on a straight-tapered panel). V-groove from the free skin up to `neck` short of the
    hinge skin, flat apex of width `flat`, included angle throw + margin."""

    def __init__(self, name, surf, s0, s1, fraction, side, throw, C, P):
        self.name, self.surf, self.s0, self.s1, self.f = name, surf, s0, s1, fraction
        self.sgn = 1 if side == "top" else -1          # +1: the hinge skin is the upper skin
        self.throw = throw
        self.vee = throw + C["margin"]
        self.tan = math.tan(math.radians(self.vee) / 2)
        self.neck, self.flat, self.gap = C["neck"], C["flat"], C["end_gap"]
        self.keep = P["bore_keepout"]

    def x_h(self, s):
        return self.surf.x_le(s) + self.f * self.surf.chord(s)

    def skins(self, s):
        return self.surf.surfaces_at(s, self.x_h(s))   # (lower, upper)

    def apex(self, s):
        """(z of the V flat, z of the hinge-skin outer surface) = the hinge axis height."""
        lo, up = self.skins(s)
        return (up - self.neck, up) if self.sgn > 0 else (lo + self.neck, lo)

    def opening(self, s):
        """half-width of the groove where it leaves the free skin"""
        lo, up = self.skins(s)
        free = lo if self.sgn > 0 else up
        return self.flat / 2 + abs(self.apex(s)[0] - free) * self.tan

    def poly(self, s):
        lo, up = self.skins(s)
        za = self.apex(s)[0]
        zf = lo - 3.0 if self.sgn > 0 else up + 3.0
        D = abs(za - zf)
        xh, w = self.x_h(s), self.flat / 2
        return [(xh - w, za), (xh + w, za), (xh + w + D * self.tan, zf), (xh - w - D * self.tan, zf)]

    def cutter(self):
        return local_loft(self.surf, self.surf._stations(self.s0 - 1.0, self.s1 + 1.0), self.poly)

    def blocked(self, x, s):
        return abs(x - self.x_h(s)) < self.opening(s) + self.keep

    def end_trim(self, s_end, direction):
        """Remove everything aft of the hinge line over end_gap from s_end into the panel
        (direction +1 = towards larger s). Always at the top of the print, so always printable."""
        sa, sb = sorted((s_end - direction * 1.0, s_end + direction * self.gap))
        surf = self.surf

        def poly(s):
            xh, xt = self.x_h(s), surf.x_le(s) + surf.chord(s) + 5.0
            return [(xh, -50.0), (xt, -50.0), (xt, 50.0), (xh, 50.0)]
        return local_loft(surf, [sa, sb], poly)

    def chord_aft(self, s):
        return (1 - self.f) * self.surf.chord(s)

    def sweep(self, s, pts_xz, th):
        """rotate local (x, z) points about the hinge axis at station s by th deg"""
        xh, zh = self.x_h(s), self.apex(s)[1]
        a = math.radians(th)
        p = np.asarray(pts_xz, float) - [xh, zh]
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return p @ R.T + [xh, zh]

class Pocket:
    """Servo-tray pocket in the bottom skin: flat floor at z_floor over [xa, xb] x [sa, sb],
    45 deg ramps at both span ends (span = print z, so a square end would be a ceiling)."""

    def __init__(self, name, surf, xa, xb, sa, sb, z_floor, z_below, keep):
        self.name, self.surf = name, surf
        self.xa, self.xb, self.sa, self.sb = xa, xb, sa, sb
        self.z_floor, self.z_below, self.keep = z_floor, z_below, keep
        self.ramp = z_floor - z_below

    def cutter(self):
        r = self.ramp
        loop = [(self.sa - r, self.z_below), (self.sb + r, self.z_below), (self.sb, self.z_floor), (self.sa, self.z_floor)]
        pts = self.surf.to_global([[self.xa, s, z] for s, z in loop])
        face = cq.Face.makeFromWires(cq.Wire.makePolygon([V(p) for p in pts], close=True))
        return cq.Solid.extrudeLinear(face, V(self.surf.R @ np.array([self.xb - self.xa, 0.0, 0.0])))

    def blocked(self, x, s):
        k, r = self.keep, self.ramp
        return self.xa - k <= x <= self.xb + k and self.sa - r - k <= s <= self.sb + r + k

def place_pocket(surf, s0, s1, hinge, bores, SP, P):
    """Put the tray pocket SP['span_offset'] outboard of s0, chordwise in the widest gap between
    the LE region, the spar bore keep-out (incl. its slit) and the hinge groove."""
    keep = P["bore_keepout"]
    s_c = s0 + SP["span_offset"]
    sa, sb = s_c - SP["span_len"] / 2, s_c + SP["span_len"] / 2
    ramp_est = SP["depth"] + 3.0
    if sa - ramp_est < s0 or sb + ramp_est > s1:
        warn("servo pocket (with its 45 deg ramps) runs past the panel ends; change span_offset")
    bad = []
    for s in np.linspace(max(sa - ramp_est, s0), min(sb + ramp_est, s1), 7):
        c, xle = surf.chord(s), surf.x_le(s)
        bad.append((-1e9, xle + 0.1 * c))
        bad.append((hinge.x_h(s) - hinge.opening(s) - keep, 1e9) if hinge else (xle + 0.9 * c, 1e9))
        for b in bores:
            a0, a1 = apply(surf.Ginv, [b.p0, b.p1])
            m = b.d / 2 + keep
            if min(a0[1], a1[1]) - m <= s <= max(a0[1], a1[1]) + m:
                ds = a1[1] - a0[1]
                lam = 0.0 if abs(ds) < 1e-9 else min(max((s - a0[1]) / ds, 0.0), 1.0)
                xb = a0[0] + lam * (a1[0] - a0[0])
                bad.append((xb - m, xb + m))
    bad.sort()
    free, cur = [], -1e9
    for a, b in bad:
        if a > cur:
            free.append((cur, a))
        cur = max(cur, b)
    if not free:
        warn("no room for the servo pocket between the spar and the hinge groove; skipped")
        return None, None
    if SP["chord_fraction"] is None:
        win = max(free, key=lambda ab: ab[1] - ab[0])
        x_c = 0.5 * (win[0] + win[1])
    else:
        x_c = surf.x_le(s_c) + SP["chord_fraction"] * surf.chord(s_c)
        win = next((w for w in free if w[0] <= x_c <= w[1]), (x_c, x_c))
    xa, xb = x_c - SP["chord_len"] / 2, x_c + SP["chord_len"] / 2
    if xa < win[0] or xb > win[1]:
        warn(f"servo pocket {SP['chord_len']:.0f} mm chordwise does not fit the free window "
             f"({win[1] - win[0]:.1f} mm between spar and hinge groove); it cuts into a keep-out")
    xs = np.linspace(xa, xb, 9)
    lo_c = surf.surfaces_at(s_c, x_c)[0]
    z_floor = lo_c + SP["depth"]
    z_lim = min(surf.surfaces_at(s, x)[1] for s in (sa, s_c, sb) for x in xs) - P["slot_gap"]
    if z_floor > z_lim:
        warn(f"servo tray depth {SP['depth']:.1f} mm does not fit; pocket floor limited to "
             f"{z_lim - lo_c:.1f} mm (tray stands {z_floor - z_lim:.1f} mm proud of the skin)")
        z_floor = z_lim
    z_below = min(surf.surfaces_at(s, x)[0] for s in np.linspace(sa - ramp_est, sb + ramp_est, 5)
                  for x in xs if not math.isnan(surf.surfaces_at(s, x)[0])) - 2.0
    pk = Pocket("servo_pocket", surf, xa, xb, sa, sb, z_floor, z_below, keep)
    meta = dict(x_c=x_c, s_c=s_c, depth=z_floor - lo_c, ramp_skin=z_floor - lo_c, window=win)
    return pk, meta

# vase-mode panel
def vase_panel(surf, s0, s1, pitch, angle, bores, P, hinge=None, trims=(), pockets=()):
    body = surf.solid(s0, s1)
    blocks = ([hinge.blocked] if hinge else []) + [p.blocked for p in pockets]
    tools = surf.slot_cutters(s0, s1, pitch, angle, P["slot_width"], P["slot_gap"], bores, P["bore_keepout"],
                              blocks=blocks)
    for b in bores:
        tools.append(b.solid())
        sl = b.slit(surf, P["slot_width"], P["bore_keepout"])
        if sl is not None:
            tools.append(sl)
    if hinge:
        tools.append(hinge.cutter())
        tools += [hinge.end_trim(se, dr) for se, dr in trims]
    tools += [p.cutter() for p in pockets]
    body = body.cut(*tools)
    return biggest_solid(body.clean())

def check_hinge_bores(hinge, bores, keep):
    cut = hinge.cutter()
    for b in bores:
        if b.keepout(keep).intersect(cut).Volume() > 1e-6:
            warn(f"{b.name} bore runs into the {hinge.name} hinge groove; move the hinge aft or the rod forward")

# STL cleanup + vase check (need trimesh; skipped without it)
def clean_stl(path):
    """Drop zero-area triangles and close tessellation cracks that OCC sometimes leaves."""
    try:
        import trimesh
        import shapely
        import networkx
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
        import shapely
        import networkx
    except ImportError:
        return "skipped (pip install trimesh shapely networkx)"
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

def pchip(xk, yk, x):
    """Monotone piecewise-cubic (Fritsch-Carlson) interpolation: never overshoots the data."""
    xk, yk, x = np.asarray(xk, float), np.asarray(yk, float), np.asarray(x, float)
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    for i in range(1, len(xk) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    for i, j, k in ((0, 0, 1), (-1, -1, -2)):        # one-sided, shape-preserving end slopes
        if len(d) == 1:
            m[i] = d[0]
            continue
        hj, hk = h[j], h[k]
        e = ((2 * hj + hk) * d[j] - hj * d[k]) / (hj + hk)
        if e * d[j] <= 0:
            e = 0.0
        elif d[j] * d[k] <= 0 and abs(e) > abs(3 * d[j]):
            e = 3 * d[j]
        m[i] = e
    i = np.clip(np.searchsorted(xk, x) - 1, 0, len(h) - 1)
    t = (x - xk[i]) / h[i]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1]

def densify_stations(st, extra_x=(), step=6.0):
    """Resample (x, w, h, z_top) stations every ~step mm with PCHIP so the loft has no room to
    overshoot between sparse stations (a spline loft through few sections bulges and dips)."""
    xk = np.array([s[0] for s in st])
    keep = sorted(set(xk.tolist()) | {float(x) for x in extra_x})
    grid = [x for x in np.arange(xk[0], xk[-1], step)
            if min(abs(x - k) for k in keep) > 0.4 * step]
    xs = np.array(sorted(keep + grid))
    cols = [pchip(xk, [s[k] for s in st], xs) for k in (1, 2, 3)]
    return [[float(x), float(a), float(b), float(c)] for x, a, b, c in zip(xs, *cols)]

def build(cfg):
    P = cfg["print"]
    clr = P["rod_clearance"]
    parts = {}      # name -> dict(shape=global solid, print=4x4 or callable, kind="vase"/"solid")
    info = []

    # ---------------- wing ----------------
    W = cfg["wing"]
    wing = Surface("wing", W["root_chord"], W["tip_chord"], W["semispan"], W["le_sweep"],
                   W["twist"], W["airfoil"], W["origin"], W["dihedral"], P["te_thickness"],
                   W.get("incidence", 0.0))
    sp = cfg["wing_spar"]
    res, f = solve_continuous_rod(wing, sp["diameter"], clr, P["min_wall"], sp["chord_fraction"],
                                  sp["max_semi_length"])
    if res is None:
        raise SystemExit("Wing spar does not fit at the root. Use a smaller diameter.")
    x_r, z_r, semi = res
    spar = Bore("wing_spar", [x_r, 0, z_r], [x_r, semi, z_r], sp["diameter"], clr)
    bores = [spar]
    info.append(f"Wing spar: {sp['diameter']:.1f} mm rod, length {2 * semi:.0f} mm, "
                f"at x={x_r:.1f}, z={z_r:.2f} (root chord fraction {f:.2f}), centred on the centreline")
    splits = [fr * wing.b for fr in W["splits"]]
    for sj in splits[1:-1]:
        if semi < sj:
            info.append(f"  (dihedral {W['dihedral']} deg limits the straight spar to {semi:.0f} mm per side; "
                        f"the wing joint at {sj:.0f} mm has no rod through it)")

    C = cfg["controls"]
    if wing.twist_tip:
        warn("wing has twist: the hinge line is not exactly straight on a twisted panel")
    n_pan = len(splits) - 1
    for i, (s0, s1) in enumerate(zip(splits[:-1], splits[1:])):
        t0 = time.time()
        name = f"wing_R{i + 1}"
        print(f"building {name} ...", flush=True)
        hinge, trims, pockets = None, (), ()
        if i == n_pan - 1:                                   # aileron = whole outer panel
            A = C["aileron"]
            hinge = Hinge("aileron", wing, s0, s1, A["hinge_fraction"], A["hinge_side"], A["throw"], C, P)
            check_hinge_bores(hinge, bores, P["bore_keepout"])
            if n_pan > 1:
                trims = [(s0, +1)]                           # inboard end = top of the tip-down print
            pk, meta = place_pocket(wing, s0, s1, hinge, bores, cfg["servo_pocket"], P)
            if pk is not None:
                pockets = [pk]
                SP = cfg["servo_pocket"]
                xg = wing.to_global([[pk.xa, meta["s_c"], 0.0], [pk.xb, meta["s_c"], 0.0]])
                info.append(f"Servo tray pocket (bottom skin, both aileron panels): {SP['chord_len']:.0f} x "
                            f"{SP['span_len']:.0f} mm, floor {meta['depth']:.1f} mm above the skin, centred at "
                            f"y=+/-{xg[0][1]:.0f}, x={xg[0][0]:.1f}..{xg[1][0]:.1f} "
                            f"({meta['s_c'] - s0:.0f} mm outboard of the joint). The opening runs "
                            f"~{meta['ramp_skin']:.1f} mm further at each span end (45 deg ramps for vase mode)")
            c0, c1 = hinge.chord_aft(s0), hinge.chord_aft(s1)
            area = 0.5 * (c0 + c1) * (s1 - s0)
            info.append(f"Ailerons: whole outer panel, {s1 - s0 - (C['end_gap'] if trims else 0):.0f} mm span, "
                        f"hinge at {A['hinge_fraction']:.2f}c on the {A['hinge_side']} skin, chord "
                        f"{c0:.1f}->{c1:.1f} mm, {2 * area / 100:.0f} cm^2 total; V {hinge.vee:.0f} deg "
                        f"-> {A['throw']:.0f} deg max towards the V side")
        body = vase_panel(wing, s0, s1, W["lattice_pitch"], W["lattice_angle"], bores, P,
                          hinge=hinge, trims=trims, pockets=pockets)
        tip_down = (i == len(splits) - 2)
        parts[name] = dict(shape=body, kind="vase", frame=wing, s0=s0, s1=s1, tip_down=tip_down)
        parts[f"wing_L{i + 1}"] = dict(shape=body.mirror("XZ"), kind="vase", mirror_of=name)
        print(f"   {time.time() - t0:.1f}s")

    # ---------------- tail ----------------
    H, Vt, Mt = cfg["htail"], cfg["vtail"], cfg["mount"]
    ht = Surface("ht", H["root_chord"], H["tip_chord"], H["semispan"], H["le_sweep"], H["twist"],
                 H["airfoil"], H["origin"], H["dihedral"], P["te_thickness"], H.get("incidence", 0.0))
    vt = Surface("vt", Vt["root_chord"], Vt["tip_chord"], Vt["span"], Vt["le_sweep"], Vt["twist"],
                 Vt["airfoil"], Vt["origin"], 90.0, P["te_thickness"])
    half_w = Mt["width"] / 2
    xm = ht.x_le(0) + ht.max_thickness_fraction() * ht.cr
    z_ht = H["origin"][2]
    U0, L0 = ht.section(0.0)                                   # root section, global z incl. incidence
    zg_up = ht.to_global(np.c_[U0[:, 0], np.zeros(len(U0)), U0[:, 1]])[:, 2]
    zg_lo = ht.to_global(np.c_[L0[:, 0], np.zeros(len(L0)), L0[:, 1]])[:, 2]
    ht_root_lo = float(np.interp(xm, U0[:, 0], zg_lo)) - z_ht    # under the max-thickness point
    ht_root_up = float(zg_up.max()) - z_ht
    boom = cfg["boom"]
    r_boom = (boom["diameter"] + clr) / 2
    z_boom = z_ht + ht_root_lo - Mt["wall"] - r_boom
    s_stub = (ht_root_up - 0.0) + 1.0                       # VT stub height above HT chord plane

    # elevator / rudder hinges (whole HT halves, whole VT above the stub; root end = top of print)
    for surf_ in (ht, vt):
        if surf_.twist_tip:
            warn(f"{surf_.name} has twist: the hinge line is not exactly straight on a twisted panel")
    E, Rd = C["elevator"], C["rudder"]
    gap = C["end_gap"]
    h_el = Hinge("elevator", ht, half_w, ht.b, E["hinge_fraction"], E["hinge_side"], E["throw"], C, P)
    h_rd = Hinge("rudder", vt, s_stub, vt.b, Rd["hinge_fraction"], Rd["hinge_side"], Rd["throw"], C, P)
    if abs(E["hinge_fraction"] * ht.cr - Rd["hinge_fraction"] * vt.cr) > 1.0 or abs(H["origin"][0] - Vt["origin"][0]) > 1.0:
        warn("elevator and rudder hinge lines are not at the same x; the mount is trimmed at the more "
             "forward one, so the other surface's fixed part overhangs the trimmed mount")
    for h in (h_el, h_rd):
        c = h.chord_aft(h.s0)
        info.append(f"{h.name.capitalize()}: whole {'HT halves' if h is h_el else 'VT above the mount'}, "
                    f"{h.s1 - h.s0 - gap:.0f} mm span{' each' if h is h_el else ''}, hinge at {h.f:.2f}c on the "
                    f"{'top' if h.sgn > 0 else 'bottom'} skin, chord {c:.1f} mm; V {h.vee:.0f} deg -> "
                    f"{h.throw:.0f} deg max towards the V side")
    # rudder swing vs the elevator root faces
    s_r = s_stub + gap
    U, L = vt.section(s_r)
    aft = np.r_[U[U[:, 0] > h_rd.x_h(s_r)], L[L[:, 0] > h_rd.x_h(s_r)]]
    lat = max(np.abs(h_rd.sweep(s_r, aft, th)[:, 1]).max() for th in (Rd["throw"], -Rd["throw"]))
    if lat > half_w + gap - 0.5:
        warn(f"rudder bottom swings {lat:.1f} mm sideways at {Rd['throw']:.0f} deg, the elevator roots are at "
             f"{half_w + gap:.1f} mm; reduce the rudder throw or widen the mount")
    info.append(f"  rudder bottom swings +/-{lat:.1f} mm at full throw; elevator roots at +/-{half_w + gap:.1f} mm")
    x_cut = min(ht.to_global([[h_el.x_h(0.0), 0.0, h_el.apex(0.0)[1]]])[0][0], vt.to_global([[h_rd.x_h(0.0), 0.0, 0.0]])[0][0])

    print("building tail ...", flush=True)
    ht_body = vase_panel(ht, half_w, ht.b, H["lattice_pitch"], H["lattice_angle"], [], P,
                         hinge=h_el, trims=[(half_w, +1)])
    parts["ht_R"] = dict(shape=ht_body, kind="vase", frame=ht, s0=half_w, s1=ht.b, tip_down=True)
    parts["ht_L"] = dict(shape=ht_body.mirror("XZ"), kind="vase", mirror_of="ht_R")
    vt_body = vase_panel(vt, s_stub, vt.b, Vt["lattice_pitch"], Vt["lattice_angle"], [], P,
                         hinge=h_rd, trims=[(s_stub, +1)])
    parts["vt"] = dict(shape=vt_body, kind="vase", frame=vt, s0=s_stub, s1=vt.b, tip_down=True)

    # mount
    x_le_ht = H["origin"][0]
    c_ht = ht.cr
    r_tube = r_boom + Mt["wall"]
    # servo block: aft face on the HT LE, top on the HT chord plane, slots either side of the boom
    TS = cfg["tail_servos"]
    sw_ = TS["wall"]
    sl_l, sl_w = TS["length"] + TS["clearance"], TS["width"] + TS["clearance"]
    blk_x1 = x_le_ht
    blk_x0 = blk_x1 - TS["tab_span"] - 2 * sw_
    x_sc = 0.5 * (blk_x0 + blk_x1)
    y_in = r_boom + sw_
    blk_hw = y_in + sl_w + sw_
    blk_z1 = z_ht
    blk_z0 = min(blk_z1 - TS["tab_height"], z_boom - r_tube)
    block = cq.Solid.makeBox(blk_x1 - blk_x0, 2 * blk_hw, blk_z1 - blk_z0, V((blk_x0, -blk_hw, blk_z0)))
    slots = [cq.Solid.makeBox(sl_l, sl_w, blk_z1 - blk_z0 + 2, V((x_sc - sl_l / 2, y0, blk_z0 - 1)))
             for y0 in (y_in, -y_in - sl_w)]
    x_front = min(x_le_ht - Mt["nose_ahead"], blk_x0)
    centre = ht.solid(0, half_w).fuse(ht.solid(0, half_w).mirror("XZ"))
    stub = vt.solid(0, s_stub)
    tube = cylinder([x_front, 0, z_boom], [x_le_ht + 0.8 * c_ht, 0, z_boom], 2 * r_tube)
    web = cq.Solid.makeBox(0.65 * c_ht, 2 * r_tube, z_ht - 0.3 - z_boom,
                           V((x_le_ht + 0.12 * c_ht, -r_tube, z_boom)))
    mount = centre.fuse(stub, tube, web, block).clean()
    boom_bore = cylinder([x_front - 1, 0, z_boom], [x_front + boom["socket_mount"], 0, z_boom], 2 * r_boom)
    mount = biggest_solid(mount.cut(boom_bore, *slots).clean())
    info.append(f"Tail servo slots: 2 x {sl_l:.1f} x {sl_w:.1f} mm through-slots at x={x_sc - sl_l / 2:.1f}..{x_sc + sl_l / 2:.1f}, "
                f"y=+/-{y_in:.1f}..{y_in + sl_w:.1f}; tabs sit on the block top at z={blk_z1:.1f}, case bottom "
                f"at z={blk_z1 - TS['tab_height']:.1f}; block {blk_x1 - blk_x0:.1f} x {2 * blk_hw:.1f} x "
                f"{blk_z1 - blk_z0:.1f} mm")
    # trim everything aft of the hinge line: open gap between the elevator roots, under the rudder
    mount = biggest_solid(mount.intersect(cq.Solid.makeBox(1000, 1000, 1000, V((x_cut - 1000, -500, -500)))).clean())
    info.append(f"Tail mount trimmed aft of x={x_cut:.1f} (hinge line)")
    mw = Mt["wall"]
    r_in = r_tube - mw
    cx0 = max(x_front + boom["socket_mount"], blk_x1) + mw
    cx1 = x_cut - mw
    U0, L0 = ht.section(0.0)
    lo_g = ht.to_global(np.c_[L0[:, 0], np.zeros(len(L0)), L0[:, 1]])
    in_rng = (lo_g[:, 0] >= cx0 - mw) & (lo_g[:, 0] <= cx1 + mw)
    z_cav = min(lo_g[in_rng, 2].min() if in_rng.any() else z_ht, z_ht - 0.3) - mw
    w_x0 = x_le_ht + 0.12 * c_ht + mw
    cavity = cylinder([cx0, 0, z_boom], [cx1, 0, z_boom], 2 * r_in)
    if z_cav > z_boom and cx1 > w_x0:
        cavity = cavity.fuse(cq.Solid.makeBox(cx1 - w_x0, 2 * r_in, z_cav - z_boom, V((w_x0, -r_in, z_boom))))
    cavity = cavity.intersect(cq.Solid.makeBox(cx1 - cx0, 2 * r_in, z_cav - (z_boom - r_in), V((cx0, -r_in, z_boom - r_in))))
    m_before = mount.Volume()
    mount = biggest_solid(mount.cut(cavity).clean())
    info.append(f"Tail mount rear section hollow: x={cx0:.1f}..{cx1:.1f}, {mw} mm walls, "
                f"{(m_before - mount.Volume()) * P['density_solid'] / 1000:.1f} g removed")
    parts["tail_mount"] = dict(shape=mount, kind="solid", print_axis="x")

    # ---------------- fuselage ----------------
    F = cfg["fuselage"]
    st = [list(s) for s in F["stations"]]
    tube_d = 2 * (r_boom + F["socket_wall"])
    st[-1] = [st[-1][0], tube_d, tube_d, z_boom + tube_d / 2]
    st = densify_stations(st, extra_x=[F["split_x"]])
    t = F["wall"]
    n = F["squareness"]
    outer = cq.Solid.makeLoft([superellipse_wire(x, w, h, zt - h / 2, n) for x, w, h, zt in st], False)
    inner_st = [list(s) for s in st]
    inner_st[0][0] += F["firewall"]
    inner_st = [inner_st[0]] + [s for s in inner_st[1:] if s[0] > inner_st[0][0] + 1.0]
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

    boom_len = (x_front + boom["socket_mount"]) - (x_end - boom["socket_fuselage"])
    info.append(f"Tail boom: {boom['diameter']} mm rod, length {boom_len:.0f} mm "
                f"(horizontal at z={z_boom:.1f}), {boom['socket_fuselage']:.0f} mm into the wingbox, "
                f"{boom['socket_mount']:.0f} mm into the tail mount")
    rod_list = [("wing_spar", spar.p1 * [1, -1, 1], spar.p1, spar.rod_d),
                ("boom", [x_end - boom["socket_fuselage"], 0, z_boom],
                 [x_front + boom["socket_mount"], 0, z_boom], boom["diameter"])]
    for name, p0, p1, d in rod_list:
        parts["rod_" + name] = dict(shape=cylinder(p0, p1, d), kind="rod")
    return parts, info

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

def vase_mass(part, P, bottom=0.6):
    """Walls at line_width; the bed face (span-normal plane at the print bottom) as `bottom` mm of
    solid layers; other span-normal planes are open tops in vase mode and weigh nothing."""
    surf, shp = part["frame"], part["shape"]
    bed = part["s1"] if part["tip_down"] else part["s0"]
    span = surf.R @ np.array([0.0, 1.0, 0.0])
    walls = cap = 0.0
    mx = at = 0.0
    for f in shp.Faces():
        a, c = f.Area(), f.Center()
        at += a
        mx += a * c.x
        if f.geomType() == "PLANE":
            n = f.normalAt()
            if abs(n.x * span[0] + n.y * span[1] + n.z * span[2]) > 0.999:
                if abs(apply(surf.Ginv, [[c.x, c.y, c.z]])[0][1] - bed) < 0.5:
                    cap += a
                continue
        walls += a
    rho = P["density_vase"] / 1000
    return walls * P["line_width"] * rho + cap * bottom * rho, mx / at


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
    if not (P["line_width"] < CONFIG["controls"]["neck"] < 2 * P["line_width"]):
        warn("controls.neck should sit between line_width and 2*line_width or the hinge skin drops out / splits the layer")

    t0 = time.time()
    parts, info = build(CONFIG)
    os.makedirs(os.path.join(args.out, "step"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "stl"), exist_ok=True)

    lines = ["RC plane print parts", "=" * 60, ""]
    lines += info + [""]
    lines.append(f"{'part':<12}{'print':<34}{'mass g':>8}{'  vase check'}")
    total = 0.0
    moment = 0.0
    rod_mass = 0.0
    assy = cq.Assembly(name="aircraft")
    vase_cache = {}
    for name, part in parts.items():
        shp = part["shape"]
        if part["kind"] == "rod":
            assy.add(shp, name=name, color=cq.Color(0.1, 0.1, 0.1))
            m_r = shp.Volume() * P["density_rod"] / 1000
            rod_mass += m_r
            moment += m_r * shp.Center().x
            continue
        if not shp.isValid():
            warn(f"{name}: OCC reports the solid as invalid; check it in SolidWorks")
        cq.exporters.export(shp, os.path.join(args.out, "step", f"{name}.step"))
        col = cq.Color(0.29, 0.48, 0.71) if part["kind"] == "vase" else cq.Color(0.78, 0.51, 0.16)
        assy.add(shp, name=name, color=col)
        if part["kind"] == "vase":
            src = part.get("mirror_of", name)
            if src not in vase_cache:
                vase_cache[src] = vase_mass(parts[src], P)
            mass, cx = vase_cache[src]
            how = "vase mode, " + ("tip on bed" if parts.get(part.get("mirror_of", name), part).get("tip_down") else "root on bed")
        else:
            mass = shp.Volume() * P["density_solid"] / 1000
            cx = shp.Center().x
            how = "normal, front face on bed"
        total += mass
        moment += mass * cx
        chk = ""
        if not args.no_stl:
            pth = os.path.join(args.out, "stl", f"{name}.stl")
            shp_p = to_print(name, parts)
            for k in (1.0, 0.7, 1.4, 0.5):        # retry tessellation if OCC leaves a crack
                cq.exporters.export(shp_p, pth, tolerance=P["stl_tolerance"] * k,
                                    angularTolerance=P["stl_angular"] * k)
                ok = clean_stl(pth)
                if ok is not False:
                    break
            else:
                warn(f"{name}.stl is not watertight after cleanup; let the slicer repair it")
            if args.check and part["kind"] == "vase":
                chk = vase_check(pth)
        lines.append(f"{name:<12}{how:<34}{mass:8.1f}  {chk}")
        print(f"  {name:<12}{how:<34}{mass:7.1f} g  {chk}", flush=True)
    assy.export(os.path.join(args.out, "step", "assembly.step"))
    lines.append(f"{'TOTAL printed':<46}{total:8.1f}")
    m_s = total + rod_mass
    x_s = moment / m_s
    x_cg = CONFIG["x_cg"]
    x_fw = CONFIG["fuselage"]["stations"][0][0]
    lines += ["", "Balance (airframe = printed parts + CF rods, no electronics)",
              f"  CF rods {rod_mass:.1f} g (solid; a tube boom is lighter), airframe {m_s:.1f} g, "
              f"airframe CG x = {x_s:.1f} mm aft of root LE",
              f"  target x_cg = {x_cg:.2f} mm -> electronics must supply {m_s * (x_s - x_cg):.0f} g.mm nose-down moment "
              f"about x_cg,",
              f"  e.g. {m_s * (x_s - x_cg) / (x_cg - (x_fw + 25)):.0f} g of motor/battery/ESC lumped at x = {x_fw + 25:.0f} mm "
              f"(25 mm behind the firewall)"]
    lines += ["", "Slicer: spiral vase / spiralize for vase parts, line width "
              f"{P['line_width']} mm, 3-4 bottom layers, no top layers.",
              "Wing panels: inboard panels print root-down (closed face at the joint, open tip),",
              "the tip panel prints tip-down. Tail halves and VT print tip-down (open root against the mount).",
              "",
              "Control surfaces (printed in place, still joined by the hinge skin):",
              "  1. Tape the hinge skin across the groove at neutral (surface still attached = aligned).",
              "  2. Run a blade along the flat at the bottom of the V to cut the neck.",
              "  3. Fold the surface fully towards the hinge-skin side (opens the V), tape across the opened joint.",
              "  Servo pockets: open on the bottom skin; glue the trays in, the 45 deg ramp areas stay open."]
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