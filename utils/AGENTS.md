# AGENTS.md — Leg Geometry Optimisation Agent

## Purpose

This project designs and optimises a **compliant, knee-based leg mechanism with a wheel**, driven from the hip, whose objective is to produce a **quasi-linear (mostly vertical) wheel trajectory** relative to the robot body.

The system is intended for:
- jumping / crouching robots
- impact absorption via passive compliance
- minimal horizontal disturbance at the wheel during vertical motion

The solver is **geometry-first**, not dynamics-first.  
It finds **link lengths and anchor positions** that produce good kinematic behaviour before adding motors, springs, or control.

---

## Mechanical Structure (Ground Truth)

All geometry is defined in the **body frame**.

### Fixed Points (Body)

- **H (Hip)**  
  - Coordinates: `H = (0, 0)`  
  - Fixed to the body  
  - Actuated joint (hip motor)

- **Bc (Body Connector)**  
  - Coordinates: `Bc = (xbc, ybc)`  
  - Fixed to the body  
  - Constraint: `ybc > 0` (above hip)
  - Typically behind the hip (`xbc < 0`)

---

### Moving Joints and Links

- **Upper leg (H → K)**  
  - Length: `Lu`  
  - Rigid rod  
  - Rotates about H  
  - Hip angle: `alpha` (primary DOF)

- **Knee joint (K)**  
  - Passive rotational joint  
  - In physical design: contains a torsion spring  
  - In this solver: purely geometric

- **Lower leg line (W — K — C)**  
  - Colinear rigid structure  
  - `K → W` length: `Ll` (**input to the optimiser**)  
  - `K → C` length: `Lkc`

- **Connector rod (Bc → C)**  
  - Length: `Lc`  
  - Rigid rod  
  - Couples body motion to knee motion

---

## Kinematic Constraints

For a pose to be valid:

1. **Hip-driven motion**
   - Knee position:
     ```
     K(alpha) = (Lu*cos(alpha), Lu*sin(alpha))
     ```

2. **Connector constraint**
   - Point `C` must satisfy:
     ```
     |C - K| = Lkc
     |C - Bc| = Lc
     ```
   - (circle–circle intersection)

3. **Lower leg colinearity**
   - `W`, `K`, `C` lie on the same line
   - `K` is between `W` and `C`
   - Wheel position:
     ```
     W = K - Ll * unit(C - K)
     ```

4. **Vertical ordering**
   - The following must be **below the hip line**:
     ```
     K.y < 0
     C.y < 0
     W.y < 0
     ```
   - `Bc` is explicitly allowed (and expected) to be above the hip

5. **Non-intersection**
   - The rods `H–K` and `Bc–C` must not intersect (excluding endpoints)

---

## Design Variables

The optimiser searches over:

- `Lu`   — upper leg length
- `Lkc`  — knee → connector distance
- `Lc`   — connector rod length
- `xbc`  — horizontal position of body connector
- `ybc`  — vertical position of body connector

The **only required input** is:

- `Ll` — lower leg length (knee → wheel)

---

## Objective (What We Optimise For)

The optimisation seeks a geometry that:

1. Produces a **quasi-vertical wheel trajectory**
   - Minimise RMS of `Wx`
   - Minimise peak-to-peak drift of `Wx`
   - Penalise curvature (`Wx_pp / |Wy_range|`)

2. Maximises **contiguous range of motion**
   - Large continuous span of valid hip angles `alpha`
   - No flickering between solution branches

3. Produces **compact, buildable geometry**
   - Penalise large `|Bc|`
   - Avoid degenerate “infinite lever” solutions

4. Keeps geometry mechanically reasonable
   - Prefer `Lu ≈ Lkc` (balanced knee geometry)
   - Avoid inverted or over-centred configurations

This is a **quasi-linear motion design problem**, not a rigid four-bar synthesis.

---

## What This Solver Is *Not*

- It does **not** model:
  - dynamics
  - forces
  - spring stiffness
  - motor torque
- It does **not** enforce wheel–ground contact

Those come **after** geometry is validated.

---

## Expected Next Steps (Outside This Agent)

- Add torsion spring model at knee
- Compute equilibrium knee angle under load
- Add Jacobian-based mechanical advantage metrics
- Export optimal geometry into CAD (Fusion 360)
- Add dynamics / control (LQR, impedance, etc.)

---

## Summary

This agent answers one question:

> **“Given a lower leg length, how should I place the hip, knee, and connector so the wheel mostly goes up and down?”**

Everything else builds on that.