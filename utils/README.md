# Quasi-Linear Wheel Leg Geometry Optimiser

This repository contains a **geometry-first optimisation tool** for designing a **hip-driven, knee-based leg with a wheel** whose motion is **mostly vertical** relative to the robot body.

The goal is to find **link lengths and anchor positions** that naturally produce:
- a near-vertical wheel trajectory,
- large usable range of motion,
- compact, buildable geometry,
- passive compliance compatibility (torsion spring at the knee).

This tool focuses **only on kinematics and geometry**.  
Dynamics, motors, springs, and control come later.

---

## What Problem This Solves

In legged or wheel-leg robots that jump, crouch, or absorb impacts, you often want:

- the **wheel to go up and down**, not sideways,
- the **body to stay relatively stable**,
- compliance to come from **mechanics**, not control.

Simple 4-bar linkages tend to:
- produce large horizontal motion,
- be very sensitive to geometry,
- or require awkward packaging.

This project searches a broader design space to find geometries that:
- *naturally* generate a quasi-linear vertical wheel path,
- using a hip-driven upper leg, a knee joint, and a connector rod.

---

## Mechanism Overview

All geometry is defined in the **body frame**.

### Fixed to the Body
- **H (Hip)** at `(0, 0)`  
  - actuated joint (motor)
- **Bc (Body Connector)** at `(xbc, ybc)`  
  - above the hip (`ybc > 0`)
  - typically behind the hip (`xbc < 0`)

### Moving Structure
- **Upper leg**: `H → K` (length `Lu`)
- **Lower leg line**: `W — K — C`
  - `K → W` length `Ll` (**input**)
  - `K → C` length `Lkc`
- **Connector rod**: `Bc → C` (length `Lc`)
- **Knee (K)**:
  - passive joint (torsion spring in real design)

The hip motor drives the system through `Lu`.  
The connector rod shapes how the knee moves as the hip rotates.

---

## What the Optimiser Does

You provide:
- **`Ll`** — lower-leg length (knee → wheel)

The optimiser searches for:
- `Lu`  (upper leg length)
- `Lkc` (knee → connector distance)
- `Lc`  (connector rod length)
- `(xbc, ybc)` (body connector position)

such that:

- the wheel path is **mostly vertical**,
- the motion is valid over a **large contiguous range of hip angles**,
- no links intersect,
- all moving parts stay **below the hip**,
- the geometry remains **compact and realistic**.

---

## What This Tool Is *Not*

- ❌ No dynamics
- ❌ No motor torque modelling
- ❌ No spring stiffness or energy storage
- ❌ No ground contact forces

This is **intentional**.

Geometry is the hardest part to get right.
Once this looks good, dynamics and control become *much easier*.

---

## Repository Structure