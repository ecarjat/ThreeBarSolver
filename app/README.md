# Three-Bar Linkage Solver

A Rust application for optimizing 3-bar linkage mechanisms for robot leg design. The solver finds linkage geometries that achieve desired vertical wheel travel while maintaining kinematic constraints.

## Mechanism Overview

The 3-bar linkage consists of 5 key points:

```
                          W (Wheel)
                          │
                          │ Lkw
                          │
    H (Hip) ─────────── K (Knee)
      │ upper leg (Lu)    │
      │                   │ Lkc
      │                   │
      │                   C (Inner Joint) ──── Bc (Pin Joint)
      │                                          link (Lc)
```

The lower leg is a **single bar** with three connection points: **W --- K --- C** (collinear).
K is in the middle, with W and C on opposite ends.

**Points:**
- **H (Hip)**: Fixed at origin (0, 0). The motor drives rotation here.
- **K (Knee)**: Connected to H by the upper leg. Middle of the lower leg bar.
- **C (Inner Joint)**: One end of the lower leg bar. Connected to Bc via the link.
- **W (Wheel)**: Other end of the lower leg bar. Ground contact point.
- **Bc (Pin Joint)**: Fixed point on the robot body. Connected to C via a link.

**Lengths:**
- `Lu` (upper_leg_hk): Distance H → K
- `Lkw` (lower_leg_kw): Distance K → W
- `Lkc` (inner_joint_offset_kc): Distance K → C
- `Lc` (link_bc_c): Distance Bc → C (the connecting link)

**Coordinate System:**
- Origin at Hip (H)
- **+Y is DOWN** (toward ground)
- Internal units: meters (UI displays millimeters)

## Optimization Problem

### Goal

Find linkage lengths and pin joint position such that:
1. The wheel (W) moves vertically from `h_crouch` to `h_ext` as the leg extends
2. The wheel's X position stays as close to zero as possible (vertical motion)
3. The mechanism doesn't self-intersect (no crossing)
4. Various kinematic constraints are satisfied

### Decision Variables

For N pose ratios, the optimization has `N×4 + 6` variables:

```
[K0_x, K0_y, C0_x, C0_y,   // Pose 0: knee and inner joint positions
 K1_x, K1_y, C1_x, C1_y,   // Pose 1
 ...
 Lu, Lkw, Lkc, Lc,         // Link lengths (shared across all poses)
 xbc, ybc]                  // Pin joint position (fixed)
```

### Residual Terms (Constraints)

The solver minimizes the sum of squared weighted residuals:

#### Length Consistency (per pose)
```rust
(dist(H, K) - Lu) × w_len           // Upper leg length
(dist(K, C) - Lkc) × w_len          // Inner joint offset
(dist(Bc, C) - Lc) × w_len          // Connecting link length
```

#### Target Wheel Position (per pose)
```rust
(W.y - target_y) × w_pose           // Wheel height error
```
where `target_y = h_crouch + ratio × (h_ext - h_crouch)`

#### Wheel X Alignment
```rust
(W.x - mean_wheel_x) × w_wheel_x    // Per-pose deviation from mean
mean_wheel_x × w_wheel_x_mean       // Mean should be near zero
```

#### Geometric Constraints
```rust
// Knee must stay above wheel
max(0, K.y - W.y + margin) × w_knee_above_wheel

// No points below ground (Y ≥ 0)
max(0, -K.y) × w_below
max(0, -C.y) × w_below
max(0, -W.y) × w_below

// No crossing between H-K and Bc-C segments
seg_distance_violation × w_no_cross

// Knee angle H-K-W constraint
max(0, angle_HKW - max_angle) × w_angle_hkw
```

#### Link Ratio Constraints
```rust
// CW/HK ratio bounds (where CW = Lkc + Lkw)
constraint_violation × w_cw_hk_ratio

// LC/HK ratio bounds
constraint_violation × w_lc_hk_ratio
```

#### Jump Transmission Shaping
For jumping robots, the hip-to-wheel Jacobian (dy/dθ) profile matters:

```rust
// Minimum hip angle span
max(0, theta_span_min - actual_span) × w_theta_span

// Monotonic hip motion
max(0, theta_step_min - dθ) × w_theta_monotonic

// Jacobian profile (linear ramp from jac_start to jac_end)
(|J| - target_J) × w_jac_profile

// Jacobian bounds
max(0, jac_min - |J|) × w_jac_bounds
max(0, |J| - jac_max) × w_jac_bounds
```

#### Regularization
```rust
// Small preference for shorter links
Lu × w_reg, Lkw × w_reg, Lkc × w_reg, Lc × w_reg

// Bias pin joint toward origin
Bc.x × w_bc_x
Bc.y × w_bc_y
```

## Solver Architecture

### Multi-Start Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    solve() entry point                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
           use_global_opt?         Multi-Start
                    │                   │
                    ▼                   ▼
        Differential Evolution    ┌───────────┐
             (global)             │  Rayon    │
                    │             │  Parallel │
                    │             └───────────┘
                    │                   │
                    │         ┌─────────┼─────────┐
                    │         ▼         ▼         ▼
                    │     Start 0   Start 1   Start N
                    │         │         │         │
                    │         ▼         ▼         ▼
                    │     Nelder-   Nelder-   Nelder-
                    │      Mead      Mead      Mead
                    │         │         │         │
                    └─────────┼─────────┼─────────┘
                              ▼
                    Select best non-crossing
                              │
                              ▼
                    Local refinement (Nelder-Mead)
                              │
                              ▼
                    Build Solution struct
```

### Initial Seed Generation

Each parallel start creates a randomized initial guess:

1. For each pose ratio, generate plausible K and C positions
2. Estimate link lengths from a reference pose (ratio=1.0)
3. Randomize pin joint position within reasonable bounds
4. Pack into flat variable vector

### Local Optimization (Nelder-Mead)

The Nelder-Mead simplex algorithm is derivative-free, making it robust for this non-smooth problem:

1. Build initial simplex from starting point + perturbations
2. Iterate up to 20,000 function evaluations
3. Converge when simplex standard deviation < 1e-8
4. Return best point and cost

### Crossing Detection

After optimization, each solution is checked for "crossing" - when the upper leg segment (H-K) intersects the connecting link segment (Bc-C). Solutions with crossing are deprioritized.

## Single-Pose Solver

For the interactive visualization, a separate solver finds the mechanism pose for a given ratio:

```
solve_pose_ratio(ratio) → PoseSolveResult
```

Given fixed linkage parameters, it:

1. **Sweep θ from -π to π** (hip angle)
   - For each θ, compute K = Lu × (cos θ, sin θ)
   - Find C as intersection of two circles:
     - Circle 1: center K, radius Lkc
     - Circle 2: center Bc, radius Lc
   - Compute W from K and C
   - Track the θ where W.y crosses target_y

2. **Bisection refinement** on the bracket where W.y crosses target

3. Return the pose with minimal |W.y - target_y|

## Key Files

```
app/
├── src/
│   ├── main.rs              # eframe entry point
│   ├── lib.rs               # Module exports
│   ├── types.rs             # Core data structures
│   ├── config.rs            # ~45 configuration parameters
│   ├── geometry.rs          # 2D geometry utilities
│   ├── linkage.rs           # Linkage-specific functions
│   ├── optimization/
│   │   ├── mod.rs
│   │   ├── bounds.rs        # Variable bounds
│   │   ├── packing.rs       # Pack/unpack variable vectors
│   │   ├── residuals.rs     # Constraint residual functions
│   │   ├── problem.rs       # argmin CostFunction trait
│   │   └── solver.rs        # Multi-start + Nelder-Mead
│   └── app/
│       ├── mod.rs
│       ├── state.rs         # UI state + solution cache
│       ├── sidebar.rs       # Parameter controls
│       ├── plot.rs          # egui_plot visualization
│       └── results.rs       # Solution display + export
└── Cargo.toml
```

## Usage

```bash
cd app
cargo run --release
```

### UI Workflow

1. Adjust height parameters (h_crouch, h_ext) in the sidebar
2. Tune constraint weights and link ratio bounds as needed
3. Click "Solve" to run optimization
4. Use the pose slider to animate through the linkage motion
5. Export solution as JSON for use in robot firmware

## Dependencies

- **eframe/egui**: Immediate-mode GUI
- **egui_plot**: 2D plotting
- **argmin**: Optimization framework (Nelder-Mead)
- **nalgebra**: Linear algebra (Vector2)
- **rayon**: Parallel multi-start
- **serde**: Configuration serialization
