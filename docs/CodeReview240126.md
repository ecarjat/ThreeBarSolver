# Code Review: 3-Bar Linkage Designer

**Date:** 2026-01-24
**Codebase:** robot2Wheel/solver (Rust)
**Total Lines:** ~3,137

---

## Overall Assessment

This is a well-structured Rust application for designing 3-bar linkage mechanisms for legged robots. The code is generally clean and follows good practices. However, there are several areas for improvement.

---

## Critical Issues

### 1. Potential Panic in Multi-start Optimization
**Location:** `src/optimization/solver.rs:161-164`

```rust
let best = results
    .iter()
    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())  // unwrap can panic on NaN
    .unwrap();  // unwrap can panic if results is empty
```

If `n_starts` is 0 (despite validation) or all costs are NaN, this will panic. Use `.expect()` with a message or handle gracefully.

### 2. Redundant Clone in Parallel Iteration
**Location:** `src/optimization/solver.rs:64`

```rust
let problem = ThreeBarProblem::new(cfg.clone());
```

`cfg` is cloned for every local solve. Consider using `Arc<Config>` if `ThreeBarProblem` can accept a reference.

### 3. Unbounded Loop in `pick_three_distinct`
**Location:** `src/optimization/solver.rs:240-253`

```rust
while c == exclude || c == a || c == b {
    c = rng.gen_range(0..n);
}
```

If `n < 4`, this could loop excessively. The `pop_size.max(4)` guard at line 173 helps, but a direct check would be safer.

---

## Performance Issues

### 4. Repeated Sorting in Multiple Places

The pattern `ratios_sorted.sort_by(...)` appears in:
- `src/optimization/solver.rs:120-121`
- `src/optimization/solver.rs:283-284`
- `src/optimization/residuals.rs:63-64`

Consider pre-computing sorted indices once in `Config` or caching them.

### 5. Inefficient Residual Vector Allocation
**Location:** `src/optimization/residuals.rs:14`

```rust
let mut r: Vec<f64> = Vec::with_capacity(100);
```

The capacity of 100 is a guess. With ~45 config fields and multiple constraints per pose, this could cause reallocations. Calculate the exact capacity based on `cfg.ratios.len()`.

### 6. Expensive Pose Sweep in `solve_pose_ratio`
**Location:** `src/optimization/solver.rs:448-476`

The 720-step sweep over all theta values is expensive. Consider:
- Using adaptive sampling
- Starting from a better initial guess
- Using Newton-Raphson instead of bisection for refinement

### 7. Duplicate Kinematics Computations
**Location:** `src/optimization/residuals.rs:133-181`

The alpha constraint computes `eval_pose_for_theta` twice (theta-eps and theta+eps) for numerical differentiation. Consider caching intermediate results or using analytical derivatives.

---

## Code Quality Issues

### 8. Magic Numbers Throughout

Examples:
- `1337` in `src/optimization/solver.rs:21` - seed multiplier
- `0.05 * scale`, `3.0 * scale` - length bounds (`src/optimization/solver.rs:26-27`)
- `1e-9`, `1e-12` - tolerances scattered throughout
- `720` steps (`src/optimization/solver.rs:448`)
- `1e6` penalty (`src/optimization/residuals.rs:82`)

Extract these as named constants or config parameters.

### 9. Inconsistent Error Handling

- `src/optimization/solver.rs:79-85`: Silently ignores `NelderMead::new` errors
- `src/optimization/solver.rs:98-102`: Silently ignores executor errors

Consider using `Result` types or at least logging errors.

### 10. Dead/Unused Code

- `w_len` in config is defined but never used in residuals
- `pose_seed` in `AppState` (`src/app/state.rs:12`) is never used

### 11. Type Conversion Precision Loss
**Location:** `src/app/state.rs:58-67`

```rust
pub h_crouch_mm: f32,  // UI state
// ...
self.config.h_crouch = (self.h_crouch_mm / 1000.0) as f64;
```

f32 → f64 → f32 round-trips lose precision. Use f64 throughout or accept the precision loss explicitly.

---

## Robustness Issues

### 12. Missing Input Validation
**Location:** `src/optimization/residuals.rs:67`

```rust
let theta = *vars.poses.get(idx).unwrap();
```

Will panic if the pose index doesn't exist. Should use `.get()` with proper error handling.

### 13. Division by Near-Zero
**Location:** `src/optimization/residuals.rs:186-188`

```rust
if dalpha.abs() > 1e-6 {
    alpha_error / (dalpha / (2.0 * eps))
```

The divisor could still be very small, causing large values.

### 14. Config Validation Not Called

`Config::validate()` is implemented but never called before solving. Invalid configs could cause undefined behavior.

---

## Design Improvements

### 15. Missing Trait Implementations

- `Solution` should implement `Default`
- `Lengths` and `PinJointLocation` could implement `From<&UnpackedVars>`

### 16. Coupling Between UI and Domain
**Location:** `src/app/state.rs`

`AppState` mixes UI concerns (mm conversion, checkbox state) with domain logic. Consider separating:
- `UiState` - UI-specific values
- `AppState` - domain state only

### 17. No Logging/Tracing

The optimization process has no logging. Consider adding `tracing` for debugging convergence issues.

### 18. Test Coverage

Only basic unit tests exist for `geometry.rs` and `linkage.rs`. Missing tests for:
- `residuals.rs` cost function
- `solver.rs` optimization logic
- Edge cases (empty ratios, degenerate geometry)

---

## Suggested Improvements Summary

| Priority | Issue | Location |
|----------|-------|----------|
| **High** | Panic on unwrap in multistart | solver.rs:161-164 |
| **High** | Missing config validation call | Before solve |
| **High** | Unbounded loop risk | solver.rs:240-253 |
| **Medium** | Repeated sorting | Multiple files |
| **Medium** | Magic numbers | Throughout |
| **Medium** | Expensive 720-step sweep | solver.rs:448 |
| **Medium** | f32/f64 precision loss | state.rs |
| **Low** | Dead code (w_len, pose_seed) | config.rs, state.rs |
| **Low** | Missing tests | Throughout |
| **Low** | UI/domain coupling | state.rs |

---

## Positive Observations

1. **Good module organization** - Clear separation of concerns between geometry, linkage, optimization, and UI
2. **Version migration** - Proper handling of config file evolution with `VersionedParams`
3. **Parallel optimization** - Effective use of rayon for multi-start optimization
4. **Constraint handling** - Mix of hard bounds and soft penalties is well-designed
5. **Comprehensive types** - Domain types are well-defined with serde support for serialization
6. **Geometric utilities** - Robust segment intersection and distance calculations with proper edge case handling

---

## Architecture Overview

```
src/
├── main.rs              # App entry, window setup
├── lib.rs               # Module declarations
├── types.rs             # Core data structures
├── config.rs            # Configuration with migration
├── geometry.rs          # 2D computational geometry
├── linkage.rs           # Mechanism kinematics
├── app/
│   ├── mod.rs
│   ├── state.rs         # Central app state
│   ├── sidebar.rs       # Configuration UI
│   ├── plot.rs          # 2D visualization
│   └── results.rs       # Results display
└── optimization/
    ├── mod.rs
    ├── problem.rs       # Argmin integration
    ├── packing.rs       # Variable packing
    ├── bounds.rs        # Optimization bounds
    ├── residuals.rs     # Cost function
    └── solver.rs        # Optimization pipeline
```

---

*Review conducted using Claude Code*
