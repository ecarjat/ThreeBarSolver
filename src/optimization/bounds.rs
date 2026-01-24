use crate::config::Config;
use crate::optimization::packing::num_variables;
use std::f64::consts::PI;

/// Variable bounds for optimization
#[derive(Debug, Clone)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Build optimization bounds from config
pub fn build_bounds(cfg: &Config) -> Bounds {
    let max_h = cfg.h_crouch.abs().max(cfg.h_ext.abs()).max(0.2);
    let max_extent = (2.5 * max_h).max(1.0);
    let min_len = 0.05 * max_h;
    let max_len = 3.0 * max_h;

    let n_ratios = cfg.ratios.len();
    let n_vars = num_variables(cfg);

    let mut lower = vec![-max_extent; n_vars];
    let mut upper = vec![max_extent; n_vars];

    // Pose angle bounds
    for i in 0..n_ratios {
        lower[i] = -PI;
        upper[i] = PI;
    }

    let len_start = n_ratios;

    // Length bounds [Lu, Lkw, Lkc, Lc]
    for i in 0..4 {
        lower[len_start + i] = min_len;
        upper[len_start + i] = max_len;
    }

    // Lkc (inner joint offset) bounds
    let lkc_idx = len_start + 2;
    lower[lkc_idx] = lower[lkc_idx].max(cfg.kc_min);
    if lower[lkc_idx] >= upper[lkc_idx] {
        upper[lkc_idx] = lower[lkc_idx] + 1e-3_f64.max(0.05 * lower[lkc_idx]);
    }
    if let Some(kc_max) = cfg.kc_max {
        upper[lkc_idx] = upper[lkc_idx].min(kc_max);
        if upper[lkc_idx] <= lower[lkc_idx] {
            upper[lkc_idx] = lower[lkc_idx] + 1e-3_f64.max(0.05 * lower[lkc_idx]);
        }
    }

    // Pin joint bounds
    let xbc_idx = len_start + 4;
    if let Some(xbc_min) = cfg.xbc_min {
        if cfg.bc_radius_max.is_none() {
            lower[xbc_idx] = lower[xbc_idx].max(xbc_min);
            if lower[xbc_idx] >= upper[xbc_idx] {
                upper[xbc_idx] = lower[xbc_idx] + 1e-3_f64.max(0.05 * lower[xbc_idx]);
            }
        }
    }

    Bounds { lower, upper }
}

impl Bounds {
    /// Clamp a vector to be within bounds
    pub fn clamp(&self, x: &mut [f64]) {
        let eps = 1e-9;
        for i in 0..x.len() {
            x[i] = x[i].clamp(self.lower[i] + eps, self.upper[i] - eps);
        }
    }
}
