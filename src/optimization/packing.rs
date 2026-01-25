use crate::config::Config;
use crate::types::{Lengths, PinJointLocation};
use std::collections::BTreeMap;

/// Variable layout for N ratios:
/// [theta0, theta1, ..., thetaN-1, Lu, Lkw, Lkc, Lc, xbc, ybc]
/// Total: N + 6 variables

/// Unpacked optimization variables
#[derive(Debug, Clone)]
pub struct UnpackedVars {
    pub poses: BTreeMap<usize, f64>, // index -> theta (hip angle)
    pub lu: f64,                     // Upper leg length H-K
    pub lkw: f64,                    // Lower leg length K-W
    pub lkc: f64,                    // Inner joint offset K-C
    pub lc: f64,                     // Link length Bc-C
    pub xbc: f64,                    // Pin joint X
    pub ybc: f64,                    // Pin joint Y
}

impl From<&UnpackedVars> for Lengths {
    fn from(vars: &UnpackedVars) -> Self {
        Self {
            upper_leg_hk: vars.lu,
            lower_leg_kw: vars.lkw,
            link_bc_c: vars.lc,
        }
    }
}

impl From<&UnpackedVars> for PinJointLocation {
    fn from(vars: &UnpackedVars) -> Self {
        Self {
            x: vars.xbc,
            y: vars.ybc,
        }
    }
}

/// Get the number of optimization variables for a given config
#[inline]
pub fn num_variables(cfg: &Config) -> usize {
    cfg.ratios.len() + 6
}

/// Pack variables into a flat vector
pub fn pack_vars(
    poses: &BTreeMap<usize, f64>,
    lu: f64,
    lkw: f64,
    lkc: f64,
    lc: f64,
    xbc: f64,
    ybc: f64,
    cfg: &Config,
) -> Vec<f64> {
    let n = num_variables(cfg);
    let mut x = vec![0.0; n];

    let mut idx = 0;
    for i in 0..cfg.ratios.len() {
        if let Some(theta) = poses.get(&i) {
            x[idx] = *theta;
        }
        idx += 1;
    }

    x[idx] = lu;
    x[idx + 1] = lkw;
    x[idx + 2] = lkc;
    x[idx + 3] = lc;
    x[idx + 4] = xbc;
    x[idx + 5] = ybc;

    x
}

/// Unpack variables from a flat vector
pub fn unpack_vars(x: &[f64], cfg: &Config) -> UnpackedVars {
    let mut poses = BTreeMap::new();
    let mut idx = 0;

    for i in 0..cfg.ratios.len() {
        poses.insert(i, x[idx]);
        idx += 1;
    }

    UnpackedVars {
        poses,
        lu: x[idx],
        lkw: x[idx + 1],
        lkc: x[idx + 2],
        lc: x[idx + 3],
        xbc: x[idx + 4],
        ybc: x[idx + 5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let cfg = Config::default();
        let mut poses = BTreeMap::new();
        poses.insert(0, -0.8);
        poses.insert(1, 0.1);
        poses.insert(2, 1.2);

        let lu = 0.3;
        let lkw = 0.2;
        let lkc = 0.05;
        let lc = 0.25;
        let xbc = 0.05;
        let ybc = 0.03;

        let packed = pack_vars(&poses, lu, lkw, lkc, lc, xbc, ybc, &cfg);
        let unpacked = unpack_vars(&packed, &cfg);

        assert!((unpacked.poses[&0] - poses[&0]).abs() < 1e-10);
        assert!((unpacked.poses[&1] - poses[&1]).abs() < 1e-10);
        assert!((unpacked.poses[&2] - poses[&2]).abs() < 1e-10);
        assert!((unpacked.lu - lu).abs() < 1e-10);
        assert!((unpacked.lkw - lkw).abs() < 1e-10);
        assert!((unpacked.lkc - lkc).abs() < 1e-10);
        assert!((unpacked.lc - lc).abs() < 1e-10);
        assert!((unpacked.xbc - xbc).abs() < 1e-10);
        assert!((unpacked.ybc - ybc).abs() < 1e-10);
    }
}
