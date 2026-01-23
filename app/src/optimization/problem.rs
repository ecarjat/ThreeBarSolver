use crate::config::Config;
use crate::optimization::residuals::{cost, pose_cost};
use argmin::core::{CostFunction, Error};
use nalgebra::Vector2;

/// Argmin-compatible problem for full linkage optimization
#[derive(Clone)]
pub struct ThreeBarProblem {
    pub cfg: Config,
}

impl ThreeBarProblem {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }
}

impl CostFunction for ThreeBarProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(cost(p, &self.cfg))
    }
}

/// Argmin-compatible problem for single pose optimization
#[derive(Clone)]
pub struct PoseProblem {
    pub cfg: Config,
    pub bc: Vector2<f64>,
    pub lu: f64,
    pub lkc: f64,
    pub lc: f64,
    pub lkw: f64,
    pub target_y: f64,
}

impl PoseProblem {
    pub fn new(
        cfg: Config,
        bc: Vector2<f64>,
        lu: f64,
        lkc: f64,
        lc: f64,
        lkw: f64,
        target_y: f64,
    ) -> Self {
        Self {
            cfg,
            bc,
            lu,
            lkc,
            lc,
            lkw,
            target_y,
        }
    }
}

impl CostFunction for PoseProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(pose_cost(
            p,
            &self.cfg,
            &self.bc,
            self.lu,
            self.lkc,
            self.lc,
            self.lkw,
            self.target_y,
        ))
    }
}
