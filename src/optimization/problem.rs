use crate::config::Config;
use crate::optimization::residuals::cost;
use argmin::core::{CostFunction, Error};

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
