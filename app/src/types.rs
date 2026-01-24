use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

/// 2D point type using nalgebra
pub type Point2D = Vector2<f64>;

/// Linkage lengths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lengths {
    pub upper_leg_hk: f64, // Lu: Hip to Knee
    pub lower_leg_kw: f64, // Lkw: Knee to Wheel (along K-C direction)
    pub link_bc_c: f64,    // Lc: Pin joint (Bc) to Inner joint (C)
}

/// Fixed pin joint location (already constrained if bc_radius_max is used)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinJointLocation {
    pub x: f64,
    pub y: f64,
}

/// Points in a single pose
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosePoints {
    pub h: Point2D,  // Hip (always origin)
    pub k: Point2D,  // Knee
    pub c: Point2D,  // Inner joint on lower leg
    pub w: Point2D,  // Wheel contact point
    pub bc: Point2D, // Pin joint (fixed)
}

/// Single pose result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    pub ratio: f64,
    pub target_wheel_y: f64,
    pub points: PosePoints,
    pub crossing: bool,
}

/// Jump kinematic report
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JumpReport {
    pub j_min: f64,             // Min dy/dtheta (m/rad)
    pub j_max: f64,             // Max dy/dtheta
    pub j_start: f64,           // J at start of stroke
    pub j_end: f64,             // J at end of stroke
    pub theta_span: f64,        // Total hip angle range (rad)
    pub y_dot_takeoff_est: f64, // Estimated vertical velocity at takeoff (m/s)
}

/// Quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Quality {
    pub max_wheel_x: f64,
    pub wheel_x_pp: f64,   // Peak-to-peak X deviation
    pub wheel_x_rms: f64,  // RMS X deviation
    pub wheel_y_span: f64, // Vertical travel range
    pub mean_wheel_x: f64,
    pub crossing: bool,
}

/// Complete solution output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub success: bool,
    pub cost: f64,
    pub h_crouch: f64,
    pub h_ext: f64,
    pub lengths: Lengths,
    pub pin_joint: PinJointLocation,
    pub inner_joint_offset_kc: f64, // Lkc
    pub jump_report: JumpReport,
    pub poses: Vec<Pose>,
    pub quality: Quality,
    pub message: Option<String>,
}

/// Pose solver result (for interactive slider)
#[derive(Debug, Clone)]
pub struct PoseSolveResult {
    pub success: bool,
    pub cost: f64,
    pub points: PosePoints,
    pub crossing: bool,
    pub seed: Vec<f64>, // For continuation
}
