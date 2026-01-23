"""
compute_lqr.py

Interactive LQR gain calculator for a 2-wheel self-balancing robot.

- Asks you for robot physical parameters (wheel radius, mass, CoM height, etc.)
- Asks you for LQR tuning weights (Q, R)
- Computes continuous-time LQR gains for state:

    x_bal = [theta, theta_dot, x, x_dot]^T

- Prints:
    K matrix
    Individual gains
    C++ snippet for LqrParams
"""

import numpy as np
from dataclasses import dataclass
from scipy.linalg import solve_continuous_are


# =======================
# Robot / model parameters
# =======================

@dataclass
class RobotParams:
    wheel_radius: float
    wheel_base: float
    body_mass: float
    wheel_mass: float
    com_height: float
    body_inertia: float  # if None or <=0, will be approximated as m * h^2
    motor_kt: float
    motor_resistance: float
    gear_ratio: float
    max_voltage: float
    gravity: float
    cart_friction: float

    def __post_init__(self):
        # If inertia not specified, approximate as m * l^2
        if self.body_inertia is None or self.body_inertia <= 0:
            self.body_inertia = self.body_mass * self.com_height ** 2

    @property
    def M(self) -> float:
        """Cart mass (wheels + any base). Very rough: 2 * wheel_mass."""
        return 2 * self.wheel_mass

    @property
    def m(self) -> float:
        """Pendulum mass (body above axle)."""
        return self.body_mass

    @property
    def l(self) -> float:
        """Pendulum CoM height."""
        return self.com_height

    @property
    def I(self) -> float:
        """Pendulum inertia about axle."""
        return self.body_inertia

    @property
    def g(self) -> float:
        return self.gravity

    @property
    def b(self) -> float:
        """Cart friction coefficient."""
        return self.cart_friction


# =======================
# Helpers
# =======================

def build_cartpole_linear_model(params: RobotParams):
    """
    Build linearized cart-pole A, B matrices around upright equilibrium,
    for state in cart order:

        x_cart = [x, x_dot, theta, theta_dot]^T

    u is horizontal force on the cart [N].
    """

    M = params.M
    m = params.m
    l = params.l
    I = params.I
    g = params.g
    b = params.b

    denom = I * (M + m) + M * m * l ** 2

    A = np.array([
        [0.0,                      1.0,                     0.0,                             0.0],
        [0.0,  -(I + m * l ** 2) * b / denom,   (m ** 2 * g * l ** 2) / denom,              0.0],
        [0.0,                      0.0,                     0.0,                             1.0],
        [0.0,          -m * l * b / denom,       m * g * l * (M + m) / denom,              0.0]
    ])

    B = np.array([
        [0.0],
        [(I + m * l ** 2) / denom],
        [0.0],
        [m * l / denom]
    ])

    return A, B


def transform_state_order(A_cart, B_cart):
    """
    Transform A, B from cart order [x, x_dot, theta, theta_dot]
    to balancer order [theta, theta_dot, x, x_dot].

    x_cart = T * x_bal

    x_bal = [theta, theta_dot, x, x_dot]^T
    """

    T = np.array([
        [0, 0, 1, 0],  # x_cart[0] = x      = x_bal[2]
        [0, 0, 0, 1],  # x_cart[1] = x_dot  = x_bal[3]
        [1, 0, 0, 0],  # x_cart[2] = theta  = x_bal[0]
        [0, 1, 0, 0],  # x_cart[3] = theta_dot = x_bal[1]
    ], dtype=float)

    A_bal = T.T @ A_cart @ T
    B_bal = T.T @ B_cart

    return A_bal, B_bal


def lqr(A, B, Q, R):
    """
    Continuous-time LQR:

        x' = A x + B u

    Solve for K in:

        u = -K x
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ P)
    return K


def compute_lqr_gains(robot_params: RobotParams,
                      q_theta=1.0,
                      q_theta_dot=0.1,
                      q_x=0.0,
                      q_x_dot=0.1,
                      r_u=0.01):
    """
    Compute LQR gains for:

        x_bal = [theta, theta_dot, x, x_dot]^T

    Cost:
        J = ∫ ( x^T Q x + u^T R u ) dt
    """

    A_cart, B_cart = build_cartpole_linear_model(robot_params)
    A_bal, B_bal   = transform_state_order(A_cart, B_cart)

    Q = np.diag([q_theta, q_theta_dot, q_x, q_x_dot])
    R = np.array([[r_u]])

    K = lqr(A_bal, B_bal, Q, R)  # shape (1, 4)
    k_theta, k_theta_dot, k_x, k_x_dot = K.flatten()

    return {
        "K": K,
        "k_theta": k_theta,
        "k_theta_dot": k_theta_dot,
        "k_x": k_x,
        "k_x_dot": k_x_dot,
        "Q": Q,
        "R": R,
        "A_bal": A_bal,
        "B_bal": B_bal,
    }


# =======================
# Interactive prompts
# =======================

def ask_float(prompt: str, default: float) -> float:
    txt = input(f"{prompt} [{default}]: ").strip()
    if txt == "":
        return default
    try:
        return float(txt)
    except ValueError:
        print("Invalid number, using default.")
        return default


def main():
    print("=== LQR Gain Calculator for Self-Balancing Robot ===")
    print("Press Enter to accept defaults.\n")

    # --- Robot parameters ---
    wheel_radius     = ask_float("Wheel radius (m)", 0.05)
    wheel_base       = ask_float("Wheel base (distance between wheels, m)", 0.15)
    body_mass        = ask_float("Body mass above axle (kg)", 3.0)
    wheel_mass       = ask_float("Wheel mass per wheel (kg)", 0.3)
    com_height       = ask_float("CoM height above axle (m)", 0.12)
    body_inertia_inp = input("Body inertia about axle I (kg*m^2) [default: m*h^2]: ").strip()
    if body_inertia_inp == "":
        body_inertia = None
    else:
        try:
            body_inertia = float(body_inertia_inp)
        except ValueError:
            print("Invalid number, using default m*h^2.")
            body_inertia = None

    motor_kt         = ask_float("Motor torque constant Kt (Nm/A)", 0.05)
    motor_resistance = ask_float("Motor phase resistance (Ohm)", 0.5)
    gear_ratio       = ask_float("Gear ratio (motor_rot / wheel_rot)", 1.0)
    max_voltage      = ask_float("Max motor voltage (V)", 12.0)
    gravity          = ask_float("Gravity (m/s^2)", 9.81)
    cart_friction    = ask_float("Cart friction coefficient b (N*s/m)", 0.0)

    params = RobotParams(
        wheel_radius=wheel_radius,
        wheel_base=wheel_base,
        body_mass=body_mass,
        wheel_mass=wheel_mass,
        com_height=com_height,
        body_inertia=body_inertia,
        motor_kt=motor_kt,
        motor_resistance=motor_resistance,
        gear_ratio=gear_ratio,
        max_voltage=max_voltage,
        gravity=gravity,
        cart_friction=cart_friction,
    )

    print("\n--- LQR tuning weights ---")
    print("Cost J = ∫ ( x^T Q x + u^T R u ) dt")
    print("x = [theta, theta_dot, x, x_dot]^T\n")

    q_theta     = ask_float("Q_theta   (penalty on tilt angle)", 50.0)
    q_theta_dot = ask_float("Q_thetaDot (penalty on tilt rate)", 5.0)
    q_x         = ask_float("Q_x       (penalty on position)", 0.0)
    q_x_dot     = ask_float("Q_xDot    (penalty on velocity)", 1.0)
    r_u         = ask_float("R_u       (penalty on control effort)", 0.01)

    print("\nComputing LQR gains...")
    result = compute_lqr_gains(
        params,
        q_theta=q_theta,
        q_theta_dot=q_theta_dot,
        q_x=q_x,
        q_x_dot=q_x_dot,
        r_u=r_u,
    )

    K = result["K"]

    print("\n=== LQR gain matrix K ===")
    print("u = -K * [theta, theta_dot, x, x_dot]^T")
    print(K)
    print()

    print("=== Individual gains (float) ===")
    print(f"kTheta    = {result['k_theta']:.6f}")
    print(f"kThetaDot = {result['k_theta_dot']:.6f}")
    print(f"kX        = {result['k_x']:.6f}")
    print(f"kXDot     = {result['k_x_dot']:.6f}")
    print()

    print("=== C++ snippet for LqrParams ===")
    print("LqrParams lqrParams;")
    print(f"lqrParams.kTheta    = {result['k_theta']:.6f}f;")
    print(f"lqrParams.kThetaDot = {result['k_theta_dot']:.6f}f;")
    print(f"lqrParams.kX        = {result['k_x']:.6f}f;")
    print(f"lqrParams.kXDot     = {result['k_x_dot']:.6f}f;")
    print(f"lqrParams.qTheta    = {q_theta:.6f}f;")
    print(f"lqrParams.qThetaDot = {q_theta_dot:.6f}f;")
    print(f"lqrParams.qX        = {q_x:.6f}f;")
    print(f"lqrParams.qXDot     = {q_x_dot:.6f}f;")
    print(f"lqrParams.rU        = {r_u:.6f}f;")
    print()

    print("Done. Paste this snippet into your C++ firmware where you configure LqrParams.")


if __name__ == "__main__":
    main()

