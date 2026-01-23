import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Adjust import if needed depending on where you put app.py
from scripts.sevenbar import Config as SevenBarConfig
from scripts.sevenbar import solve as solve_sevenbar
from scripts.sevenbar import solve_pose_theta, plot_pose_matplotlib
from scripts.threebar import Config as ThreeBarConfig
from scripts.threebar import solve as solve_threebar
from scripts.threebar import solve_pose_ratio
# If your file is literally named 7bar.py, python can't import it as a module.
# Rename it to _7bar.py or sevenbar.py (recommended), then import as above.


st.set_page_config(layout="wide")
mode = st.sidebar.selectbox("Solver", ["7-bar", "3-bar"])
st.title(
    "7-Bar Linkage Designer + DR-Rotation Animator"
    if mode == "7-bar"
    else "3-Bar Linkage Designer"
)

def mm_to_m(val_mm: float) -> float:
    return float(val_mm) / 1000.0

# ---------------------------
# State
# ---------------------------
if "design_7bar" not in st.session_state:
    st.session_state.design_7bar = None

if "anim_seed_7bar" not in st.session_state:
    st.session_state.anim_seed_7bar = None

if "design_3bar" not in st.session_state:
    st.session_state.design_3bar = None

if "pose_seed_3bar" not in st.session_state:
    st.session_state.pose_seed_3bar = None


if mode == "7-bar":
    # ---------------------------
    # Sidebar controls
    # ---------------------------
    st.sidebar.header("Geometry")
    Hext_mm = st.sidebar.number_input("Hext (mm)", value=650.0, step=10.0, format="%.1f")
    ax1_mm = st.sidebar.number_input("AX1 = dist(1,3) (mm)", value=65.0, step=5.0, format="%.1f")
    Hext = mm_to_m(Hext_mm)
    ax1 = mm_to_m(ax1_mm)

    st.sidebar.header("Thickness / clearance")
    w_TR_mm = st.sidebar.number_input("TR thickness w_TR (mm)", value=10.0, step=1.0, format="%.1f")
    w_DR_mm = st.sidebar.number_input("DR thickness w_DR (mm)", value=10.0, step=1.0, format="%.1f")
    gap_mm = st.sidebar.number_input("Clearance gap (mm)", value=2.0, step=0.5, format="%.2f")
    w_TR = mm_to_m(w_TR_mm)
    w_DR = mm_to_m(w_DR_mm)
    gap = mm_to_m(gap_mm)

    st.sidebar.header("Pose ratios (design solver)")
    ratios = st.sidebar.multiselect("ratios", options=[1.0, 0.8, 0.2], default=[1.0, 0.8, 0.2])

    st.sidebar.header("Weights")
    w_len = st.sidebar.slider("w_len", 0.0, 5000.0, 250.0, 10.0)
    w_pose = st.sidebar.slider("w_pose", 0.0, 5000.0, 900.0, 10.0)
    w_clear = st.sidebar.slider("w_clear", 0.0, 20000.0, 1500.0, 50.0)
    w_above = st.sidebar.slider("w_above", 0.0, 20000.0, 1200.0, 50.0)
    w_tr_orient = st.sidebar.slider("w_tr_orient", 0.0, 20000.0, 1200.0, 50.0)
    w_soft = st.sidebar.slider("w_soft", 0.0, 5000.0, 60.0, 5.0)
    w_soft_ineq = st.sidebar.slider("w_soft_ineq", 0.0, 5000.0, 60.0, 5.0)

    st.sidebar.header("Advanced")
    close_scale_mm = st.sidebar.number_input("close_scale (mm)", value=20.0, step=5.0, format="%.1f")
    above_margin_mm = st.sidebar.number_input("above_margin (mm)", value=8.0, step=1.0, format="%.1f")
    tr_orient_margin = st.sidebar.number_input("tr_orient_margin", value=1e-5, step=1e-5, format="%.6f")
    close_scale = mm_to_m(close_scale_mm)
    above_margin = mm_to_m(above_margin_mm)

    solve_btn = st.sidebar.button("Solve design (lengths + multi-pose)", type="primary")

    # ---------------------------
    # Build config
    # ---------------------------
    cfg = SevenBarConfig(
        Hext=Hext,
        ax1=ax1,
        w_len=w_len,
        w_pose=w_pose,
        w_soft=w_soft,
        w_soft_ineq=w_soft_ineq,
        w_TR=w_TR,
        w_DR=w_DR,
        gap=gap,
        w_clear=w_clear,
        w_above=w_above,
        w_tr_orient=w_tr_orient,
        tr_orient_margin=tr_orient_margin,
        close_scale=close_scale,
        ratios=tuple(sorted(ratios, reverse=True)) if ratios else (1.0, 0.8, 0.2),
        above_margin=above_margin,
    )

    # ---------------------------
    # Solve design
    # ---------------------------
    if solve_btn:
        with st.spinner("Solving design..."):
            design = solve_sevenbar(cfg)
            st.session_state.design_7bar = design
            st.session_state.anim_seed_7bar = None

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Design output")
        if st.session_state.design_7bar is None:
            st.info("Click **Solve design** to compute shared lengths + poses.")
        else:
            st.json(st.session_state.design_7bar)

    with right:
        st.subheader("Animation: DR rotates about point 3")

        if st.session_state.design_7bar is None:
            st.info("Solve design first.")
        else:
            lengths = st.session_state.design_7bar["lengths"]

            theta_deg = st.slider("DR angle θ (deg)", -180.0, 180.0, 0.0, 1.0)
            theta_rad = float(np.deg2rad(theta_deg))

            # Solve pose with continuation seed for stability
            pose = solve_pose_theta(cfg, lengths, theta_rad, x0=st.session_state.anim_seed_7bar)
            st.session_state.anim_seed_7bar = np.array(pose["seed"], dtype=float)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plot_pose_matplotlib(ax, pose, title=f"θ={theta_deg:.1f}°")
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.write("Pose solver status:")
                st.write({
                    "success": pose["success"],
                    "cost": pose["cost"],
                    "message": pose["message"],
                })
else:
    st.sidebar.header("Geometry")
    Hcrouch_mm = st.sidebar.number_input("Hcrouch (mm)", value=350.0, step=10.0, format="%.1f")
    Hext_mm = st.sidebar.number_input("Hext (mm)", value=650.0, step=10.0, format="%.1f")
    kc_min_mm = st.sidebar.number_input("kc_min (mm)", value=50.0, step=5.0, format="%.1f")
    bc_radius_max_mm = st.sidebar.number_input(
        "bc_radius_max (mm, 0=off)",
        value=100.0,
        step=10.0,
        format="%.1f",
    )
    Hcrouch = mm_to_m(Hcrouch_mm)
    Hext = mm_to_m(Hext_mm)
    kc_min = mm_to_m(kc_min_mm)
    bc_radius_max = mm_to_m(bc_radius_max_mm)
    st.sidebar.header("Link constraints")
    kc_max_mm = st.sidebar.number_input("kc_max (mm, 0=off)", value=100.0, step=10.0, format="%.1f")
    xbc_min_mm = st.sidebar.number_input("xbc_min (mm)", value=0.0, min_value=0.0, step=5.0, format="%.1f")
    kc_max = mm_to_m(kc_max_mm)
    xbc_min = mm_to_m(xbc_min_mm)
    cw_hk_ratio_min = st.sidebar.number_input(
        "cw/hk ratio min (0=off)",
        value=0.9,
        step=0.05,
        format="%.2f",
    )
    cw_hk_ratio_max = st.sidebar.number_input(
        "cw/hk ratio max (0=off)",
        value=1.1,
        step=0.05,
        format="%.2f",
    )
    lc_hk_ratio_min = st.sidebar.number_input(
        "lc/hk ratio min (0=off)",
        value=1.1,
        step=0.05,
        format="%.2f",
    )
    lc_hk_ratio_max = st.sidebar.number_input(
        "lc/hk ratio max (0=off)",
        value=1.1,
        step=0.05,
        format="%.2f",
    )
    w_bc_x = st.sidebar.number_input("w_bc_x", value=0.01, step=0.005, format="%.4f")
    w_bc_y = st.sidebar.number_input("w_bc_y", value=0.05, step=0.005, format="%.4f")

    st.sidebar.header("Sampling")
    ratios = st.sidebar.multiselect(
        "ratios (0..1)",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        default=[0.0, 0.5, 1.0],
    )
    samples = st.sidebar.number_input("samples (if no ratios)", value=3, min_value=2, step=1)

    st.sidebar.header("Jumping profile")
    jac_start_mm = st.sidebar.number_input("jac_start (mm/rad)", value=40.0, step=10.0, format="%.1f")
    jac_end_mm = st.sidebar.number_input("jac_end (mm/rad)", value=160.0, step=10.0, format="%.1f")
    jac_min_mm = st.sidebar.number_input("jac_min (mm/rad)", value=15.0, step=5.0, format="%.1f")
    jac_max_mm = st.sidebar.number_input("jac_max (mm/rad)", value=350.0, step=10.0, format="%.1f")
    jac_start = mm_to_m(jac_start_mm)
    jac_end = mm_to_m(jac_end_mm)
    jac_min = mm_to_m(jac_min_mm)
    jac_max = mm_to_m(jac_max_mm)

    st.sidebar.header("Theta constraints")
    theta_span_min = st.sidebar.number_input("theta_span_min (rad)", value=0.9, step=0.05, format="%.3f")
    theta_step_min = st.sidebar.number_input("theta_step_min (rad)", value=0.02, step=0.005, format="%.3f")

    st.sidebar.header("Motor")
    omega_max = st.sidebar.number_input("omega_max (rad/s, 0=off)", value=11.0, step=0.1, format="%.3f")

    st.sidebar.header("Optimization")
    n_starts = st.sidebar.number_input("n_starts", value=8, min_value=1, step=1)
    use_global = st.sidebar.checkbox("use global optimization", value=False)

    solve_btn = st.sidebar.button("Solve 3-bar design", type="primary")

    ratio_tuple = tuple(sorted(ratios)) if ratios else tuple(np.linspace(0.0, 1.0, int(samples)))
    cfg = ThreeBarConfig(
        Hcrouch=Hcrouch,
        Hext=Hext,
        ratios=ratio_tuple,
        n_starts=int(n_starts),
        use_global_opt=use_global,
        jac_start=jac_start,
        jac_end=jac_end,
        jac_min=jac_min,
        jac_max=jac_max,
        theta_span_min=theta_span_min,
        theta_step_min=theta_step_min,
        kc_min=kc_min,
        kc_max=float(kc_max) if float(kc_max) > 0.0 else None,
        xbc_min=float(xbc_min),
        cw_hk_ratio_min=float(cw_hk_ratio_min) if float(cw_hk_ratio_min) > 0.0 else None,
        cw_hk_ratio_max=float(cw_hk_ratio_max) if float(cw_hk_ratio_max) > 0.0 else None,
        lc_hk_ratio_min=float(lc_hk_ratio_min) if float(lc_hk_ratio_min) > 0.0 else None,
        lc_hk_ratio_max=float(lc_hk_ratio_max) if float(lc_hk_ratio_max) > 0.0 else None,
        w_bc_x=w_bc_x,
        w_bc_y=w_bc_y,
        bc_radius_max=float(bc_radius_max) if float(bc_radius_max) > 0.0 else None,
        omega_max=float(omega_max) if float(omega_max) > 0.0 else None,
    )

    if solve_btn:
        with st.spinner("Solving 3-bar design..."):
            design = solve_threebar(cfg)
            st.session_state.design_3bar = design
            st.session_state.pose_seed_3bar = None

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Design output")
        if st.session_state.design_3bar is None:
            st.info("Click **Solve 3-bar design** to compute lengths + poses.")
        else:
            st.json(st.session_state.design_3bar)

    with right:
        st.subheader("Mechanism pose")
        if st.session_state.design_3bar is None:
            st.info("Solve design first.")
        else:
            if not st.session_state.design_3bar.get("success", True):
                st.error(st.session_state.design_3bar.get("message", "No valid non-crossing solution found."))
            pose_ratio = st.slider("pose ratio", 0.0, 1.0, 0.0, 0.01)
            lengths = st.session_state.design_3bar["lengths"]
            pin_joint = st.session_state.design_3bar["pin_joint"]
            inner_offset = st.session_state.design_3bar["inner_joint_offset_KC"]

            pose = solve_pose_ratio(
                cfg,
                lengths=lengths,
                pin_joint=pin_joint,
                inner_joint_offset_kc=inner_offset,
                ratio=pose_ratio,
                x0=st.session_state.pose_seed_3bar,
            )
            st.session_state.pose_seed_3bar = np.array(pose["seed"], dtype=float)
            if pose.get("crossing", False):
                st.warning("Crossing detected at this pose ratio.")

            pts = pose["points"]
            H = np.array([pts["H"]["x"], pts["H"]["y"]], dtype=float)
            K = np.array([pts["K"]["x"], pts["K"]["y"]], dtype=float)
            C = np.array([pts["C"]["x"], pts["C"]["y"]], dtype=float)
            W = np.array([pts["W"]["x"], pts["W"]["y"]], dtype=float)
            Bc = np.array([pts["Bc"]["x"], pts["Bc"]["y"]], dtype=float)

            pose_cols = st.columns([2, 1])
            with pose_cols[0]:
                fig, ax = plt.subplots(figsize=(4.0, 4.0))
                segments = [
                    ("H-K", H, K),
                    ("K-C", K, C),
                    ("Bc-C", Bc, C),
                    ("K-W", K, W),
                ]
                for _, a, b in segments:
                    ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=2)

                for label, p in [("H", H), ("K", K), ("C", C), ("W", W), ("Bc", Bc)]:
                    ax.scatter([p[0]], [p[1]], s=40)
                    ax.text(p[0], p[1], f"  {label}", fontsize=8, va="center")

                ax.set_aspect("equal", adjustable="box")
                ax.grid(True)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m) (+Y down)")
                ax.invert_yaxis()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

            with pose_cols[1]:
                st.subheader("Pose coordinates")
                poses = st.session_state.design_3bar.get("poses", {})
                keys = sorted(poses.keys(), key=float) if poses else []
                if poses:
                    rows = []
                    for key in keys:
                        points = poses[key]["points"]
                        rows.append(
                            {
                                "ratio": float(key),
                                "Bc": f"({points['Bc']['x']:.4f}, {points['Bc']['y']:.4f})",
                                "C": f"({points['C']['x']:.4f}, {points['C']['y']:.4f})",
                                "K": f"({points['K']['x']:.4f}, {points['K']['y']:.4f})",
                                "W": f"({points['W']['x']:.4f}, {points['W']['y']:.4f})",
                            }
                        )
                    st.table(rows)
                else:
                    st.info("No pose coordinate data available.")

            st.write("Quality metrics:")
            st.write(st.session_state.design_3bar.get("quality", {}))
