import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata



# ------------------------------
# CFD utility functions
# ------------------------------

def create_nozzle_grid(nx: int, ny: int, length_conv: float, length_div: float, h_in: float, h_throat: float, h_out: float):
    """
    Create a simple algebraic structured mesh for a symmetric convergent-divergent nozzle.

    The computational domain is (xi, eta) in [0, 1] x [-1, 1], mapped to physical (x, y):
      x = L * xi
      y = eta * h(x)

    where h(x) is the local half-height of the nozzle.

    Returns
    -------
    x, y : 2D arrays [nx, ny]
        Physical coordinates of each grid point.
    h, dhdx : 1D arrays [nx]
        Nozzle half-height and wall slope at each x-station.
    """
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(-1.0, 1.0, ny)

    length_total = length_conv + length_div
    x_1d = length_total * xi

    # Piecewise smooth quadratic profile:
    # inlet -> throat in first half, throat -> outlet in second half.
    x_mid = length_conv
    left = x_1d <= x_mid
    right = ~left

    h = np.empty_like(x_1d)
    # Converging part (quadratic, zero slope at inlet and throat for smoothness)
    s_left = np.zeros_like(x_1d)
    s_left[left] = x_1d[left] / max(length_conv, 1e-12)
    h[left] = h_in + (h_throat - h_in) * (3.0 * s_left[left] ** 2 - 2.0 * s_left[left] ** 3)

    # Diverging part (quadratic/cubic blend with smooth throat/outlet)
    s_right = np.zeros_like(x_1d)
    s_right[right] = (x_1d[right] - x_mid) / max(length_div, 1e-12)
    h[right] = h_throat + (h_out - h_throat) * (3.0 * s_right[right] ** 2 - 2.0 * s_right[right] ** 3)

    # Wall slope via vectorized finite differences.
    dhdx = np.gradient(h, x_1d)

    # Build 2D structured grid with algebraic stretching in y.
    x = np.repeat(x_1d[:, None], ny, axis=1)
    y = eta[None, :] * h[:, None]

    return x, y, h, dhdx


def compute_metrics(x: np.ndarray, y: np.ndarray):
    """
    Compute curvilinear transformation metrics needed by transformed Euler equations.

    Continuous definitions:
      J = x_xi * y_eta - x_eta * y_xi
      xi_x  =  y_eta / J
      xi_y  = -x_eta / J
      eta_x = -y_xi / J
      eta_y =  x_xi / J

    Here derivatives are computed using vectorized finite differences in index space.
    """
    # Derivatives in computational index directions (uniform indexing).
    x_xi = np.gradient(x, axis=0)
    x_eta = np.gradient(x, axis=1)
    y_xi = np.gradient(y, axis=0)
    y_eta = np.gradient(y, axis=1)

    jac = x_xi * y_eta - x_eta * y_xi
    jac = np.where(np.abs(jac) < 1e-12, 1e-12, jac)

    xi_x = y_eta / jac
    xi_y = -x_eta / jac
    eta_x = -y_xi / jac
    eta_y = x_xi / jac

    return jac, xi_x, xi_y, eta_x, eta_y


def primitives_to_conservative(rho, u, v, p, gamma):
    """Convert primitive variables to conservative state vector Q = [rho, rho*u, rho*v, E]."""
    e_internal = p / (gamma - 1.0)
    e_kinetic = 0.5 * rho * (u ** 2 + v ** 2)
    e_total = e_internal + e_kinetic
    q = np.stack([rho, rho * u, rho * v, e_total], axis=0)
    return q


def conservative_to_primitives(q, gamma, r_gas):
    """Convert conservative variables to primitive variables with positivity protection."""
    rho = np.maximum(q[0], 1e-8)
    u = q[1] / rho
    v = q[2] / rho
    e_total = q[3]

    e_kinetic = 0.5 * rho * (u ** 2 + v ** 2)
    p = np.maximum((gamma - 1.0) * (e_total - e_kinetic), 1e-6)
    t = p / (rho * r_gas)
    a = np.sqrt(np.maximum(gamma * p / rho, 1e-12))
    mach = np.sqrt(u ** 2 + v ** 2) / np.maximum(a, 1e-12)

    return rho, u, v, p, t, a, mach


def physical_fluxes(rho, u, v, p, e_total):
    """
    Euler flux vectors in physical Cartesian coordinates:

    F = [ rho*u,
          rho*u^2 + p,
          rho*u*v,
          (E+p)*u ]

    G = [ rho*v,
          rho*u*v,
          rho*v^2 + p,
          (E+p)*v ]
    """
    f = np.stack(
        [
            rho * u,
            rho * u * u + p,
            rho * u * v,
            (e_total + p) * u,
        ],
        axis=0,
    )

    g = np.stack(
        [
            rho * v,
            rho * u * v,
            rho * v * v + p,
            (e_total + p) * v,
        ],
        axis=0,
    )
    return f, g


def transformed_fluxes(q, gamma, r_gas, jac, xi_x, xi_y, eta_x, eta_y):
    """
    Compute transformed contravariant fluxes used in mapped-grid Euler equations.

    Q_hat = Q / J
    F_hat = (xi_x * F + xi_y * G) / J
    G_hat = (eta_x * F + eta_y * G) / J
    """
    rho, u, v, p, _, _, _ = conservative_to_primitives(q, gamma, r_gas)
    e_total = q[3]
    f, g = physical_fluxes(rho, u, v, p, e_total)

    f_hat = (xi_x[None, :, :] * f + xi_y[None, :, :] * g) / jac[None, :, :]
    g_hat = (eta_x[None, :, :] * f + eta_y[None, :, :] * g) / jac[None, :, :]
    q_hat = q / jac[None, :, :]

    return q_hat, f_hat, g_hat


def apply_boundary_conditions(q, gamma, r_gas, p0, t0, p_back, dhdx_wall):
    """
    Apply 2D nozzle BCs on conservative state in-place:

    1) Inlet (i=0): impose stagnation-based inflow estimate.
       - Use interior static pressure to infer an inlet Mach from isentropic relation.
       - Reconstruct inlet static (p, T, rho), set v=0 and u=M*a.

    2) Outlet (i=-1): pressure outlet for subsonic, extrapolation for supersonic.

    3) Walls (j=0 and j=-1): exact slip condition V.n = 0 using local wall slope.
       For wall y_w(x), normal is proportional to (-dy_w/dx, 1), so:
         u*(-dy_w/dx) + v*(1) = 0  ->  v = u*dy_w/dx
       Top wall has dy_w/dx = +dh/dx, bottom has dy_w/dx = -dh/dx.
       Thermodynamic state is extrapolated from nearest interior line.
    """
    rho, u, v, p, _, a, mach = conservative_to_primitives(q, gamma, r_gas)

    # Inlet (left boundary): stagnation-conditioned inflow
    p_int = np.maximum(p[1, :], 1e-6)
    m_in = np.sqrt(
        np.maximum(
            (2.0 / (gamma - 1.0)) * ((p0 / p_int) ** ((gamma - 1.0) / gamma) - 1.0),
            1e-8,
        )
    )
    m_in = np.clip(m_in, 0.02, 2.0)

    t_in = t0 / (1.0 + 0.5 * (gamma - 1.0) * m_in ** 2)
    p_in = p0 / (1.0 + 0.5 * (gamma - 1.0) * m_in ** 2) ** (gamma / (gamma - 1.0))
    rho_in = p_in / (r_gas * t_in)
    a_in = np.sqrt(gamma * p_in / rho_in)
    u_in = m_in * a_in
    v_in = np.zeros_like(u_in)

    q[:, 0, :] = primitives_to_conservative(rho_in, u_in, v_in, p_in, gamma)

    # Recompute primitives after inlet update for outlet classification.
    rho, u, v, p, _, a, mach = conservative_to_primitives(q, gamma, r_gas)

    # Outlet (right boundary): pressure BC if subsonic, extrapolation if supersonic.
    is_subsonic_out = mach[-2, :] < 1.0

    rho_out = rho[-2, :].copy()
    u_out = u[-2, :].copy()
    v_out = v[-2, :].copy()
    p_out = p[-2, :].copy()

    p_out[is_subsonic_out] = p_back

    q[:, -1, :] = primitives_to_conservative(rho_out, u_out, v_out, p_out, gamma)

    # Wall BCs (bottom j=0, top j=-1), exact slip projection using wall slope.
    rho, u, v, p, _, _, _ = conservative_to_primitives(q, gamma, r_gas)

    # Bottom wall: y_w = -h(x), so dy_w/dx = -dh/dx
    slope_bottom = -dhdx_wall
    u_bottom = u[:, 1]
    v_bottom = u_bottom * slope_bottom
    rho_bottom = rho[:, 1]
    p_bottom = p[:, 1]
    q[:, :, 0] = primitives_to_conservative(rho_bottom, u_bottom, v_bottom, p_bottom, gamma)

    # Top wall: y_w = +h(x), so dy_w/dx = +dh/dx
    slope_top = dhdx_wall
    u_top = u[:, -2]
    v_top = u_top * slope_top
    rho_top = rho[:, -2]
    p_top = p[:, -2]
    q[:, :, -1] = primitives_to_conservative(rho_top, u_top, v_top, p_top, gamma)

    return q


def compute_dt(q, gamma, r_gas, x, y, cfl):
    """
    Compute a global explicit time step from local spectral radius estimate.

    dt <= CFL / max( |u|/dx + |v|/dy + a*sqrt(1/dx^2 + 1/dy^2) )

    We estimate local dx, dy from physical grid spacing magnitudes.
    """
    rho, u, v, _, _, a, _ = conservative_to_primitives(q, gamma, r_gas)

    dx_local = np.sqrt(np.gradient(x, axis=0) ** 2 + np.gradient(y, axis=0) ** 2)
    dy_local = np.sqrt(np.gradient(x, axis=1) ** 2 + np.gradient(y, axis=1) ** 2)

    dx_local = np.maximum(dx_local, 1e-8)
    dy_local = np.maximum(dy_local, 1e-8)

    spectral = np.abs(u) / dx_local + np.abs(v) / dy_local + a * np.sqrt(1.0 / dx_local ** 2 + 1.0 / dy_local ** 2)
    s_max = np.max(spectral)
    dt = cfl / max(s_max, 1e-8)
    return dt


def macormack_step(q, gamma, r_gas, x, y, jac, xi_x, xi_y, eta_x, eta_y, dt, dhdx_wall, p0, t0, p_back):
    """
    Advance one explicit time step with vectorized MacCormack predictor-corrector.

    Predictor (forward differences):
      Q*_hat = Q_hat^n - dt * (dF_hat/dxi|forward + dG_hat/deta|forward)

    Corrector (backward differences on predicted state):
      Q^{n+1}_hat = 0.5*(Q_hat^n + Q*_hat - dt*(dF*_hat/dxi|backward + dG*_hat/deta|backward))

    Spatial operations are fully vectorized with numpy slicing; no loops over i/j.
    """
    q = apply_boundary_conditions(q, gamma, r_gas, p0, t0, p_back, dhdx_wall)

    q_hat, f_hat, g_hat = transformed_fluxes(q, gamma, r_gas, jac, xi_x, xi_y, eta_x, eta_y)

    # Predictor: forward differences on interior nodes.
    dfdxi_f = f_hat[:, 1:-1, 1:-1] - f_hat[:, 0:-2, 1:-1]
    dgdeta_f = g_hat[:, 1:-1, 1:-1] - g_hat[:, 1:-1, 0:-2]

    q_hat_star = q_hat.copy()
    q_hat_star[:, 1:-1, 1:-1] = q_hat[:, 1:-1, 1:-1] - dt * (dfdxi_f + dgdeta_f)

    # Convert predicted mapped state back to physical conservative state.
    q_star = q_hat_star * jac[None, :, :]
    q_star = apply_boundary_conditions(q_star, gamma, r_gas, p0, t0, p_back, dhdx_wall)

    q_hat_s, f_hat_s, g_hat_s = transformed_fluxes(q_star, gamma, r_gas, jac, xi_x, xi_y, eta_x, eta_y)

    # Corrector: backward differences on predicted fluxes.
    dfdxi_b = f_hat_s[:, 2:, 1:-1] - f_hat_s[:, 1:-1, 1:-1]
    dgdeta_b = g_hat_s[:, 1:-1, 2:] - g_hat_s[:, 1:-1, 1:-1]

    q_hat_new = q_hat.copy()
    q_hat_new[:, 1:-1, 1:-1] = 0.5 * (
        q_hat[:, 1:-1, 1:-1]
        + q_hat_star[:, 1:-1, 1:-1]
        - dt * (dfdxi_b + dgdeta_b)
    )

    # Small Jameson-like Laplacian smoothing to stabilize high-frequency numerical noise.
    # This is fully vectorized and acts only on interior cells.
    eps2 = 0.001
    lap = (
        q_hat_new[:, 2:, 1:-1]
        + q_hat_new[:, :-2, 1:-1]
        + q_hat_new[:, 1:-1, 2:]
        + q_hat_new[:, 1:-1, :-2]
        - 4.0 * q_hat_new[:, 1:-1, 1:-1]
    )
    q_hat_new[:, 1:-1, 1:-1] += eps2 * lap

    q_new = q_hat_new * jac[None, :, :]
    q_new = apply_boundary_conditions(q_new, gamma, r_gas, p0, t0, p_back, dhdx_wall)

    return q_new


def run_solver(
    nx,
    ny,
    n_iter,
    cfl,
    gamma,
    r_gas,
    p0,
    t0,
    p_back,
    length_conv,
    length_div,
    h_in,
    h_throat,
    h_out,
    tol,
):
    """Main CFD driver: grid, initialization, time marching, convergence tracking."""
    x, y, h, dhdx = create_nozzle_grid(nx, ny, length_conv, length_div, h_in, h_throat, h_out)
    jac, xi_x, xi_y, eta_x, eta_y = compute_metrics(x, y)

    # Uniform low-Mach initialization from inlet stagnation conditions.
    m0 = 0.2
    t_init = t0 / (1.0 + 0.5 * (gamma - 1.0) * m0 ** 2)
    p_init = p0 / (1.0 + 0.5 * (gamma - 1.0) * m0 ** 2) ** (gamma / (gamma - 1.0))
    rho_init = p_init / (r_gas * t_init)
    a_init = np.sqrt(gamma * p_init / rho_init)
    u_init = m0 * a_init

    rho = rho_init * np.ones((nx, ny))
    u = u_init * np.ones((nx, ny))
    v = np.zeros((nx, ny))
    p = p_init * np.ones((nx, ny))

    q = primitives_to_conservative(rho, u, v, p, gamma)
    q = apply_boundary_conditions(q, gamma, r_gas, p0, t0, p_back, dhdx)

    residual_history = []

    for _ in range(n_iter):
        q_old = q.copy()
        dt = compute_dt(q, gamma, r_gas, x, y, cfl)

        q = macormack_step(
            q,
            gamma,
            r_gas,
            x,
            y,
            jac,
            xi_x,
            xi_y,
            eta_x,
            eta_y,
            dt,
            dhdx,
            p0,
            t0,
            p_back,
        )

        # L2 residual on density as steady-state monitor.
        res = np.sqrt(np.mean((q[0, 1:-1, 1:-1] - q_old[0, 1:-1, 1:-1]) ** 2))
        residual_history.append(res)

        if res < tol:
            break

    rho, u, v, p, t, _, mach = conservative_to_primitives(q, gamma, r_gas)

    return {
        "x": x,
        "y": y,
        "h": h,
        "rho": rho,
        "u": u,
        "v": v,
        "p": p,
        "t": t,
        "mach": mach,
        "residual": np.array(residual_history),
    }


# ------------------------------
# Streamlit app
# ------------------------------

st.set_page_config(page_title="CFD 2D de Tobera - Euler MacCormack", layout="wide")
st.title("CFD 2D de Tobera Convergente-Divergente")
st.markdown(
    """
Esta aplicacion resuelve las **ecuaciones de Euler 2D compresibles e inviscidas** sobre una malla estructurada
algebraica usando un esquema **MacCormack predictor-corrector vectorizado**.

Todas las actualizaciones espaciales del solucionador usan slicing de NumPy (sin bucles sobre la malla espacial).
"""
)


def build_export_csv(results):
    """
    Build CSV text for centerline and both walls (top/bottom).

    Output columns:
      dataset, x_m, y_m, rho_kg_m3, u_m_s, v_m_s, p_pa, t_k, mach
    """
    x = results["x"]
    y = results["y"]
    rho = results["rho"]
    u = results["u"]
    v = results["v"]
    p = results["p"]
    t = results["t"]
    mach = results["mach"]

    j_center = x.shape[1] // 2

    datasets = {
        "linea_central": (slice(None), j_center),
        "pared_inferior": (slice(None), 0),
        "pared_superior": (slice(None), -1),
    }

    lines = ["dataset,x_m,y_m,rho_kg_m3,u_m_s,v_m_s,p_pa,t_k,mach"]
    for name, (ii, jj) in datasets.items():
        block = np.column_stack(
            [
                x[ii, jj],
                y[ii, jj],
                rho[ii, jj],
                u[ii, jj],
                v[ii, jj],
                p[ii, jj],
                t[ii, jj],
                mach[ii, jj],
            ]
        )
        rows = [f"{name},{','.join(f'{val:.10e}' for val in row)}" for row in block]
        lines.extend(rows)

    return "\n".join(lines)


def build_geometry_figure(x, y):
    """Create Plotly figure showing nozzle contour and structured grid lines."""
    fig = go.Figure()

    # Streamwise grid lines
    for j in range(y.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=x[:, j],
                y=y[:, j],
                mode="lines",
                line=dict(color="rgba(50, 80, 120, 0.35)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Transverse grid lines
    for i in range(x.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=x[i, :],
                y=y[i, :],
                mode="lines",
                line=dict(color="rgba(50, 80, 120, 0.25)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x[:, -1],
            y=y[:, -1],
            mode="lines",
            name="Pared superior",
            line=dict(color="#B22222", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[:, 0],
            y=y[:, 0],
            mode="lines",
            name="Pared inferior",
            line=dict(color="#B22222", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[:, x.shape[1] // 2],
            y=y[:, y.shape[1] // 2],
            mode="lines",
            name="Linea central",
            line=dict(color="#0B6E4F", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Geometria 2D y malla estructurada",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(title="Elementos"),
    )

    return fig


def build_physical_contour(x_2d, y_2d, z_2d, x_1d, h_1d, title, color_scale, cb_title):
    """
    Interpolate solution to a Cartesian canvas and mask values outside nozzle walls.
    This renders contours strictly inside the physical nozzle geometry.
    """
    x_cart = np.linspace(np.min(x_1d), np.max(x_1d), 200)
    y_cart = np.linspace(np.min(y_2d), np.max(y_2d), 100)
    X_grid, Y_grid = np.meshgrid(x_cart, y_cart)

    points = np.column_stack((x_2d.ravel(), y_2d.ravel()))
    values = z_2d.ravel()
    Z_grid = griddata(points, values, (X_grid, Y_grid), method="linear")

    h_limit = np.interp(x_cart, x_1d, h_1d)
    mask_outside = np.abs(Y_grid) > h_limit[None, :]
    Z_masked = np.where(mask_outside, np.nan, Z_grid)

    fig = go.Figure(
        data=go.Contour(
            x=x_cart,
            y=y_cart,
            z=Z_masked,
            colorscale=color_scale,
            contours=dict(showlabels=True),
            colorbar=dict(title=cb_title),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_1d,
            y=h_1d,
            mode="lines",
            name="Pared superior",
            line=dict(color="black", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_1d,
            y=-h_1d,
            mode="lines",
            name="Pared inferior",
            line=dict(color="black", width=3),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def area_mach_function(mach, gamma):
    """Isentropic area-Mach relation A/A* as a function of Mach."""
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach ** 2)
    exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / np.maximum(mach, 1e-12)) * term ** exp


def solve_mach_from_area_ratio(area_ratio, gamma, supersonic_branch):
    """Solve isentropic Mach for a given area ratio using bisection."""
    if area_ratio <= 1.0 + 1e-10:
        return 1.0

    if supersonic_branch:
        low, high = 1.0 + 1e-8, 20.0
    else:
        low, high = 1e-8, 1.0 - 1e-8

    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = area_mach_function(mid, gamma) - area_ratio
        f_low = area_mach_function(low, gamma) - area_ratio

        if f_low * f_mid <= 0.0:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


def compute_mach_comparison_1d(results, gamma, r_gas):
    """
    Compare two 1D Mach estimates along x:
    1) Isentropic 1D Mach from area ratio A/A*.
    2) Mach from section-averaged velocity magnitude and averaged sound speed.
    """
    x = results["x"]
    h = results["h"]
    u = results["u"]
    v = results["v"]
    t = results["t"]

    x_1d = x[:, 0]
    area = 2.0 * h
    area_ratio = area / np.maximum(np.min(area), 1e-12)
    i_throat = int(np.argmin(area))

    mach_iso = np.zeros_like(x_1d)
    for i, ar in enumerate(area_ratio):
        branch_is_supersonic = i > i_throat
        mach_iso[i] = solve_mach_from_area_ratio(ar, gamma, branch_is_supersonic)

    vmag = np.sqrt(u ** 2 + v ** 2)
    v_mean = np.mean(vmag, axis=1)
    t_mean = np.mean(t, axis=1)
    a_mean = np.sqrt(np.maximum(gamma * r_gas * t_mean, 1e-12))
    mach_vel_mean = v_mean / np.maximum(a_mean, 1e-12)

    return x_1d, mach_iso, mach_vel_mean


def build_mach_comparison_figure(x_1d, mach_iso, mach_vel_mean):
    """Create comparison plot between 1D isentropic and section-mean Mach."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_1d,
            y=mach_iso,
            mode="lines",
            name="Mach 1D isentropico (A/A*)",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_1d,
            y=mach_vel_mean,
            mode="lines",
            name="Mach por velocidad media de seccion",
            line=dict(color="#d62728", width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title="Comparativa de numero de Mach por seccion",
        xaxis_title="x [m]",
        yaxis_title="Numero de Mach [-]",
        legend=dict(title="Curvas"),
    )
    return fig


tab1, tab2, tab3 = st.tabs([
    "1. Modelado de Geometría",
    "2. Condiciones de Entrada",
    "3. Cálculo y Resultados",
])

# ------------------------------
# Tab 1: Geometria + malla
# ------------------------------
with tab1:
    st.subheader("Definicion de la tobera y resolucion de malla")

    unidad_geom = st.radio(
        "Unidad geometrica",
        options=["m", "mm"],
        horizontal=True,
    )

    if unidad_geom == "m":
        factor_a_m = 1.0
        fmt_long = "%.6f"
        fmt_alt = "%.6f"
        cfg = {
            "l_conv_default": 0.5,
            "l_div_default": 0.5,
            "h_in_default": 0.18,
            "h_throat_default": 0.08,
            "h_out_default": 0.16,
            "l_min": 1e-4,
            "h_min": 1e-5,
            "l_step": 0.005,
            "h_step": 0.001,
        }
    else:
        factor_a_m = 1e-3
        fmt_long = "%.3f"
        fmt_alt = "%.3f"
        cfg = {
            "l_conv_default": 500.0,
            "l_div_default": 500.0,
            "h_in_default": 180.0,
            "h_throat_default": 80.0,
            "h_out_default": 160.0,
            "l_min": 0.1,
            "h_min": 0.01,
            "l_step": 1.0,
            "h_step": 0.1,
        }

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        length_conv_ui = st.number_input(
            f"Longitud convergente [{unidad_geom}]",
            value=cfg["l_conv_default"],
            min_value=cfg["l_min"],
            step=cfg["l_step"],
            format=fmt_long,
        )
        h_in_ui = st.number_input(
            f"Semialtura de entrada [{unidad_geom}]",
            value=cfg["h_in_default"],
            min_value=cfg["h_min"],
            step=cfg["h_step"],
            format=fmt_alt,
        )
    with col_g2:
        length_div_ui = st.number_input(
            f"Longitud divergente [{unidad_geom}]",
            value=cfg["l_div_default"],
            min_value=cfg["l_min"],
            step=cfg["l_step"],
            format=fmt_long,
        )
        h_throat_ui = st.number_input(
            f"Semialtura de garganta [{unidad_geom}]",
            value=cfg["h_throat_default"],
            min_value=cfg["h_min"],
            step=cfg["h_step"],
            format=fmt_alt,
        )
        h_out_ui = st.number_input(
            f"Semialtura de salida [{unidad_geom}]",
            value=cfg["h_out_default"],
            min_value=cfg["h_min"],
            step=cfg["h_step"],
            format=fmt_alt,
        )

    # Todas las variables geometricas se convierten internamente a metros para el solver.
    length_conv_m = length_conv_ui * factor_a_m
    length_div_m = length_div_ui * factor_a_m
    h_in_m = h_in_ui * factor_a_m
    h_throat_m = h_throat_ui * factor_a_m
    h_out_m = h_out_ui * factor_a_m

    st.caption(
        f"Longitud total de tobera: {length_conv_ui + length_div_ui:.3f} {unidad_geom} "
        f"({length_conv_m + length_div_m:.6f} m)"
    )

    st.markdown("### Resolucion de la malla")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        nx = st.slider("Nx (puntos en x)", min_value=41, max_value=241, value=121, step=10)
    with col_m2:
        ny = st.slider("Ny (puntos en y)", min_value=21, max_value=101, value=53, step=4)

    x_prev, y_prev, _, _ = create_nozzle_grid(nx, ny, length_conv_m, length_div_m, h_in_m, h_throat_m, h_out_m)
    fig_geom = build_geometry_figure(x_prev, y_prev)
    st.plotly_chart(fig_geom, use_container_width=True)

    if h_throat_m >= min(h_in_m, h_out_m):
        st.warning("Para una tobera convergente-divergente valida, la garganta debe ser menor que entrada y salida.")

# ------------------------------
# Tab 2: Condiciones de entrada
# ------------------------------
with tab2:
    st.subheader("Parametros termodinamicos y condiciones de contorno")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        p0 = st.number_input("Presion de estancamiento P0 [Pa]", value=101325.0 * 3.0, min_value=1000.0, step=1000.0)
        t0 = st.number_input("Temperatura de estancamiento T0 [K]", value=300.0, min_value=100.0, step=5.0)
        p_back = st.number_input("Presion de respaldo Pb [Pa]", value=101325.0, min_value=100.0, step=500.0)
    with col_f2:
        r_gas = st.number_input("Constante de gas R [J/(kg*K)]", value=287.0, min_value=1.0, step=1.0)
        gamma = st.number_input("Relacion de calores especificos gamma [-]", value=1.4, min_value=1.01, max_value=2.0, step=0.01)

# ------------------------------
# Tab 3: Controles + resultados
# ------------------------------
with tab3:
    st.subheader("Configuracion numerica y ejecucion")

    perf_mode = st.selectbox(
        "Modo de rendimiento",
        options=["Personalizado", "Grueso (Mas rapido)", "Fino (Mayor resolucion)"],
        index=0,
    )

    presets = {
        "Grueso (Mas rapido)": {"n_iter": 900, "cfl": 0.40},
        "Fino (Mayor resolucion)": {"n_iter": 2500, "cfl": 0.30},
    }

    if perf_mode == "Personalizado":
        n_iter = st.slider("Iteraciones maximas", min_value=200, max_value=5000, value=1500, step=100)
        cfl = st.slider("Numero CFL", min_value=0.05, max_value=0.9, value=0.35, step=0.05)
    else:
        cfg = presets[perf_mode]
        n_iter = cfg["n_iter"]
        cfl = cfg["cfl"]
        st.info(f"Modo activo: Iteraciones maximas={n_iter}, CFL={cfl:.2f}.")

    tol = st.number_input("Tolerancia de convergencia (L2 de densidad)", value=1e-6, min_value=1e-10, format="%.1e")

    run_button = st.button("Ejecutar Simulacion", type="primary")

    if run_button:
        with st.spinner("Ejecutando solucionador MacCormack vectorizado..."):
            results = run_solver(
                nx=nx,
                ny=ny,
                n_iter=n_iter,
                cfl=cfl,
                gamma=gamma,
                r_gas=r_gas,
                p0=p0,
                t0=t0,
                p_back=p_back,
                length_conv=length_conv_m,
                length_div=length_div_m,
                h_in=h_in_m,
                h_throat=h_throat_m,
                h_out=h_out_m,
                tol=tol,
            )

        x = results["x"]
        y = results["y"]
        h = results["h"]
        mach = results["mach"]
        p = results["p"]
        t = results["t"]
        residual = results["residual"]
        x_1d = x[:, 0]

        st.success(f"Simulacion finalizada. Iteraciones ejecutadas: {len(residual)}")

        c1, c2 = st.columns(2)

        with c1:
            fig_m = build_physical_contour(
                x_2d=x,
                y_2d=y,
                z_2d=mach,
                x_1d=x_1d,
                h_1d=h,
                title="Contorno de numero de Mach",
                color_scale="Turbo",
                cb_title="Numero de Mach",
            )
            st.plotly_chart(fig_m, use_container_width=True)

        with c2:
            fig_p = build_physical_contour(
                x_2d=x,
                y_2d=y,
                z_2d=p,
                x_1d=x_1d,
                h_1d=h,
                title="Contorno de presion estatica",
                color_scale="Viridis",
                cb_title="Presion estatica [Pa]",
            )
            st.plotly_chart(fig_p, use_container_width=True)

        fig_t = build_physical_contour(
            x_2d=x,
            y_2d=y,
            z_2d=t,
            x_1d=x_1d,
            h_1d=h,
            title="Contorno de temperatura estatica",
            color_scale="Plasma",
            cb_title="Temperatura estatica [K]",
        )
        st.plotly_chart(fig_t, use_container_width=True)

        x_cmp, mach_iso, mach_vel_mean = compute_mach_comparison_1d(results, gamma, r_gas)
        fig_cmp = build_mach_comparison_figure(x_cmp, mach_iso, mach_vel_mean)
        st.plotly_chart(fig_cmp, use_container_width=True)

        diff_abs = np.abs(mach_vel_mean - mach_iso)
        st.markdown(
            f"""
**Resumen comparativo (por seccion):**
- Error absoluto medio: {np.mean(diff_abs):.4f}
- Error absoluto maximo: {np.max(diff_abs):.4f}
"""
        )

        fig_r = go.Figure(
            data=go.Scatter(y=residual, mode="lines", name="Residuo L2 (densidad)")
        )
        fig_r.update_layout(
            title="Historia de convergencia",
            xaxis_title="Iteracion",
            yaxis_title="Residuo L2",
            yaxis_type="log",
        )
        st.plotly_chart(fig_r, use_container_width=True)

        csv_text = build_export_csv(results)
        st.download_button(
            label="Descargar datos de linea central y paredes (CSV)",
            data=csv_text,
            file_name="datos_tobera_linea_central_paredes.csv",
            mime="text/csv",
        )
    else:
        st.info("Configure los parametros y pulse 'Ejecutar Simulacion' para calcular y visualizar resultados.")
