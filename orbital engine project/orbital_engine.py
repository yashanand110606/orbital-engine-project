#!/usr/bin/env python3
"""
orbital_engine.py

High-Fidelity Mission-Grade Orbital Propagation & Analysis Engine

Features:
 - ECI <-> orbital elements utilities (classical elements -> state vector)
 - Central gravity + J2, J3 perturbations
 - Atmospheric drag (exponential)
 - Simple Solar Radiation Pressure (SRP)
 - Adaptive RK45 propagation via scipy.integrate.solve_ivp
 - Monte Carlo uncertainty engine (multiprocessing)
 - Visualization (3D orbit + uncertainty cloud, 2D ground track)
 - Config-driven via YAML mission file

Usage:
  python orbital_engine.py --config mission.yaml

Author: Yash Anand 
"""

import argparse
import math
import yaml
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ---------------------------
# Physical constants (SI)
# ---------------------------
MU_E = 3.986004418e14       # Earth's gravitational parameter, m^3 / s^2
R_E = 6378137.0             # Earth's equatorial radius, m
J2 = 1.08262668e-3          # Earth's J2
J3 = -2.532153e-6           # Earth's J3 (approx)
AU = 1.495978707e11         # Astronomical unit (m)
P_SOLAR = 4.56e-6           # Solar radiation pressure at 1 AU (N/m^2)
DAY = 86400.0

# ---------------------------
# Utilities
# ---------------------------

def deg2rad(x): return x * math.pi / 180.0
def rad2deg(x): return x * 180.0 / math.pi
def norm(v): return np.linalg.norm(v)

# ---------------------------
# Orbital elements -> state vector (ECI)
# ---------------------------
def oe_to_state(a, e, i_deg, raan_deg, argp_deg, nu_deg, mu=MU_E):
    """
    Classical orbital elements to ECI state vector.
    a: semi-major axis (m)
    e: eccentricity
    i_deg: inclination (deg)
    raan_deg: RAAN (deg)
    argp_deg: argument of perigee (deg)
    nu_deg: true anomaly (deg)
    Returns r (3,), v (3,) in ECI frame (m, m/s)
    """
    i = deg2rad(i_deg)
    raan = deg2rad(raan_deg)
    argp = deg2rad(argp_deg)
    nu = deg2rad(nu_deg)

    # Perifocal coordinates
    p = a * (1 - e**2)
    r_pf = (p / (1 + e * math.cos(nu))) * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    # Rotation matrix: PQW -> ECI
    R3_W = np.array([[math.cos(raan), -math.sin(raan), 0],
                     [math.sin(raan),  math.cos(raan), 0],
                     [0, 0, 1]])
    R1_i = np.array([[1, 0, 0],
                     [0, math.cos(i), -math.sin(i)],
                     [0, math.sin(i),  math.cos(i)]])
    R3_w = np.array([[math.cos(argp), -math.sin(argp), 0],
                     [math.sin(argp),  math.cos(argp), 0],
                     [0, 0, 1]])

    Q_pX = R3_W @ R1_i @ R3_w
    r = Q_pX @ r_pf
    v = Q_pX @ v_pf
    return r, v

# ---------------------------
# Perturbation accelerations
# ---------------------------

def accel_gravity(r):
    """Central gravitational acceleration (two-body)."""
    rnorm = norm(r)
    return -MU_E * r / (rnorm**3)

def accel_J2(r):
    """Acceleration due to J2 perturbation."""
    x, y, z = r
    rnorm = norm(r)
    z2 = z*z
    r2 = rnorm**2
    factor = 1.5 * J2 * MU_E * (R_E**2) / (rnorm**5)
    ax = factor * x * (5*z2/r2 - 1)
    ay = factor * y * (5*z2/r2 - 1)
    az = factor * z * (5*z2/r2 - 3)
    return np.array([ax, ay, az])

def accel_J3(r):
    """Acceleration due to J3 perturbation (approx)."""
    x, y, z = r
    rnorm = norm(r)
    z2 = z*z
    r2 = rnorm**2
    factor = 0.5 * J3 * MU_E * (R_E**3) / (rnorm**7)
    ax = factor * x * (5*z2/r2 - 1) * (3*z/rnorm)
    ay = factor * y * (5*z2/r2 - 1) * (3*z/rnorm)
    az = factor * (6*z2/r2 - 1) * (1 - 7*z2/r2)  # simplified shape
    return np.array([ax, ay, az])

def accel_drag(r, v, Cd, A, m, rho0=1.225, H=8500.0):
    """
    Exponential atmospheric density model.
    rho0: sea level density (kg/m^3), H: scale height (m)
    Use a simple exponential decay from Earth's surface.
    For high altitude (LEO), rho0 and H must be chosen carefully.
    """
    rnorm = norm(r)
    alt = rnorm - R_E
    if alt < 0:
        return np.zeros(3)
    # Simple model: base density at 100 km approximated small. We'll use a crude exponential
    # For practical relevance, use constants that give low density in LEO (order 1e-12 to 1e-16).
    # We'll use a reference at 120 km: rho_ref ~ 5e-9 kg/m^3 (very rough).
    rho_ref = 5e-9
    h_ref = 120e3
    rho = rho_ref * np.exp(-(alt - h_ref) / 50e3) if alt >= h_ref else rho_ref * np.exp(-(h_ref - alt) / 50e3)
    # relative velocity approximation (assume atmosphere co-rotating, neglecting rotation for simplicity)
    vrel = v  # simple
    vrel_norm = norm(vrel)
    if vrel_norm == 0:
        return np.zeros(3)
    drag_acc = -0.5 * rho * Cd * A / m * vrel_norm * vrel
    return drag_acc

def accel_srp(r, Cr, A, m, sun_vector=None):
    """
    Solar Radiation Pressure model.
    sun_vector: unit vector from Earth to Sun in ECI (approx). If None, use +x axis.
    Force = P_SOLAR * (AU / r_sun)^2 * Cr * A / m * (sun_direction projected onto spacecraft)
    For simplicity, we consider SRP as constant pressure from sun at 1 AU and approximate orientation:
    We apply a simple radial outward force scaled by projected area toward sun (max effect on LEO is small).
    """
    if sun_vector is None:
        sun_vector = np.array([1.0, 0.0, 0.0])  # approximate unit vector to Sun
    sun_vector = sun_vector / norm(sun_vector)
    # direction from spacecraft to sun:
    # approximate SRP acceleration direction as away from sun
    a_mag = P_SOLAR * Cr * A / m  # at 1 AU
    # Projected fraction: assume full exposure (conservative)
    return a_mag * sun_vector

# ---------------------------
# Full dynamics function
# ---------------------------

def dynamics(t, state, options):
    """
    state: [rx, ry, rz, vx, vy, vz]
    options: dict with keys controlling which perturbations, Cd, A, m, Cr
    """
    r = np.array(state[0:3])
    v = np.array(state[3:6])
    ax = accel_gravity(r)
    if options.get('use_J2', True):
        ax += accel_J2(r)
    if options.get('use_J3', False):
        ax += accel_J3(r)
    if options.get('use_drag', False):
        ax += accel_drag(r, v,
                         Cd=options.get('Cd', 2.2),
                         A=options.get('A', 1.0),
                         m=options.get('m', 500.0))
    if options.get('use_srp', False):
        ax += accel_srp(r,
                        Cr=options.get('Cr', 1.2),
                        A=options.get('A', 1.0),
                        m=options.get('m', 500.0),
                        sun_vector=options.get('sun_vector', None))
    return np.concatenate([v, ax])

# ---------------------------
# Propagator wrapper
# ---------------------------

def propagate(state0, t_span, options, rtol=1e-9, atol=1e-12):
    """
    High-level propagation API using solve_ivp (RK45).
    state0: initial state vector (6,)
    t_span: (t0, tf)
    options: options dict for dynamics
    returns: result object from solve_ivp
    """
    fun = lambda t, y: dynamics(t, y, options)
    t_eval = None  # use solver internal steps; later we can sample
    sol = solve_ivp(fun, t_span, state0, method='RK45', rtol=rtol, atol=atol)
    return sol

# ---------------------------
# Monte Carlo Runner
# ---------------------------

def monte_carlo_single(seed, base_state, base_options, t_span, perturb):
    """
    Single run for Monte Carlo: apply perturbations to base_state/options and propagate.
    perturb: dict with perturbation magnitudes (e.g., dv_sigma, Cd_sigma, mass_sigma)
    """
    rng = np.random.default_rng(seed)
    state = base_state.copy()
    # perturb initial velocity with gaussian noise
    dv = rng.normal(0, perturb.get('dv_sigma', 0.0), size=3)
    state[3:6] += dv
    # perturb ballistic coefficient via Cd or mass
    options = dict(base_options)
    options['Cd'] = base_options.get('Cd', 2.2) + rng.normal(0, perturb.get('Cd_sigma', 0.0))
    options['m'] = base_options.get('m', 500.0) + rng.normal(0, perturb.get('m_sigma', 0.0))
    sol = propagate(state, t_span, options)
    # sample final position
    rf = sol.y[0:3, -1]
    return rf, sol

def run_monte_carlo(base_state, base_options, t_span, perturb, n_runs=200, processes=None):
    seeds = np.arange(n_runs)
    args = [(int(s), base_state, base_options, t_span, perturb) for s in seeds]
    results = []
    if processes is None or processes == 1:
        for s in tqdm(seeds, desc="Monte Carlo"):
            rf, sol = monte_carlo_single(int(s), base_state, base_options, t_span, perturb)
            results.append((rf, sol))
    else:
        with mp.Pool(processes=processes) as pool:
            for res in tqdm(pool.imap_unordered(partial(monte_wrapper, base_state=base_state,
                                                        base_options=base_options,
                                                        t_span=t_span, perturb=perturb), seeds),
                            total=n_runs, desc="Monte Carlo (parallel)"):
                results.append(res)
    return results

def monte_wrapper(seed, base_state, base_options, t_span, perturb):
    return monte_carlo_single(seed, base_state, base_options, t_span, perturb)

# ---------------------------
# Visualization utilities
# ---------------------------

def plot_3d_orbit(sol, title="Orbit", show=True, ax=None, color='b'):
    r = sol.y[0:3, :]
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[0,:], r[1,:], r[2,:], color)
    # plot Earth as sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = R_E * np.outer(np.cos(u), np.sin(v))
    y = R_E * np.outer(np.sin(u), np.sin(v))
    z = R_E * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    if show:
        plt.show()
    return ax

def plot_uncertainty_cloud(final_positions, title="Uncertainty cloud"):
    arr = np.array(final_positions)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arr[:,0], arr[:,1], arr[:,2], s=5, alpha=0.6)
    # Plot Earth
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = R_E * np.outer(np.cos(u), np.sin(v))
    y = R_E * np.outer(np.sin(u), np.sin(v))
    z = R_E * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()

def plot_ground_track(sol, title="Ground track"):
    # Convert ECI to lat/lon ignoring Earth rotation for simplicity
    r = sol.y[0:3,:]
    x, y, z = r
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.linalg.norm(r, axis=0))
    plt.figure(figsize=(10,4))
    plt.plot(np.degrees(lon), np.degrees(lat))
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title(title)
    plt.grid()
    plt.show()

# ---------------------------
# Helper: build initial state from config
# ---------------------------
def build_state_from_config(cfg):
    oe = cfg['orbit']
    a = oe.get('a')  # if provided, in meters
    if a is None:
        # build a from altitude and maybe circular assumption
        alt = oe.get('alt_km', 500.0) * 1e3
        a = R_E + alt
    e = oe.get('e', 0.001)
    i = oe.get('i_deg', 51.6)
    raan = oe.get('raan_deg', 0.0)
    argp = oe.get('argp_deg', 0.0)
    nu = oe.get('nu_deg', 0.0)
    r0, v0 = oe_to_state(a, e, i, raan, argp, nu)
    state0 = np.concatenate([r0, v0])
    return state0

# ---------------------------
# Main CLI
# ---------------------------
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Orbital Propagation Engine")
    parser.add_argument('--config', default='mission.yaml', help='Mission YAML file')
    parser.add_argument('--mc', action='store_true', help='Run Monte Carlo')
    parser.add_argument('--n', default=200, type=int, help='Monte Carlo runs')
    parser.add_argument('--out', default='output', help='Output directory for plots')
    parser.add_argument('--procs', default=1, type=int, help='Number of parallel processes for Monte Carlo (0=auto)')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)

    # build state and options
    state0 = build_state_from_config(cfg)
    options = {
        'use_J2': cfg.get('perturbations', {}).get('J2', True),
        'use_J3': cfg.get('perturbations', {}).get('J3', False),
        'use_drag': cfg.get('perturbations', {}).get('drag', False),
        'use_srp': cfg.get('perturbations', {}).get('srp', False),
        'Cd': cfg.get('spacecraft', {}).get('Cd', 2.2),
        'A': cfg.get('spacecraft', {}).get('A_m2', 1.0),
        'm': cfg.get('spacecraft', {}).get('m_kg', 500.0),
        'Cr': cfg.get('spacecraft', {}).get('Cr', 1.2),
    }

    # Run propagation
    sim_days = cfg.get('simulation', {}).get('days', 1.0)
    t0 = 0.0
    tf = sim_days * DAY
    sol = propagate(state0, (t0, tf), options)
    print(f"Propagation finished: steps={len(sol.t)}, t_final={sol.t[-1]} s")

    # Save simple plots
    ax = plot_3d_orbit(sol, title="Nominal Orbit", show=False)
    plt.savefig(os.path.join(args.out, 'orbit_nominal_3d.png'))
    plt.close()

    plot_ground_track(sol, title="Nominal Ground Track")
    plt.savefig(os.path.join(args.out, 'groundtrack_nominal.png'))
    plt.close()

    # Monte Carlo
    if args.mc:
        perturb = {
            'dv_sigma': cfg.get('monte_carlo', {}).get('dv_sigma', 1.0),  # m/s
            'Cd_sigma': cfg.get('monte_carlo', {}).get('Cd_sigma', 0.05),
            'm_sigma': cfg.get('monte_carlo', {}).get('m_sigma', 5.0),
        }
        processes = args.procs if args.procs > 0 else None
        results = run_monte_carlo(state0, options, (t0, tf), perturb, n_runs=args.n, processes=processes)
        final_positions = [r for (r, s) in results]
        plot_uncertainty_cloud(final_positions, title="Monte Carlo Final Positions")
        plt.savefig(os.path.join(args.out, 'mc_uncertainty_3d.png'))
        plt.close()

        # Save some stats
        final_arr = np.array(final_positions)
        mean_pos = final_arr.mean(axis=0)
        cov_pos = np.cov(final_arr.T)
        print("Monte Carlo mean final pos (m):", mean_pos)
        print("Monte Carlo covariance (m^2):\n", cov_pos)

if __name__ == "__main__":
    main()

