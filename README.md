# orbital-engine-project
High-Fidelity Orbital Propagation & Mission Analysis Engine

A mission-grade orbital mechanics simulation engine implementing multi-perturbation orbital dynamics, adaptive numerical integration, Monte Carlo uncertainty analysis, and visualization for satellite mission analysis.

This project is designed to mimic real aerospace mission analysis workflows used in flight dynamics and trajectory simulation environments.

 Project Overview

This engine simulates satellite orbital motion around Earth while accounting for real-world perturbations and uncertainties. It provides tools for:

* High-accuracy orbit propagation
* Perturbation modeling (J2, J3, Drag, Solar Radiation Pressure)
* Monte Carlo uncertainty propagation
* Statistical analysis using covariance matrices
* Ground track and 3D orbit visualization

 Motivation

Modern space missions require accurate modeling of:

* Non-spherical Earth gravity effects
* Atmospheric drag in Low Earth Orbit
* Solar radiation pressure disturbances
* Launch and system uncertainty propagation

This project demonstrates how physics-based simulation integrates with numerical methods and software engineering practices to create mission analysis tools.

 Physics & Models Implemented

 Gravity Model

* Two-body central gravity
* J2 perturbation (Earth oblateness)
* J3 higher order gravity effect

 Atmospheric Drag

* Exponential atmospheric density model
* Velocity-dependent drag force

Solar Radiation Pressure

* Constant SRP force model at 1 AU
* Configurable spacecraft reflectivity

 Numerical Methods

* Adaptive RK45 ODE Solver (`scipy.solve_ivp`)
* High precision tolerances for orbital stability
* Time-domain propagation

 Monte Carlo Simulation

The engine supports statistical uncertainty analysis by simulating multiple mission scenarios with random perturbations in:

* Initial velocity
* Drag coefficient
* Spacecraft mass

Outputs include:

* Mean final position
* Covariance matrix
* 3D uncertainty cloud visualization

Visualization Features

3D Orbit Visualization

Shows satellite trajectory around Earth.

 Ground Track Plot

Latitude vs Longitude path over Earth surface.

Monte Carlo Uncertainty Cloud

Shows possible final satellite positions due to uncertainties.

Software Architecture

* Config-driven mission setup (YAML)
* Modular physics modeling
* Parallel Monte Carlo execution support
* Reproducible simulation pipeline

Project Structure

orbital_engine_project/
│
├ orbital_engine.py
├ mission.yaml
├ output/
│   ├ orbit_nominal_3d.png
│   ├ groundtrack_nominal.png
│   ├ mc_uncertainty_3d.png
```

---

## ⚙ Installation

### Requirements

* Python 3.9+
* NumPy
* SciPy
* Matplotlib
* PyYAML
* tqdm

Install dependencies:

```bash
pip install numpy scipy matplotlib pyyaml tqdm
```
 Running the Simulation
Nominal Orbit Simulation

```bash
python orbital_engine.py --config mission.yaml
```
Monte Carlo Uncertainty Simulation

```bash
python orbital_engine.py --config mission.yaml --mc
```

Example Outputs

* Orbital trajectory visualization
* Satellite ground track
* Monte Carlo uncertainty cloud
* Position covariance matrix

Real-World Applications

* Satellite mission design
* Orbit lifetime prediction
* Launch error sensitivity analysis
* Navigation uncertainty modeling
* Earth observation coverage planning
* Re-entry risk analysis

 Aerospace Relevance

This project reflects core concepts used in:

* Flight Dynamics Software
* Mission Analysis Tools
* Orbit Determination Systems
* Trajectory Simulation Frameworks

Future Improvements

Planned extensions include:

* J4 gravity perturbation
* Earth rotation (ECI ↔ ECEF transformations)
* GPU acceleration
* C++ physics solver backend
* Covariance ellipse visualization
* Real-time orbit animation

Author

**Yash Anand**
BSc Physics | Orbital Mechanics & Simulation Enthusiast
Aspiring Software Engineer (Dynamics)

 Key Skills Demonstrated

* Orbital Mechanics
* Numerical Simulation
* Scientific Computing
* Monte Carlo Methods
* Aerospace Software Design
* Data Visualization

Highlight

This project demonstrates end-to-end development of a mission-style orbital simulation engine combining physics modeling, numerical methods, and software architecture design.


👉 SpaceX-style project description
👉 GitHub project badges + polish

Just tell me 😄🚀
