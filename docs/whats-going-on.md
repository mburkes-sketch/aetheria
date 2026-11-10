# What's Going On? (Conceptual Overview)

This document provides a conceptual, plain-English overview of the astrophysics and machine learning pipeline in Aetheria. It explains the physical motivation, what each script in `src/` does, and how all components fit together.

---

## The Big Picture: Why Simulate Cosmic Dust?

When starlight traverses space and strikes cosmic dust grains—in circumstellar disks, cometary comae, interstellar clouds, or planetary rings—it scatters in different directions. This phenomenon is called **light scattering**.

How light scatters is an optical "fingerprint" governed by three fundamental factors:
1. **Dust particle size & morphology**: Whether grains are individual sub-micron spheres or porous fractal aggregates consisting of thousands of clustered monomers ($N$).
2. **Dust chemical composition**: Governed by the complex refractive index $m = n + ik$, where the real part $n$ dictates phase velocity deflection and the imaginary part $k$ dictates electromagnetic absorption.
3. **Scattering angle ($\theta$) & Phase angle ($\alpha = \pi - \theta$)**: The viewing angle relative to the incident starlight.

By measuring the **intensity** ($S_{11}$) and **linear polarization** ($P = -S_{12}/S_{11}$) of scattered light across phase angles, astronomers can infer what distant cosmic dust is composed of without ever having a physical sample in hand.

### The Computational Challenge
Rigorous numerical electrodynamics simulations—such as the Discrete Dipole Approximation (FASTMM) or the Multiple Sphere T-Matrix method (MSTM)—are computationally expensive, taking hours or days per cluster geometry. 

**Aetheria** trains Artificial Neural Networks (ANNs) on benchmark simulations and laboratory measurements to act as an ultrafast optical surrogate. Given composition, particle size, and angle, the network predicts the full scattering curve in milliseconds.

Here is how the pipeline operates, step-by-step:

---

## 1. Data Ingestion ([`src/data_loader.py`](file:///home/maxmk/aetheria/src/data_loader.py))

Before neural networks can learn, raw data from disparate instruments and simulation codes must be ingested and aligned to a unified schema:

* **FASTMM Simulations (`.h5` files)**: Discrete-dipole simulations of complex cluster aggregates. Filenames encode metadata (e.g., `am_carbon_01size_8k_output.h5` $\implies$ $r = 0.1\ \mu\text{m}$, $N = 8,000$ particles, amorphous carbon).
* **MSTM Simulations (`.dat` files)**: Multi-sphere T-matrix calculations covering dense clusters and wide angular sweeps. Headers or filename conventions supply $(n, k)$.
* **Laboratory Measurements (`.txt` files)**: Experimental goniospectrophotometer measurements of real mineral powders from the Amsterdam-Granada Light Scattering Database (e.g., olivines, basalts, meteorites, palagonite).

### Key Data Normalization & Cleaning Steps:
1. **Intensity Normalization at $30^\circ$**: Different computer simulations and physical laboratory sensors output arbitrary flux scales (raw cross-sections vs. detector millivolts). We normalize intensity by dividing both $S_{11}$ and $S_{12}$ by $S_{11}(30^\circ)$:
   $$S_{11}'(\theta) = \frac{S_{11}(\theta)}{S_{11}(30^\circ)}, \quad S_{12}'(\theta) = \frac{S_{12}(\theta)}{S_{11}(30^\circ)}$$
   Because the common factor $S_{11}(30^\circ)$ cancels out, the physical polarization ratio is strictly preserved:
   $$P(\theta) = -\frac{S_{12}'(\theta)}{S_{11}'(\theta)} = -\frac{S_{12}(\theta)}{S_{11}(\theta)}$$
2. **Filtering Low-Intensity Noise**: At high scattering angles or strong absorption, light intensity $S_{11}$ can drop near zero. Computing polarization involves dividing by $S_{11}$; when $S_{11} \le 10^{-4}$, numerical noise or detector limits cause extreme, unphysical polarization spikes ($|P| \gg 1.0$). Points with $S_{11} \le 10^{-4}$ are filtered out during ingestion.
3. **Dual Dataset Output**:
   - [`data/processed/dust_data.parquet`](file:///home/maxmk/aetheria/data/processed/dust_data.parquet): Pointwise table for standard angle-by-angle MLP models.
   - [`data/processed/dust_function_data.parquet`](file:///home/maxmk/aetheria/data/processed/dust_function_data.parquet): Grouped table containing full angular vectors per dust geometry for sequence and parameter regression.
4. **Diagnostic Verification Plots**: Automatically generates raw scatter plots for each material into [`data/processed/graphs/`](file:///home/maxmk/aetheria/data/processed/graphs/) to visually verify data fidelity prior to preprocessing.

---

## 2. Feature Engineering & Preprocessing ([`src/preprocessing.py`](file:///home/maxmk/aetheria/src/preprocessing.py))

Raw physical numbers vary across dozens of orders of magnitude and non-linear physical regimes. This script engineers physics-informed features tailored for neural networks:

### Physics-Informed Feature Transforms:
* **Trigonometric Angle Decomposition ($\cos\theta, \sin\theta$)**: Scalar angles suffer from artificial boundary discontinuities at $0$ and $2\pi$. Projecting $\theta$ onto the unit circle provides continuous periodic representations.
* **Dimensionless Size Parameter ($x = \frac{2\pi r}{\lambda}$)**: In optics, scattering regimes depend on particle size relative to the illumination wavelength ($\lambda = 0.65\ \mu\text{m}$). This distinguishes Rayleigh scattering ($x \ll 1$), Mie resonance ($x \sim 1\text{--}10$), and geometric optics ($x \gg 10$).
* **Logarithmic Morphological Features**: Aggregate particle count ($N \in [1, 64000]$) and monomer radius ($r \in [0.01, 1.5]\ \mu\text{m}$) are transformed via $\log_{10}(N + 10^{-9})$ and $\log_{10}(r + 10^{-9})$ to prevent large clusters from dominating loss gradients.
* **Log-Scaling & Extended Clipping for Absorption ($k$)**: Imaginary refractive index $k$ spans more than 8 orders of magnitude (from $10^{-8}$ for transparent laboratory minerals to $1.86$ for opaque iron sulfides). We compute:
  $$\text{log\_refractive\_index\_img} = \operatorname{clip}\left(\log_{10}(k + 10^{-9}), -9.0, \infty\right)$$
  The lower clip bound is set to $-9.0$ so that near-transparent materials (e.g., pure water ice at $k \approx 1.43 \times 10^{-7} \implies -6.84$, and `Sample 7_lab` at $k = 10^{-8} \implies -8.0$) remain distinctly separated from the clipping floor while avoiding numerical underflow when $k \to 0$.
* **Arcsinh Transform on Cross-Polarization**: $S_{12}$ oscillates across zero into both positive and negative values. The inverse hyperbolic sine ($\operatorname{arcsinh}$) behaves logarithmically for large magnitudes while remaining strictly smooth and linear around zero, preventing gradient instability.

### Target-Only Outlier Filtering:
* Applies an Interquartile Range (IQR) filter ($3.0 \times \text{IQR}$) **strictly per composition** on target variables (`arcsinh_S12`, `polarization`). Applying this per material prevents discarding valid high-polarization signatures of dark materials (e.g., carbon) when compared with low-polarization minerals.

### Data Splitting & Scaling:
* **80/20 Train/Test Split**: Standard random partition for evaluating model interpolation on seen materials with unseen particle cluster geometries.
* **Holdout Composition Strategy**: Isolates an entire physical material (e.g., `OLIVINE M_lab` or `sample_7_lab`) that the neural network **never sees during training**. This benchmarks true physical generalizability to novel minerals.
* **Hybrid Scaling**: Inputs and $S_{11}, S_{12}$ targets are standardized with `StandardScaler` (zero mean, unit variance). Direct polarization is transformed via `QuantileTransformer` to map its bounded, non-Gaussian distribution to a normal Gaussian profile.

---

## 3. Model Architecture & Training ([`src/train.py`](file:///home/maxmk/aetheria/src/train.py))

Rather than using a single monolithic network, the repository trains **four specialized neural networks**, each architected for the specific mathematical dynamics of light scattering:

1. **Intensity Model ($S_{11}$)**:
   - *Architecture*: Scikit-Learn `MLPRegressor`, hidden layers `(32, 16, 8)`, ReLU activations.
   - *Target*: Standard-scaled $\log_{10}(S_{11})$.
   - *Behavior*: Learns smooth forward diffraction peaks and backscattering slopes.
2. **Cross-Polarization Model ($S_{12}$)**:
   - *Architecture*: Scikit-Learn `MLPRegressor`, hidden layers `(32, 16, 8)`, `tanh` activations.
   - *Target*: Standard-scaled $\operatorname{arcsinh}(S_{12})$.
   - *Behavior*: Hyperbolic tangent activations ensure smooth, non-jagged oscillations across the zero-crossing boundary.
3. **Direct Polarization Model ($P$)**:
   - *Architecture*: Scikit-Learn `MLPRegressor`, hidden layers `(32, 16, 8)`, ReLU activations, reduced learning rate ($10^{-4}$).
   - *Target*: Quantile-transformed polarization.
   - *Behavior*: Captures sharp, localized resonance peaks.
4. **Analytical Polarization Equation Model (`MyRegressorModel`)**:
   - *Framework*: PyTorch deep neural network wrapped via Skorch.
   - *Architecture*: 5 material input properties $\to [128, 128, 64, 32] \to$ 4 output parameters.
   - *Activation*: `nn.SiLU()` (Swish) for smooth physical gradients, with Sigmoid bounds.
   - *Parameters Predicted*: Predicts physical parameters $(b, c_1, c_2, a_0)$ for the astronomical polarization phase function:
     $$P(\alpha) = b \cdot |\sin\alpha|^{c_1} \cdot \left|\cos\frac{\alpha}{2}\right|^{c_2} \cdot \sin\left(\frac{\alpha - a_0}{2}\right)$$
     where $\alpha = \pi - \theta$ is the phase angle, $b$ is the amplitude, $c_1, c_2$ are forward/backscattering exponents, and $a_0$ is the inversion angle.
   - *Custom Loss (`LossEquation`)*: Computes MSE masked to valid angular observations (ignoring $-999.0$ sequence padding) and includes an explicit penalty on maximum peak polarization error ($\alpha \cdot \text{MSE}(P_{\max, \text{pred}}, P_{\max, \text{true}})$).

---

## 4. Evaluation & Visualization Suite ([`src/evaluate.py`](file:///home/maxmk/aetheria/src/evaluate.py))

To verify that predictions adhere to optical physics, [`src/evaluate.py`](file:///home/maxmk/aetheria/src/evaluate.py) computes quantitative benchmarks ($R^2$, MSE, MAE) and generates comparative phase curves across full angular sweeps ($0^\circ \text{ to } 180^\circ$):

### Four Distinct Polarization Evaluation Workflows:
1. **Direct MLP Model**: Evaluates point-by-point neural predictions.
2. **Derived Physical Polarization ($P_{\text{derived}} = -S_{12}/S_{11}$)**: Reconstructs polarization from predicted intensity and cross-polarization elements, verifying internal physical consistency.
3. **Analytical Equation Model**: Uses predicted parameters $(b, c_1, c_2, a_0)$ to generate strictly smooth, physically bounded curves without unphysical oscillations.
4. **Hybrid Fit Model**: Takes direct MLP predictions and fits the analytical equation using `scipy.optimize.curve_fit` to combine neural feature matching with guaranteed curve smoothness.

### Diagnostic Visualizations in `results/`:
The evaluation suite saves comparison plots into organized subdirectories:
- [`results/S11/`](file:///home/maxmk/aetheria/results/S11/): Log-scaled scattering intensity vs. phase angle.
- [`results/S12/`](file:///home/maxmk/aetheria/results/S12/): Cross-polarization element oscillating across zero.
- [`results/polarization/`](file:///home/maxmk/aetheria/results/polarization/): Direct MLP polarization curves vs. ground truth.
- [`results/derived_polarization/`](file:///home/maxmk/aetheria/results/derived_polarization/): Derived ratio $-S_{12}/S_{11}$ vs. ground truth.
- [`results/polarization_equation/`](file:///home/maxmk/aetheria/results/polarization_equation/): Analytical equation curves reconstructed from neural parameters $(b, c_1, c_2, a_0)$.
- [`results/hybrid_polarization/`](file:///home/maxmk/aetheria/results/hybrid_polarization/): Hybrid smoothed curves overlaid on direct MLP predictions and observations.
