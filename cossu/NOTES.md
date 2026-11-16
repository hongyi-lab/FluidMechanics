# 20261117_cossu_partA_v1: threshold curves of Cossu, Brandt, Bagheri & Henningson (2011)

**Goal:** a purely descriptive reading of the paper's critical A_U–A_W curves, redrawn against ln A_W. No pass/fail criteria.

**Paper:** Phys. Fluids 23, 074103 (2011), doi 10.1063/1.3614480.
- Local PDF: `literature/papers/Cossu_Brandt_Bagheri_Henningson_2011_PoF_23_074103.pdf`, sha256 `0f3ab744…b9ca7`.
- This is the **authors' preprint dated 16 June 2011** (mech.kth.se/~luca/papers/AUAW.pdf), not the AIP typeset version. Figure and page numbers below refer to this preprint.

**Script:** `code/scripts/cossu_partA.py`.
- Run with `python -I code/scripts/cossu_partA.py experiments/20261117_cossu_partA_v1`.
- CPU only; it takes a few seconds.
- The exact settings are in `config_used.json`; the printed summary is in `log.txt`.

## Files

| File | Content |
|---|---|
| `cossu_threshold_lnAW.png/.pdf` | **Main figure.** A_U (horizontal) against ln A_W,crit (vertical), with A_W in % on the right axis. Shows Couette (Fig. 5b) and Blasius (Fig. 9b), the fitted lines, the A_W = A_U line and the lower-branch saddle. |
| `cossu_all_panels_lnY.png/.pdf` | Same view for every threshold panel: Figs. 5b/9b (A_U–A_W), 5a/9a (u_rms–w_rms), and the toy model of Fig. 2. |
| `check_linear_axes.png` | The extracted points on the paper's own linear axes, for comparison with the PDF by eye (they match). |
| `extracted_points.csv`, `extracted_points.json` | All points in % as plotted, plus ln(y/100). The JSON also holds the calibration ticks, labels, residuals and the frame of each panel. |
| `description.json` | All fits: every upper-range cut-off, the named ranges, the two-segment fit, local slopes, the y = x crossing, the rank-paired rms/amplitude ratios and the toy-model fixed point. |

## Extraction (vector graphics, no raster digitisation)

- **Data points.** All plots are gnuplot vector graphics; `pdftocairo -svg` renders them losslessly.
  - Each triangle marker is gnuplot's `TriU`: a closed triangle plus a zero-length, round-capped dot (`Pnt`) drawn at the exact data position.
  - The dot is taken as the data point. Its position matches the gnuplot geometry: apex at +1.12 vpt, base at −0.50 vpt.
  - The lower-branch saddle is a filled Bézier circle; its centre is used.
- **Axis calibration.**
  - Each major tick on the left and bottom frame edges is paired with the nearest numeric tick label from `pdftotext -bbox`, which uses the same page coordinates.
  - A straight line is then fitted per axis.
  - Maximum calibration residual: ≤ 1.3e-3 (in the plotted % units) on every axis.
- **Digitisation error.** Per point: ≤ 1.7e-3 % (calibration residual plus the 1/256 pt coordinate quantisation of the SVG).
  - Propagated by Monte Carlo, this changes the fitted slopes by ≤ 2e-4 per %-pt, which is negligible.
  - The real error of the thresholds is the bisection tolerance. The paper does not state it.
- **Independent check (toy model, Fig. 2).** The fixed point of eq. (1) at R = 300 with the paper's coefficients gives the lower branch at (100u, 100w) = (3.511, 0.765).
  - The extracted filled circle is at (3.493, 0.756).
  - The difference is about 0.25 pt on the page, which is larger than the extraction precision. So it is in the plotted value, not in the extraction.
- **Pairing of panels.** Panels (a) and (b) of Figs. 5 and 9 have the same number of points (8 and 13).
  - Paired by rank, u_rms/A_U is almost constant (see the definitions section). This supports the reading that they are the same bisection cases.
  - The paper does not list the pairing. It does refer to "the third point from the right in Fig. 9" across panels.

## Setup recorded from the paper

### Plane Couette (Sec. III, Figs. 4–6)
- Walls at y = ±h move at ±U_w; Re = U_w h/ν = **500**.
- Box **L_x = 2.5π, L_z = π** (units of h).
- DNS with Simson (Fourier–Chebyshev–Fourier, RK3/Crank–Nicolson), grid 32 × 65 × 32. U_ref = **2U_w**.
- **Primary streak.**
  - The DNS starts from U_C + A_1 u_V^(opt), where u_V^(opt) are the linear optimal streamwise vortices (α = 0).
  - "Here we denote by U1(y, z, A1) the streaky velocity field u1 extracted at the time where it reaches its peak perturbation energy."
- **Secondary perturbation** (Schoppa & Hussain form), eq. (2): **u_W = C {0, 0, g(y) sin(αx)}**.
  - "with g(y) = (1 − y²) and C a normalizing constant ensuring that ∥u_W∥ = 1."
  - It is a z-independent spanwise velocity that bends the streak sinuously. It is a prescribed shape, not an eigenmode and not an optimal.
  - Its amplitude A_2 multiplies u_W, and it is added to U1 at the streak's peak-energy time.
  - α is not stated explicitly. The fundamental for this box would be 2π/L_x = 0.8.
- **Lower-branch saddle** (Nagata lower branch): A_U = 22.0 %, A_W = 1.79 % (Fig. 5b); 100u_rms = 10.12, 100w_rms = 0.517 (Fig. 5a).

### Blasius boundary layer (Sec. IV, Figs. 7–9)
- Re_δ* = **700**, U_ref = U_e.
- Box **L_x = 9δ\*** ("αδ* = 0.7"), **L_z = 5.737δ\*** ("βδ* = 1.1"). The box height and the grid are not stated.
- Local parallel-flow assumption, with a body force that keeps the Blasius profile from diffusing.
- **Primary streak.** The optimal steady perturbations of Andersson et al. (1999) are used as inflow with amplitude A_1, and evolved by DNS and nonlinear PSE.
  - U1(y, z, A1) = U0(y) + u1 is extracted at x_max, where the streak energy is largest.
- **Secondary perturbation:** eq. (2) with, as printed, "g(y) = y/(πσ²))^{−1/2} exp[−((y − y0)²/(σ²)]" (parentheses unbalanced in the original), σ = 1.7, y0 = 1.8.
- No converged edge state, so no saddle symbol is drawn.

### Toy model (Sec. II, Fig. 2)
- Waleffe's four-variable model, eq. (1a–d), R = 300.
- Coefficients: [k_m, k_u, k_v, k_w] = [1.57, 2.28, 2.77, 2.67]; [σ_u, σ_v, σ_w] = [1.29, 0.22, 0.68]; σ_m = 0.
- Streak instability threshold U_c = k_w²/(Rσ_w), so 100U_c = 3.5.
- Secondary perturbation: A_2 u_W with u_W = [0, 0, 1, 0].

## Definitions (exact quotes, preprint p. 5)

> "In the discussion above we have used the rms values of each velocity component, defined as u²_rms = ∫_Ω (u − U_C)² dΩ/U²_ref and w²_rms = ∫_Ω w² dΩ/U²_ref but the results remain almost unchanged when other measures of the streak and of the secondary perturbation amplitudes are considered. In particular in Fig. 5b the critical curve is reported in terms of the amplitudes A_U − A_W defined as:
> A_U = (max_{y,z} ΔU − min_{y,z} ΔU)/2 U_ref; (3a)
> A_W = (max_{x,y,z} W − min_{x,y,z} W)/2 U_ref, (3b)
> where we choose as reference velocity the difference of the velocity of the two walls: U_ref = 2U_w. A_U is the primary streak amplitude as defined for streamwise uniform streaks [13] in terms of the maximum and minimum of the distortion of the streamwise velocity ΔU(y, z) = U(y, z) − U_C(y). A_W is defined in a similar way to measure the amplitude of sinuous secondary perturbations."

What these definitions mean:
- **A_U** is half the peak-to-peak of the streamwise-velocity distortion over (y, z), divided by U_ref.
  - It is a pointwise (max–min) amplitude, not an L2 norm, and it is normalised by the wall-velocity difference (Couette) or U_e (BL), not by ‖U‖.
- **A_W** is half the peak-to-peak of the total spanwise velocity W over (x, y, z), divided by U_ref.
  - It is a **velocity amplitude, not an energy**. While W is dominated by the added mode, A_W is proportional to A_2.
- Both are values **of the initial condition**, at the moment the sinuous perturbation is added. The streak then evolves freely: "whose amplitude is not frozen in our simulations".

**Rank-paired ratios** (my computation from the extracted points; the paper does not give these):
- Couette u_rms/A_U = 0.475–0.478; BL 0.269–0.285.
- Couette w_rms/A_W = 0.5165 at the four smallest A_U.
  - This equals √(8/15 · 1/2) = 0.5164, the value for W = C(1 − y²) sin(αx) when ∫_Ω is read as a volume average, i.e. when "rms" is a true rms.
  - At A_U = 20.8, 24.5, 27.9 and 31.1 % the ratio falls to 0.494, 0.442, 0.414 and 0.472. So at large A_U the initial W is not only the added mode; the paper does not say this.
  - BL w_rms/A_W = 0.322–0.326 throughout.
- **Derived (mine, Couette only):** with ‖U_C‖_rms = U_w/√3, ‖u_streak‖_rms/‖U_C‖_rms = 2√3 u_rms ≈ **1.65 A_U**.

## Extracted threshold points (A_U %, A_W,crit %)

- **Couette, Fig. 5b** (8 points): (4.35, 9.599) (8.65, 5.333) (12.86, 3.585) (16.93, 2.335) (20.82, 1.379) (24.48, 0.776) (27.92, 0.508) (31.13, 0.466).
- **Blasius, Fig. 9b** (13 points): (2.28, 7.958) (4.54, 6.499) (6.76, 6.001) (8.91, 5.139) (10.98, 4.541) (13.96, 3.621) (17.16, 3.020) (20.19, 2.539) (22.98, 2.100) (24.31, 1.900) (25.57, 1.699) (28.85, 1.140) (31.74, 0.640).
- Figs. 5a, 9a and 2 are in the CSV.

## The three descriptive answers

ln is taken of A_W as a fraction. Slopes are d ln A_W / d A_U per percentage point of A_U; ×100 gives the slope per unit A_U. "±" is the OLS standard error; the digitisation contribution is < 1e-4.

### (1) Is ln A_W,crit linear in A_U on the large-A_U side, and what is the slope?

Approximately yes, between the paper's cross-over (A_W = A_U) and A_U ≈ 26–28 %. Above that the last 1–2 points bend away: Couette flattens, Blasius steepens.

| Curve | Fit range (points) | Slope per %-pt | R² | Max abs. residual in ln |
|---|---|---|---|---|
| **Couette, headline** | A_U 8.65–27.92 % (6) | **−0.125 ± 0.006** (−12.5 ± 0.6 per unit A_U; e-folding 8.0 %-pt) | **0.992** | 0.11 |
| Couette, all right of cross-over | 8.65–31.13 % (7) | −0.117 ± 0.006 | 0.987 | – |
| Couette, including the leftmost point | 4.35–27.92 % (7) | −0.123 ± 0.004 | 0.995 | – |
| Couette, all points | 4.35–31.13 % (8) | −0.118 ± 0.004 | 0.992 | – |
| **Blasius, headline** | A_U 6.76–25.57 % (9) | **−0.0653 ± 0.0009** (−6.5 ± 0.1 per unit A_U; e-folding 15.3 %-pt) | **0.9985** | 0.03 |
| Blasius, all right of cross-over | 6.76–31.74 % (11) | −0.079 ± 0.006 | 0.953 | – |
| Blasius, including the two leftmost points | 2.28–25.57 % (11) | −0.0644 ± 0.0009 | 0.998 | – |
| Blasius, all points | 2.28–31.74 % (13) | −0.075 ± 0.005 | 0.962 | – |

- **Top end.** A continuous two-segment fit puts the only change of slope at the top end.
  - Couette: A_U = 27.9 %, slope −0.123 → −0.045 (one point beyond).
  - Blasius: A_U = 27.2 %, slope −0.064 → −0.200 (two points beyond).
  - These breaks are near the paper's linear streak-instability amplitude, A_U ≈ 25 %.
- **The two flows differ.** The Couette and Blasius slopes differ by a factor of about 1.9. This contrasts with the paper's statement that the critical curve is "quantitatively similar" for the two flows, which was made on linear axes.
- **rms panels.**
  - Couette (Fig. 5a), 100u_rms 4.11–13.34: slope −0.285 ± 0.017 per %-pt of u_rms (R² 0.986). Converted with u_rms ≈ 0.476 A_U, this is about −0.136 per %-pt of A_U.
  - BL (Fig. 9a), 2.43–7.08: slope −0.233 ± 0.004 (R² 0.998), about −0.063 per %-pt of A_U.
  - Both agree with the A_U fits.
- **Toy model (Fig. 2).** ln w_c is not straight: its local slope steepens steadily from −0.22 to −0.47 per %-pt of 100u. A straight fit on all points gives R² 0.986.

### (2) Does A_W,crit tend to a constant as A_U → 0?

**The data do not reach small A_U, and there is no plateau.**
- **Smallest points:**
  - Couette: A_U = 4.35 %, A_W = 9.60 %.
  - Blasius: A_U = 2.28 %, A_W = 7.96 %.
- **No flattening at the low end:**
  - Couette: the local ln slope over the lowest pair is −0.137 per %-pt (dA_W/dA_U = −0.99).
  - Blasius: the local slopes over the lowest three pairs are −0.089, −0.036 and −0.072 (dA_W/dA_U = −0.64 for the lowest pair). The two lowest Blasius points sit on the same straight line (R² 0.998 including them).
- **Extrapolation only, not data:** continuing the headline lines to A_U = 0 gives 17 % (Couette) and 9.2 % (Blasius).
- **Toy model:** the smallest point is 100u = 0.67 (≈ 0.19 U_c), where w_c = 1.57 %.
  - The magnitude of the ln slope decreases toward small u, so the curve flattens somewhat there.
  - But dw_c/du is still −0.31 there, so it is not constant.
- **The paper gives no limiting value.** It only says:
  > "The points on the far left in the plots in Fig. 5 indicate that transition can be induced by an initial secondary sinuous perturbation almost without initial primary streaks or vortices."
  
  (p. 6; similarly p. 3 for the toy model: "transition can be obtained even in the almost complete absence of initial streaks and of initial vortices".)

### (3) Where do the two branches meet?

**No second (flat) branch is visible, so no meeting point can be located from a change of slope at small A_U.** Two points can be reported instead:

- **The paper's "cross-over" (A_W,crit = A_U).** Linear interpolation of ln A_W − ln A_U between the bracketing points gives:
  - Couette: **A_U = A_W ≈ 7.0 %**, between the points at 4.35 and 8.65 %.
  - Blasius: **≈ 6.2 %**, between 4.54 and 6.76 %.
  - For comparison: rms panels 100u_rms = 100w_rms ≈ 3.5 (Couette) and 1.9 (BL); toy model 1.36.
  - The paper says "below A_U ≈ 6%" (Couette, p. 5) and "A_U < 5−6%" (abstract).
  - In the data, the curve passes straight through the cross-over: there is no change of slope there.
- **The only slope change:** the top-end break of the two-segment fit.
  - Couette: A_U ≈ 27.9 %, A_W ≈ 0.54 %.
  - Blasius: A_U ≈ 27.2 %, A_W ≈ 1.57 %.

## What the paper says about the regions (quotes)

- Abstract:
  > "The modal instability of the streaks is responsible for transition for A_U > 25−27% for the considered flows, where sinuous perturbations of amplitude below A_W ≈ 1−2% are sufficient to counteract the streak viscous dissipation and induce breakdown. The critical amplitude of the sinuous perturbations increases when the streamwise streak amplitude is decreased. With secondary perturbations amplitude A_W ≈ 4%, breakdown is induced on stable streamwise streaks with A_U ≈ 13%, following the secondary transient growth scenario first examined by Schoppa & Hussain [...]. A cross-over, where the critical amplitude of the sinuous perturbation becomes larger than the amplitude of streamwise streaks, is observed for streaks of small amplitude A_U < 5−6%. In this case the transition is induced by an initial transient amplification of streamwise vortices, forced by the decaying sinuous mode. This is followed by the growth of the streaks and final breakdown."
- p. 5:
  > "In terms of these amplitudes, the streamwise-uniform streaks become linearly unstable when of A_U > 25%. For primary streak amplitudes above this linear critical value, breakdown is obtained with secondary perturbations of amplitude A_W ≈ 1%. Such a value is determined as a compromise between the viscous dissipation of the streaks (whose amplitude is not frozen in our simulations) and the time it takes for the instability modes to reach energy levels able to trigger new and more detrimental nonlinear interactions."
- The paper uses linear axes throughout. It does not state an exponential or ln-linear dependence, and it does not state an asymptote of A_W at small A_U.

## The user's mapping (marked "to be checked"): what in the paper supports or contradicts it

- **"They impose the streak directly; A_U ↔ our S (a streak amplitude)."**
  - Supported:
    - The streak is part of the initial condition (U1 + A_2 u_W) and is not forced.
    - A_U is the streak amplitude of that initial condition.
  - Qualifications:
    - U1 is a DNS-generated, nonlinearly saturated streak, grown by lift-up from optimal vortices of amplitude A_1 and taken at its peak-energy time. It is the full velocity field at that time, so residual vortices are possible; for the toy model the paper writes "U1 ≈ [U, 0, 0, M]".
    - After t = 0 the streak decays freely.
    - A_U is a max–min amplitude normalised by U_ref (2U_w or U_e), not ‖u_streak‖/‖U‖. For Couette, ‖u_streak‖_rms/‖U_C‖_rms ≈ 1.65 A_U (derived above). The paper's L2-type measure u_rms equals 0.476 A_U (Couette) and about 0.27 A_U (BL).
- **"A_W is the seed."**
  - A_W is the initial amplitude of a prescribed sinuous shape (eq. 2, ‖u_W‖ = 1). It is a velocity amplitude, not an energy: ln A_W = ln A_2 + const, and an energy would be ∝ A_W².
  - The shape is fixed, not optimised (Schoppa & Hussain form). The paper warns of "the artificial type of perturbations we have considered (spatially periodic with a specific initial shape)".
- **"Branch (1) = seed × gain."**
  - The paper attributes large A_U (> 25–27 %) to modal instability and A_U ≈ 13 % to secondary transient growth. Its explanation of the A_W ≈ 1 % level is a "compromise between the viscous dissipation of the streaks [...] and the time it takes for the instability modes to reach energy levels able to trigger [...] nonlinear interactions". This reads naturally as seed × (time-limited) gain, but the paper does not formulate it that way.
  - In the data, the ln-linear stretch is not confined to large A_U. It runs from the smallest A_U (2–4 %) through the cross-over up to ≈ 26–28 %. The large-A_U end (> 27 %, the modal-instability region) is where the curves bend.
- **"Branch (2) = the wave generating its own vortex (the y-y term in S)."**
  - The paper's small-A_U mechanism is "a distinct two-stage mechanism [...]: the initially decaying sinuous mode [...] forces the growth of streamwise vortices [...] that then induce the growth and breakdown of streaks" (p. 6).
  - In the **toy model** this forcing is quadratic in the wave:
    > "the sinuous mode triggers the growth of vortices v through the non-normal coupling in equation (1b): dv/dt = −(k_v²/R)v + σ_v χ", with χ = w² (p. 3–4).
    
    This supports the y-y reading.
  - In the **DNS** the paper's explanation is **linear** in the wave (eqs. 4–5):
    > "Considering, for simplicity, an initial condition in the complete absence of streaks u = U1(y), v = 0, w = w2(x, y) [...] Dω_x/Dt = ∂u/∂y ω_y. (5) The tilting of the initial wall-normal vorticity ω_y = −∂w2/∂x by the wall normal shear ∂U1/∂y is responsible for the initial amplification of streamwise vorticity".
    
    The vortices produced are streamwise-periodic and generate "streaks of finite length". The paper does not invoke a quadratic (w·w) forcing for the DNS, so this point does not support the y-y-term reading.
  - The paper places this mechanism at A_U < 5–6 % (the cross-over region). But the data show no constant-A_W branch there: the points continue the same ln-linear trend.
- **Other details that bear on the mapping:**
  - The bisection is on A_2 at fixed A_1, i.e. on the wave amplitude, with the streak amplitude fixed.
  - The marginal trajectories approach the lower-branch saddle (Couette) before leaving.
  - The paper notes a "small offset" between the saddle and the critical curve. In the main figure the saddle (22.0 %, 1.79 %) lies above the curve.

## Verdict

Descriptive only, so there is no pass/fail verdict.
- The A_U–ln A_W curves are close to straight from the smallest A_U up to ≈ 26–28 %:
  - Couette: −0.125 ± 0.006 per %-pt (R² 0.992).
  - Blasius: −0.065 ± 0.001 per %-pt (R² 0.998).
- The curves bend only at the top end.
- No small-A_U plateau is reached: the smallest A_U values are 4.4 % and 2.3 %.
- The paper's cross-over (A_W = A_U) is at ≈ 7.0 % (Couette) and 6.2 % (Blasius), and it is not a kink.
- `docs/STATUS.md` and `experiments/README.md` were deliberately not edited, per the task instruction to modify no existing file except `references.bib`.
