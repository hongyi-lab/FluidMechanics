# Part A: threshold curves of Cossu, Brandt, Bagheri & Henningson (2011), redrawn as A_U vs ln A_W (descriptive)

Paper: Phys. Fluids 23, 074103 (2011). The authors' preprint was used; the PDF is not included here.

**Extraction:** the threshold points were read from the PDF's vector graphics, so digitisation error is negligible (< 0.002 % per point). Script: `code/scripts/cossu_partA.py`. Details and quotes: `NOTES.md`. Data: `extracted_points.csv` / `.json`. Fits: `description.json`.

## Answers

| Question | Answer |
|---|---|
| (1) Is ln A_W,crit linear in A_U on the large-A_U branch? | **Yes**: Couette Re = 500 (A_U 8.7–27.9 %): slope −0.125 ± 0.006 per %-pt, R² 0.992. Blasius Re_δ* = 700 (A_U 6.8–25.6 %): −0.0653 ± 0.0009, R² 0.9985. The last 1–2 points above ~27 % bend away. |
| (2) Does A_W,crit tend to a constant as A_U → 0? | **The data do not reach small A_U, and there is no plateau.** Smallest points: Couette A_U 4.35 % with A_W 9.6 %; Blasius A_U 2.28 % with A_W 8.0 %. Both still lie on the straight line. |
| (3) Where do the two branches meet? | **No second branch is visible.** The paper's "cross-over" A_W = A_U (≈ 7.0 % Couette, 6.2 % Blasius) is not a kink in the curve. |

## Setup and definitions
- **Amplitudes (eq. 3):** A_U = half the peak-to-peak of ΔU(y, z), divided by U_ref; A_W = half the peak-to-peak of W(x, y, z), divided by U_ref.
  - Both are values of the initial condition.
  - Both are velocity amplitudes, not energies.
- **Plane Couette:** box 2.5π × π, U_ref = 2U_w.
- **Blasius boundary layer:** box 9δ* × 5.737δ*, U_ref = U_e.
- **Secondary perturbation:** the fixed Schoppa–Hussain shape.

The paper explains its small-A_U regime by linear tilting of the wave vorticity by the base shear (eq. 5). It does not attribute it to a quadratic, wave-generated vortex.
