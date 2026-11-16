# Gate 4a, second run (2026-11-17 00:29) - Gate 4 stops here

- Config: `code/configs/gate4_dns.yaml` (criteria unchanged; history line added). Code: `code/src/committor/dns.py` with the
  split nonlinear term; `code/scripts/gate4_dns.py --stage 4a`. Grid 48x24x24. 274 s.
- Result: every 4a criterion passes except item 5 (dt).
  - 1: laminar du/dt 0 (perturbation form) / 2.2e-16 (full form); max |div u| 1.4e-14; symmetry residual 8.7e-15.
  - 2: Galerkin identity max |diff| 1.1e-16 (100 states, |rhs| up to 0.20).
  - 3: decay rel. err a1-1 2.4e-13, a3 1.9e-13, a9 1.4e-12; lift-up streak cos(gamma z) 7.8e-14, cos(pi y) cos(gamma z) 3.1e-13;
       other streak components 4e-8 of the two (nonlinear, O(1e-8)).
  - 4: |<u, N(u)>| / (||u|| ||N||) <= 7.1e-18.
  - 5: FAIL. dt 0.25: NaN (Courant 6.4); 0.125 / 0.0625 / 0.03125: max rel. dE vs dt/2 = 0.27 / 0.28 / 0.37.
  - 6: alone vs batch: dE 3.3e-16, du 0.
- Symmetries shared by all nine modes (numerical): y mirror, shift-reflect, shift-rotate (the user's two are confirmed); none of
  reflect z, shift x Lx/2, shift z Lz/2, both shifts, rotation, reflect x, reflect y.
- Diagnostic (descriptive, `code/scripts/gate4_dt_diagnostic.py`, `dt_diagnostic.json`, `fig_dt_diagnostic.png`): at E(0) = 1e-3
  directions 1, 2, 4 become turbulent (E > 0.1 at t = 72, 25, 27; max E 0.77-0.98); the dt change exceeds 1e-4 at t = 31-99
  for every dt from 0.25 to 0.03125 (max 0.08-0.53); direction 3 decays and passes at dt = 0.25 (8.7e-6). At E(0) = 1e-6 all
  four pass at dt = 0.25 (max 9.4e-6). dt = 0.25 is unstable once turbulent (Courant 6-7).
- Per the user's rule (any Gate 4 criterion fails: stop and report, no change of criteria or method), the pilot, the
  resolution ladder, Gates 5 and 6 were not run.
- Independent check (workflow wf_bbae45db-71e, 2 agents, scripts in the session scratchpad check4a/): failure genuine, no solver
  bug. Lawson RK4 order 4 also in turbulence; energy budget to 3e-18; MFE gives the same B/B/A/B fates for these directions
  (E(0) = 1e-3 is 4x / 21x / 10x above their MFE edges); dt differences grow at 0.06-0.15 per unit time after transition, like a
  1e-10 twin. Energy outside the nine modes reaches 10% by t ~ 1.3-2.3 (linear leakage, identical at E0 = 1e-6). Reporting
  flaws in gate4_dns.py (no effect on item 5): Python max() drops NaN in sym_residual_max / div_max / courant_max (the dt = 0.25
  NaN trajectory is excluded); divergence and Courant sampled every 10 time units; E_range field null. 48x24x24 is marginal in
  transition bursts (flat spectra to the cutoff); on 64x32x32 the transition times shift (direction 4: 27 -> 48).
