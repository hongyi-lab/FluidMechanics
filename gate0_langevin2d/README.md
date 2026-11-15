# Gate 0 figures: 2D Langevin positive control (2026-11-15)

Committor pipeline check on dX = -grad(V+U) dt + sqrt(2 eps) dW, V = (x^2-1)^2, U = y^2/2, eps = 0.25,
A = {x <= -1}, B = {x >= 1}; exact committor depends on x only. 17,245 states, M = 200 paths each.

- `fig1_q_vs_best_coordinate.png` — MC committor vs x on test states, isotonic map (train) and exact q(x); skill 0.993.
- `fig2_committor_histogram_test.png` — true q of test states predicted q in [0.4, 0.6], for x, E, y, |y|.
- `fig3_q_vs_E.png` — MC committor vs E = ((x+1)^2 + y^2)/2.
- `fig0_mc_vs_exact_q.png` — MC estimate vs exact q, q vs x, randomised PIT.
