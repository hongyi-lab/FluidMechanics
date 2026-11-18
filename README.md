# FluidMechanics: committor and reaction coordinates of the MFE nine-mode shear-flow model

Each `gateN_*` folder holds the figures and a README for one gate. `gate3b_mfe` and `gate4_dns` also hold the full report (Chinese) and the run outputs; `gate3c_mfe` and `gate3d_mfe` hold the run outputs (results in their READMEs). `gate4_dns`–`gate6_dns` are the full-DNS checks (Gate 4a solver checks; Gate 4b parameter choice; Gates 5–6 first gate, second gate and composite; report `gate5_dns/report_zh.md`); `gate7_dns` is Gate 7 (is the fast channel a numerical artefact or a real route? report `gate7_dns/report_zh.md`); `gate8_dns` is Gate 8 (blind test of a third condition, the fast oblique-wave route; its 8c part completes Gate 7; report `gate8_dns/report_zh.md`); `gate9_dns` is Gate 9 (blind test of the combined rule on 28 rays outside the nine patterns; in progress, frozen predictions in `gate9_dns/results_9_2/`); `overview` is a one-page overview figure of the project; `cossu` is the Part A comparison with Cossu et al. (2011).

- `code/`: the shared package `committor` (`code/src/committor`), one script per gate (`code/scripts`), configs (`code/configs`) and tests (`code/tests`).
- Install: `pip install -e .` from this folder. The environment is described in `envs/` (PyTorch float64 on a CUDA GPU).
- Gate 3b, for example: `python code/scripts/gate3b_mfe.py --config code/configs/gate3b_mfe.yaml --run-dir <dir>`.
- The scripts read the Gate 2 / Gate 3 run folders, which are not in this repository (data size).
