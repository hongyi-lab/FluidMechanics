# Environment

Isolated conda env (`<env>` = any prefix you choose):

    conda create -p <env> python=3.11
    <env>/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch
    <env>/bin/pip install numpy scipy scikit-learn matplotlib tqdm pyyaml pytest
    <env>/bin/pip install -e .          # from the repository root

Exact versions: `requirements-freeze.txt` (torch 2.14.1+cu126, RTX A6000, driver 580).
