The `run.sh` is generally not up to date - you should view `run.bat` for how to run this.

This script (`run.bat` / `main.py`) computes the KL Divergence from the original distribution to the imputed distribution. In doing so, it also outputs the distributional labels for the original and imputed data.

The data output by this script is usually used by scripts in the `../website` or the `../kl_distribution` folder. See their `README.md` to understand what they do and how they work.

UPDATE 5/23/2023 `run.bat` and the `compare_kl` scripts have been updated to include Multitask. However, the other code (`kl_distribution` and `website`) have not been.