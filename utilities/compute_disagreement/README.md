# compute_disagreement/compute_disagreement.py

This script computes the disagreement within a dataset. That is, how many people disagree with the majority label. Ties don't matter here, since regardless of which label is chosen, an equal number of people will disagree.

(`test_compute_disagreement.py` is left in as a small test script to show how it works on a small dataset.)

`compute_disagreement.py` is an executable script, and descriptions of the parameters can be found via `--help` or by looking at the code.

While this script can be used on its own, in our pipeline, the `get_disagreement` function is merely called by the `split_by_disagreement` script. See that folder for more details on how to use it.