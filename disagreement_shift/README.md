# disagreement_shift
This folder contains all the code to generate the website that has the examples from Figure 5 and Appendix I (Figures 8-13).

To do this, you will first need to have both the Kernel and NCF imputations.

Then, use `disagreement_shift/label_distribution` to compute the KL divergence values between the original and imputed data. The same script will also compute the distributional/soft labels in the process. Afterwards, follow `/website/README.md` to understand how to rename the results from the previous script and where to copy the files to, and then run the `website` script which will launch the site. Finally, visit the site, which will automatically download the outputs.