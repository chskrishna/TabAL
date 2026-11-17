# Figure captions (single-split precursor)

These figures summarize the archived 1,200-trajectory precursor dataset. They
must not be presented as the final three-split, 2,700-trajectory paper results.

**Figure 1. Representation effects on active-learning efficiency.** (a) Paired
RDKit2D-minus-ECFP4 normalized AULC differences, averaged within each of 50
dataset–seed units; intervals are 95% paired bootstrap intervals. The overall
precursor gain is 0.0700 [0.0583,
0.0818]. (b) Mean paired gain for every learner–policy
combination. (c) Macro AULC averaged over acquisition policies. Positive values
favor RDKit2D.

**Figure 2. Learning curves by representation, learner, and acquisition
policy.** Lines show mean test AUROC across 50 dataset–seed units; shaded bands
show 95% bootstrap intervals. Label budgets range from 32 to 256.

**Figure 3. Acquisition-policy effects relative to random acquisition.** Points
show paired normalized AULC differences for entropy and entropy-plus-diversity
relative to random acquisition; intervals are 95% paired bootstrap intervals
over 50 dataset–seed units. Positive values favor the informed policy.

**Figure S1. Runtime of complete active-learning trajectories.** Box plots show
the distribution of total recorded fit, prediction, and acquisition-scoring
time over dataset–seed trajectories. The logarithmic axis reflects the large
model-dependent runtime range. Measurements are specific to the archived
Windows/NVIDIA RTX A4000 execution environment.
