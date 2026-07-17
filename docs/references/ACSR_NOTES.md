# ACSR reference notes

Primary source for the adaptive stimulation-artifact method in
`neural_signal_toolkit.artifacts.acsr`:

Kim M, Moon Y, Hunt J, et al. (2021).
A Novel Technique to Reject Artifact Components for Surface EMG Signals
Recorded During Walking With Transcutaneous Spinal Cord Stimulation: A Pilot Study.
Frontiers in Human Neuroscience 15:660583.
https://doi.org/10.3389/fnhum.2021.660583

Author MATLAB implementation:
https://github.com/mjkim0927/kim-frontiers-2021

Default parameters used in this toolkit (matching the paper):
- window length N = 200 ms
- overlap t = 100 ms
- training length NS ≈ 3 s (optional via `ACSRParams.train_ms`)
