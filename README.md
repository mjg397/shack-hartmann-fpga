# shack-hartmann-fpga
Ultra-low-latency, fully streaming FPGA architecture for Shack-Hartmann wavefront sensing and Zernike-based wavefront reconstruction, featuring lenslet-driven centroid extraction and without frame buffering for real-time adaptive optics systems.

The current evidence package is simulation-only. It verifies exact RTL/model arithmetic on selected candidate frames and a 9-cycle final-pixel-to-done event latency in simulation; it does not establish achieved clock timing, resources, power, board operation, physical-sensor performance, or a world-record claim. See the [authoritative final report](evaluation/results/runs/final_evidence_package_v1_20260930_final/REPORT.md).

This research made use of HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments (Por et al. 2018).
