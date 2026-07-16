# shack-hartmann-fpga
Ultra-low-latency, fully streaming FPGA architecture for Shack-Hartmann wavefront sensing and Zernike-based wavefront reconstruction, featuring lenslet-driven centroid extraction and without frame buffering for real-time adaptive optics systems.

Currently the world's lowest-latency Shack-Hartmann wavefront sensing and Zernike-based wavefront reconstruction pipeline, with experimentally confirmed latency of 120ns from arrival of final pixel to Zernike output on Altera De1-SoC and sucessful timing closure on Zynq 7000 at 150MHz, with corresponding latency of 53.3ns.

For more information on this project's implementation on the Altera De1-SoC, please refer to the following website. To find more information about implementation on the Zynq 7000 and a more comphrehensive understanding of our novel architecture, be on the lookout from an upcoming preprint.
https://people.ece.cornell.edu/land/courses/ece5760/FinalProjects/s2026/sjb336_mss464_mjg397/sjb336_mss464_mjg397/sjb336_mss464_mjg397/index.html

This research made use of HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments (Por et al. 2018).
