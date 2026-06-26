# Simple timing constraint for the PL-only Shack-Hartmann pipeline.
# Default target is 100 MHz. Change the period if your board clock differs.

create_clock -name clk -period 10.000 [get_ports clk]
