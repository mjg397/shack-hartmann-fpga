import sys

with open('full_pipeline_sim/slope_calculation.v', 'r') as f:
    lines = f.readlines()

with open('full_pipeline_sim/slope_calculation.v', 'w') as f:
    for line in lines:
        if 'wire signed [27:0] x_ref = slopes_ref_x_mem[subapetures_completed];' in line:
            f.write(line.replace('[subapetures_completed]', '[subapetures_completed - 1]'))
        elif 'wire signed [27:0] y_ref = slopes_ref_y_mem[subapetures_completed];' in line:
            f.write(line.replace('[subapetures_completed]', '[subapetures_completed - 1]'))
        else:
            f.write(line)
