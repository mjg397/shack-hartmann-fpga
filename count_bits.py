with open('full_pipeline_sim/data/subaperture_bitmap.hex', 'r') as f:
    h = f.read().strip()
b = bin(int(h, 16))[2:]
print("Number of 1s:", b.count('1'))
