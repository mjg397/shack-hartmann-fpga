# Reciprocal Precision Justification

This note explains why the proposed low-latency reciprocal can use:

```text
8-bit LUT index, 256 entries
one Newton iteration
one DSP per multiply
Q0.25 reciprocal output bus
```

without needing a larger seed table for the current Shack-Hartmann frame.

The short answer is:

```text
The table is already large enough.
The 18-bit correction operand is the precision limiter.
For the current frame, the design gives about 22 guaranteed reciprocal bits
over the nonzero intensity range and less than 0.0001 pixel centroid error.
```

## What "Correct Precision" Means Here

There are three different precision ideas that are easy to mix together:

```text
bus width
guaranteed correct reciprocal bits
scientific output error
```

A Q0.25 reciprocal output means the bus has 25 fractional bit positions. It
does not automatically mean all 25 bits are mathematically correct for every
possible divisor. The scientifically important question is whether the final
centroids, slopes, and Zernike outputs move by an acceptable amount.

For this project, the practical target used in the verification scripts is:

```text
max centroid error < 0.001 pixel
```

The proposed one-Newton design is much better than that target on the current
frame.

## Proposed Low-Latency Reciprocal

The proposed hardware shape is:

```text
a_norm      : Q1.17, 18 bits
x0 seed     : Q0.16 seed from LUT, widened for the multiply
a*x0        : one 18x25 DSP multiply
correction  : 18 bits
x0*corr     : one 25x18 DSP multiply
output      : Q0.25 carried on the reciprocal bus
```

One Newton reciprocal step is:

```text
x1 = x0 * (2 - a*x0)
```

where:

```text
a  = normalized input intensity
x0 = seed approximation to 1/a
x1 = refined approximation to 1/a
```

The two required multiplies are:

```text
a*x0
x0*(2 - a*x0)
```

With the proposed bit widths, each multiply fits the Artix-7 DSP48E1 shape:

```text
18 x 25
25 x 18
```

That means the low-latency implementation uses two DSPs for one Newton step,
not one reused DSP.

## Why One Newton Works

Let the seed have relative error:

```text
x0 = (1/a) * (1 + e)
```

The Newton update is:

```text
x1 = x0 * (2 - a*x0)
```

Substitute the seed error:

```text
a*x0 = 1 + e
x1 = (1/a) * (1 + e) * (1 - e)
x1 = (1/a) * (1 - e^2)
```

So after one Newton step, the main error term is squared:

```text
new error ~= e^2
```

This is why a modest seed table can become very accurate after one Newton
iteration. The table does not need to directly provide 22-25 good bits. It only
needs to be good enough that one Newton step pushes the error below the
scientific tolerance.

## Why a Bigger Table Does Not Fix Everything

Increasing the table improves `x0`. That helps until `x0` is no longer the
dominant source of error.

In the one-DSP-per-multiply design, the correction term is kept to 18 bits:

```text
correction = 2 - a*x0
```

That 18-bit correction is then multiplied by the 25-bit seed/refined value.
Once the LUT seed is good enough, the correction quantization and rounding
become the main limit. Past that point, a bigger table does not buy many more
guaranteed output bits.

This is exactly what the sizing sweep shows.

## Verification Result

The current frame's nonzero subaperture intensities are:

```text
min nonzero sI = 59
max sI         = 825
```

The following command tests the one-DSP-per-multiply design over that operating
range:

```bash
python3 rtl/verification/reciprocal_lut_sizing.py --seed-bits 16 --required-bits 22 --min-divisor 59
```

Result:

```text
Smallest passing table: 8 index bits, 256 entries
Guaranteed bits       : 22 fixed-point fractional bits
Max centroid error    : 9.41753387e-05 pixels
Max Zernike error     : 3.28302383e-04
```

That means the existing scale of table, 256 entries, is enough for the current
frame if the required reciprocal quality is 22 useful fractional bits.

## Direct Floating-Point Comparison

The proposed one-Newton DSP-shaped reciprocal was also compared directly
against the floating-point scientific reference:

```text
max centroid error    : 9.83418629e-05 pixels
rms centroid error    : 4.40686125e-05 pixels
max valid slope error : 9.83418629e-05 pixels
rms valid slope error : 5.30814286e-05 pixels
max Zernike error     : 4.65227845e-04
rms Zernike error     : 2.50051260e-04
```

Those errors are small relative to the `0.001 pixel` centroid target.

## What Happens If the Table Is Bigger

For the one-DSP-per-multiply design with an 18-bit correction, the sweep showed:

```text
256 entries  -> 22 guaranteed bits over divisors 59..65535
512 entries  -> 22 guaranteed bits over divisors 59..65535
1024 entries -> 22 guaranteed bits over divisors 59..65535
4096 entries -> 22 guaranteed bits over divisors 59..65535
```

So a larger table does not solve the precision limit in this architecture. The
correction width is the limiting factor.

If the correction is widened to 25 bits, the table starts to matter again:

```text
512 entries -> 24 guaranteed bits over divisors 59..65535
```

But that no longer matches the simple one-DSP-per-multiply `25 x 18` plan for
the second multiply. A 25-bit correction would require a wider multiply, split
DSPs, or a different architecture.

## Conservative Boundary

The result above is based on the current frame's nonzero intensity range. If a
future frame has much dimmer valid subapertures, then the operating range
changes. In that case, rerun the sizing test with a smaller `--min-divisor`.

For worst-case all nonzero 16-bit divisors:

```bash
python3 rtl/verification/reciprocal_lut_sizing.py --seed-bits 16 --required-bits 22
```

the one-DSP-per-multiply, 18-bit-correction design does not pass 22 guaranteed
bits. That is because very small divisors are much harder to approximate after
denormalization and saturation.

So the honest statement is:

```text
For the current SHWFS frame and nonzero sI >= 59, 256 entries are enough.
For every possible nonzero 16-bit divisor, this one-Newton/18-bit-correction
design is not enough for 22 guaranteed bits.
```

## Recommendation

Use this as the next hardware experiment:

```text
LUT index width : 8 bits
LUT entries     : 256
seed width      : current Q0.16 is acceptable for the tested range
Newton steps    : 1
DSPs            : 2, one per multiply
output bus      : Q0.25
valid handling  : pipeline valid/count/xI/yI alongside reciprocal
```

Do not make the LUT bigger first. If later tests show more precision is needed,
the better next changes are:

```text
widen the correction path
add a second Newton iteration
or use more DSPs for a wider multiply
```

Those changes attack the real precision limiter. A larger table alone does not.
