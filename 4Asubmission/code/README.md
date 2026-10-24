# CS180 Project 4A — Image Warping and Mosaicing

This workspace contains a from-scratch Python implementation of the required
homography, inverse warping, nearest-neighbor interpolation, bilinear
interpolation, and feathered mosaic blending. It deliberately does **not** use
forbidden high-level homography or image-warping functions.

## 1. Add your photos

Place your images using this structure:

```text
images/
  mosaic_1/left1.jpeg right1.jpeg
  mosaic_2/left2.jpeg middle2.jpeg right2.jpeg
  mosaic_3/left3.jpeg middle3.jpeg right3.jpeg
  rectification/rectification1.jpeg rectification2.jpeg
```

The first image passed to `mosaic` is the reference frame. Use at least 4
correspondences; 8–12 well-spread matches normally works better.

## 2. Create an environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Prepare working copies

The phone photos are 4032×3024, so create smaller copies before warping. This
does not alter the originals in `images/`.

```bash
python prepare_images.py --max-side 1600
```

The subsequent commands use `working_images/`. It contains the same folder and
file names as `images/`.

## 4. Run rectification (twice)

```bash
python run.py rectify working_images/rectification/rectification1.jpeg \
  --output outputs/rect_1_bilinear.png --method bilinear

python run.py rectify working_images/rectification/rectification1.jpeg \
  --output outputs/rect_1_nearest.png --method nearest
```

For each command, click the four corners in this exact order: **top-left,
top-right, bottom-right, bottom-left**. Repeat for `rect_2.jpg`. The paired
nearest/bilinear outputs are required for the interpolation comparison.

## 5. Build a mosaic

```bash
python run.py mosaic \
  --images working_images/mosaic_1/left1.jpeg working_images/mosaic_1/right1.jpeg \
  --output outputs/mosaic_1.png \
  --save-data correspondences/mosaic_1.json \
  --count 10 --method bilinear
```

For each non-reference image, two windows appear:

1. Click the selected features in the non-reference image.
2. Click the **same** features in the reference image in the identical order.

Choose features spread across the overlap (corners of signs, shelf corners,
window corners), rather than a cluster of points in one area. The command saves
the final mosaic, a JSON record of its point matches/homographies, and numbered
correspondence visualizations beside the mosaic. Repeat for all three mosaic
sets.

## Useful standalone point picker

Use this when you want an explicit correspondence visualization for the report:

```bash
python run.py pick working_images/mosaic_1/right1.jpeg working_images/mosaic_1/left1.jpeg \
  --output correspondences/left_to_middle.json --count 10
```

This saves the JSON point coordinates and a labeled `.png` visualization.

## Deliverable checklist

- Two or more photographed overlapping image sets.
- Correspondence visualizations, the `Ah=b` derivation, and recovered `H`
  matrices (the CLI prints each matrix).
- At least two rectifications, each shown with nearest-neighbor and bilinear
  output plus a short quality/speed comparison.
- Three mosaics with their source photos and a description of feathered weighted
  blending.
