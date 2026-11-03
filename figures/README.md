# Figures & Visual Assets

This directory contains architectural diagrams, animated teasers, visual comparisons, and diagnostic examples of super-resolution model behavior.

## Key Assets

| File / Folder | Description |
| :--- | :--- |
| `kenya_sentinel2.gif` | Animated multi-frame low-resolution Sentinel-2 input over Kenya (10m ground resolution). |
| `kenya_superres.gif` | Animated $4\times$ super-resolved prediction showing sharp road networks and field boundaries (2.5m resolution). |
| `esrgan_generator.svg` | Vector diagram of the 23-block Residual-in-Residual Dense Network (`SSR_RRDBNet`) generator trunk. |
| `image_pair.svg` | Schematic illustrating the temporal pairing of Sentinel-2 multi-frame revisits with aerial NAIP ground truth. |
| `teaser.svg` | High-level project teaser comparing input, super-resolved output, and aerial reference. |
| `model_architecture_diagram.jpg` | Complete neural network architecture schematic including fusion, RRDB trunk, and upsamplers. |
| `uncertainty_pipeline_diagram.jpg` | Diagram of the test-time augmentation (TTA) and uncertainty calibration workflow. |
| `car_hallucination.jpg` | Diagnostic example showing vehicular phantom artifacts generated along transit corridors. |
| `vessel_hallucination.jpg` | Diagnostic example showing maritime phantom artifacts generated in open water bodies. |
| `kota_india.svg` | Qualitative comparison examining geographic generalization on non-US arid urban scenes. |
| `stitch_example.svg` | Illustration of seamless $16 \times 16$ tile mosaic stitching. |
| `comparison_sets/` | Per-tile crops for side-by-side qualitative comparisons across experiments. |

