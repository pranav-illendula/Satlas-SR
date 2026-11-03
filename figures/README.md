# Figures & Visual Assets

This directory contains core architectural diagrams, animated teasers, visual demonstrations, and diagnostic examples of super-resolution model behavior from the published Satlas-SR benchmark.

## Key Assets

| File | Description |
| :--- | :--- |
| `kenya_sentinel2.gif` | Animated multi-frame low-resolution Sentinel-2 input over Nakuru, Kenya (10m ground resolution). |
| `kenya_superres.gif` | Animated $4\times$ super-resolved prediction showing sharp road networks and field boundaries (2.5m resolution). |
| `esrgan_generator.svg` | Vector diagram of the 23-block Residual-in-Residual Dense Network (`SSR_RRDBNet`) generator trunk. |
| `image_pair.svg` | Schematic illustrating the temporal pairing of Sentinel-2 multi-frame revisits with aerial NAIP ground truth. |
| `teaser.svg` | High-level project teaser comparing input, super-resolved output, and aerial reference. |
| `car_hallucination.jpg` | Diagnostic example showing vehicular phantom artifacts generated along transit corridors. |
| `vessel_hallucination.jpg` | Diagnostic example showing maritime phantom artifacts generated in open water bodies. |
| `kota_india.svg` | Qualitative comparison examining geographic generalization on non-US arid urban scenes (Kota, India). |
| `stitch_example.svg` | Illustration of seamless $16 \times 16$ tile mosaic stitching. |
