# Structure and Purpose of all files

## prepare_im2latex.py
- **Generates Math Latex Data**
- Downloads and extracts the Im2LaTeX dataset.
- Resizes images to 64×256 grayscale format.
- Maps each image to its LaTeX formula.
- Saves processed images in `data/math/images`.
- Stores labels in `data/math/labels.csv`.

## prepare_iiit-5k.py
- **Generates English Words Data**
- Downloads and extracts the IIIT-5K dataset.
- Resizes images to 64×256 grayscale format.
- Maps each image to its text.
- Saves processed images in `data/english/images`.
- Stores labels in `data/math/labels.csv`.