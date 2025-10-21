stitch2D-CLI
========

Python Library to stitch microscopic images.


Install
-------

Install with pip:

```bash
pip install "git+https://github.com/JiwanChung/stitch2d_cli.git"
```

Quick start
-----------

1. Run on two image files:

```bash
stitch2d-cli ./examples/inputs2/0.png ./examples/inputs2/1.png -o ./examples/output2.png
```

2. Batched run on image directories:

```
stitch2d-cli batch ./examples
```

Given a folder structure like:

```
examples/
├── sample1/
│   ├── img1.png
│   ├── img2.png
├── sample2/
│   ├── view_a.jpg
│   ├── view_b.jpg
└── outputs/
```

the command will:

- Iterate through each subdirectory (sample1, sample2, …).
- Detect .png or .jpg image files inside each folder.
- Fuse the images using iter_files().
- Save each result as:

```
examples/outputs/sample1.png
examples/outputs/sample2.png
```
