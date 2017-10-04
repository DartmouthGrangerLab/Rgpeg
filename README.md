# Rgpeg

Code related to the paper:

[https://arxiv.org/abs/1708.00138]

This is pure python code, tested in version 2.7, but much of it may work in 3.* as well. There are two ways to use this repository.

## 1) Automatically generate figures related to the paper

### Requirements:

numpy

scipy

matplotlib

PIL

### Instructions:

From the directory containing roc.py, execute roc.py directly, without options:

```
cd <directory containing roc.py>
python roc.py
```

You should see a new folder, named 'results', appear. It should contain .pdfs of figures. This code has been optimized for readability not performance, so it may take several minutes to render all figures.

## 2) Generate encoded and decoded images for your own analyses

### Requirements:

numpy

### Instructions:

Call *jpeg_encode_decode()* or *rgpeg_encode_decode()* directly in your own code.

The only files needed in this case are **rgpeg.py**, **jpeg.py**, and **library.py**