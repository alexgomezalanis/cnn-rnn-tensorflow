# CNN + RNN Tensorflow

A [tensorflow](https://www.tensorflow.org) implementation of the CNN + RNN architecture proposed in [A Deep Identity Representation for Noise Robust Spoofing Detection](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1909.pdf) by Alejandro Gomez-Alanis, Antonio M. Peinado, Jose A. Gonzalez and Angel M. Garcia.

## Installation

* Install [tensorflow](https://www.tensorflow.org) following the website.

* Clone this repository.

* Optional: Install [HTK toolkit](http://htk.eng.cam.ac.uk) to extract log Mel filter bank features.

## Datasets

* Download [ASVspoof 2015](https://datashare.is.ed.ac.uk/handle/10283/853) dataset.

* Spectral features are obtained using [HTK toolkit](http://htk.eng.cam.ac.uk).

## Citation

If you use our model, please cite the following paper:

```
@article{gomez2018,
  title={A Deep Identity Representation for Noise Robust Spoofing Detection},
  author={Alejandro Gomez-Alanis, Antonio M. Peinado, Jose A. Gonzalez, and Angel M. Gomez},
  journal={Proceedings Interspeech},
  pages={676--680},
  year={2018},
  publisher={ISCA}
}
```

