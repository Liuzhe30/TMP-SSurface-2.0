# TMP-SSurface-2.0

A Deep Learning-Based Predictor for Surface Accessibility of Transmembrane Protein Residues.
<p align="center"><img width="50%" src="images/CNN-LSTM.png" /></p>
<p align="center"><img width="100%" src="images/TMP-SSurface-2.0.png" /></p>

## Download data

We provide the test dataset used in this study,  you can download test.fasta to evaluate our method.

## Quick Start

### Requirements
- Python ≥ 3.6
- Tensorflow and Keras
- Psi-Blast for generating PSSM files

### Testing & Evaluation in Command Line
We provide run.py that is able to run pre-trained models. Run it with:
```
python run.py -f sample/sample.fasta -p sample/pssm/ -o results/
```

* To set the path of fasta file, use `--fasta` or `-f`.
* To set the path of generated PSSM files, use `--pssm_path` or `-p`.
* To save outputs to a directory, use `--output` or `-o`.

## Progress
- [x] README for running TMP-SSurface-2.0.

## Citation
Please cite the following paper for using this code: 
```
Liu Z, Gong Y, Guo Y, Zhang X, Lu C, Zhang L, Wang H. TMP- SSurface2: A Novel Deep Learning-Based Surface Accessibility Predictor for Transmembrane Protein Sequence. Front Genet. 2021 Mar 15;12:656140. doi: 10.3389/fgene.2021.656140. PMID: 33790952; PMCID: PMC8006303.
```