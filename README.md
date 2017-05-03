# Binning in Scikit Learn

_The project does NOT contain any of the data required to run the code
in the project. See below for websites where the datasets can be obtained._

## Requirements

The code in this repo runs on Python 2.7.x and requires the following
python packages:

```
numpy >= 1.9.2
scikit-learn >= 0.18.0
pandas >= 0.19.1
matplotlib >= 1.5.3
```

Run `pip install -r requirements.txt` in project directory to install
needed packages.


## Running the code

To run the binning method comparison pipeline run the following in the
project directory: `python run_me.py`

The conversion model code can be run using the following command from the
project directory: `python conversion.py`

_The conversion model dataset is not distributed with this project, as it is
owned by MassMutual._

## Datasets

All datasets must be placed in the `datasets` folder, and named as listed below.

- glass.txt - [UIC link](https://archive.ics.uci.edu/ml/datasets/Glass+Identification)
- heart.txt - [UIC link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- iris.txt - [UIC link](https://archive.ics.uci.edu/ml/datasets/Iris)
- wine.txt - [UIC link](https://archive.ics.uci.edu/ml/datasets/Wine)
- breast_cancer_wisc.txt - [UIC link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## Tests

To run the unit tests of the scikit-learn classes written for this project,
you will need to run the test script as a package using *python 3*.

This would look something like this from _the folder *above* project directory_:
```
python3 -m tree-binning-sklearn.tests.test_binning
```

If nothing is printed, the all of the tests passed.
