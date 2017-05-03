# Binning in Scikit Learn

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

_The conversion model dataset is not distributed with this project._


## test_size

To run the unit tests of the scikit-learn classes written for this project,
you will need to run the test script as a package using *python 3*.

This would look something like this from _the folder *above* project directory_:
```
python3 -m tree-binning-sklearn.tests.test_binning
```

If nothing is printed, the all of the tests passed.
