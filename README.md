Porto Segro
===
- [ポルトテグロコンペ](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) 

# Structures
```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models
│   └── lgbm.py
├── notebooks
│   └── eda.ipynb
├── scripts
│   └── convert_to_feather.py
├── utils
│   └── __init__.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py
└── tox.ini
```
# Commands

## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## Run LightGBM

```
python run.py
```

## flake8

```
flake8 .
```
