# Who is left out of digital finance? Predicting account type in Sub-Saharan Africa

Machine Learning summative project by Aime Igirimpuhwe.

Using the World Bank Global Findex 2021 survey, this project predicts whether an adult in Sub-Saharan Africa has **no account, a mobile money account only, a financial institution account only, or both**, from information a survey or outreach team could realistically collect (demographics, phone and internet access, how the person earns, saves and borrows). It compares traditional machine learning (scikit-learn) with deep learning (TensorFlow/Keras, Functional API + `tf.data`).

- **Google Drive** (raw data zip + saved models): https://drive.google.com/drive/folders/1HyPiJfCO2qJNIgWBvC7ivd9KWkM0AHnc?usp=sharing
- **Video presentation:** https://drive.google.com/file/d/1xPVGYgFGBHXzTKNNS_p_HtIvxA7QytqY/view?usp=sharing

## Results at a glance

Test set (5,410 adults, used once). Macro F1 is the main metric because the classes are imbalanced.

| Model | Macro F1 (95% CI) | Balanced acc. | Accuracy | Macro ROC-AUC |
|---|---|---|---|---|
| Best ML: gradient boosting on 24 selected inputs (`ML-10`) | 0.597 (0.583 to 0.610) | 0.609 | 0.630 | 0.848 |
| Best DL: one-hidden-layer MLP (`DL-01`) | 0.596 (0.582 to 0.609) | 0.589 | 0.656 | 0.850 |
| Chance (stratified guess, `ML-01`, validation) | 0.248 | 0.248 | 0.304 | 0.499 |

- ML and DL are statistically tied. Simpler models are as good as deeper ones on this tabular survey data.
- Training on all 128 columns, leaks included (`ML-12`), gives 0.9995 macro F1. That shows why the leakage audit is necessary.
- Holding out whole countries drops macro F1 to 0.482 ± 0.029, so country context matters a lot.
- Full experiment table: `results/experiments.csv` (19 runs, IDs match `logs/fit/`).

## Data

- Source: World Bank Microdata Library, *Global Financial Inclusion (Global Findex) Database 2021*, ref. WLD_2021_FINDEX_v03_M, DOI [10.48529/jq97-aj70](https://doi.org/10.48529/jq97-aj70).
- Raw file: 143,887 adults, 139 economies, 128 variables. The project uses the 36,062 adults from 36 Sub-Saharan African economies.
- The raw data is **not** in this repository. Download `WLD_2021_FINDEX_v03_M_csv.zip` from the World Bank (free account) or from the Google Drive folder above, and put the zip in `data/`. The notebook reads it directly.

## Repository layout

| Path | What it is |
|---|---|
| `Findex_Account_Type_Classification.ipynb` | The full pipeline, executed, with outputs |
| `requirements.txt` | Exact package versions used |
| `data/` | Create it and put the raw zip here (not committed) |
| `results/experiments.csv` | One row per experiment: model, settings, train/validation metrics |
| `results/final_test_metrics.csv` | Test results of the best ML and best DL model |
| `results/leave_country_out.csv` | Grouped cross-validation holding out whole countries |
| `results/accuracy_by_economy.csv` | Per-country accuracy vs a per-country majority guess |
| `figures/` | All plots used in the report |
| `logs/fit/` | TensorBoard logs for every experiment (same IDs as the experiment table) |
| `models/` | Created when the notebook saves the best models (shared via Google Drive, not committed) |

## How to run

Locally (tested with Python 3.13):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# put WLD_2021_FINDEX_v03_M_csv.zip in data/
jupyter notebook Findex_Account_Type_Classification.ipynb
```

On Google Colab: upload the notebook, create `MyDrive/Findex_Summative/data/` in your Drive, put the zip there, and run all cells. The notebook mounts Drive automatically. Every package it uses is preinstalled on Colab.

The whole notebook runs top to bottom on a laptop CPU in roughly 10 minutes. All randomness is fixed with seed 42.

## Viewing the experiment logs

```bash
tensorboard --logdir logs/fit
```

The **HParams** tab lists every run with its settings and validation metrics. The **Scalars** tab shows per-epoch training curves for the neural networks.

## Pipeline summary

1. **Target:** 4 classes from `account_fin` and `account_mob`.
2. **Leakage audit:** removed every variable that is only asked to account owners, only asked to non-owners, or whose answer codes mention an account (e.g. `receive_wages = 1` means "into an account"). Then dropped any remaining feature whose missing rate differs by more than 10 points across classes (`fin22c`, `fin24a`).
3. **Feature engineering** inside a scikit-learn pipeline: digital access, income flows, informal finance, worry index, age groups, country-level phone/internet rates (fitted on training rows only), log adult population.
4. **Splits:** stratified 70/15/15, seed 42. Validation for model selection, test used once.
5. **Experiments:** 11 scikit-learn runs (dummy, logistic regression, random forest, gradient boosting, SVM, feature selection, ablation) and 7 Keras runs (depth, batch norm, dropout, class weights, country embedding, learning rate schedule, L2, learning-rate stress test).
6. **Evaluation:** macro F1 (main), balanced accuracy, accuracy, macro ROC-AUC, bootstrap confidence intervals, confusion matrices, ROC and precision-recall curves, learning and validation curves, subgroup and per-country error analysis, leave-countries-out test.

## Data credit

Findex data © World Bank, used under its terms of use and cited above.
