# Student Habits & Performance Prediction

A beginner-friendly Python machine learning project. It studies whether lifestyle and study habits can help predict:

1. **Number of school absences** using Poisson Regression and Negative Binomial Regression.
2. **Final Mathematics grade (G3)** using Linear Regression.
3. **Pass or Fail** using LDA and QDA. Pass means `G3 >= 10`.

The project has a Streamlit dashboard with simple charts, model comparisons, and evaluation results. The dashboard includes a compact sun/moon button in the top-right corner; the page, charts, metrics, and tables change together.

## 1. Dataset

This project uses the **Mathematics file (`student-mat.csv`)** from the UCI Student Performance dataset. The dataset contains information about students from two Portuguese schools, including study time, past failures, family education, travel time, free time, going out, alcohol use, health, absences, and grades.

The app first checks for `data/student-mat.csv`. If it is not present, it downloads the CSV from a public copy of the dataset. An internet connection is needed for the first run unless you place the file in the `data` folder yourself.

Official dataset page: <https://archive.ics.uci.edu/dataset/320/student+performance>

Dataset citation:

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

The UCI listing states that the dataset is licensed under CC BY 4.0. Please retain the attribution when redistributing it.

## 2. Models used

| Task | Target | Models | Evaluation |
|---|---|---|---|
| Absence prediction | `absences` | Poisson, Negative Binomial | Mean, variance, dispersion ratio, AIC, test RMSE |
| Grade prediction | `G3` | Linear Regression | Test R², test RMSE, residual plot |
| Pass/fail prediction | `G3 >= 10` | LDA, QDA | Accuracy, ROC-AUC, confusion matrix |

### Why two count models?

Poisson Regression is a common model for count data. It assumes the count variance is tied to the mean. When the data is overdispersed (variance is greater than the mean), Negative Binomial Regression can be a better fit. The app displays both the basic mean-versus-variance check and the Poisson model's Pearson dispersion ratio.

### Why are G1 and G2 not used?

`G1` and `G2` are earlier-period grades. They are excluded from the predictors so the grade and pass/fail models use the lifestyle, study, family, and attendance information rather than earlier marks. The UCI dataset description notes that G1 and G2 are strongly correlated with G3. Excluding them makes this a more meaningful early-prediction task.

## 3. Features used

The project uses these numeric features:

- `age`: student's age
- `Medu`, `Fedu`: mother's and father's education levels
- `traveltime`: travel time to school
- `studytime`: weekly study time category
- `failures`: past class failures
- `famrel`: family relationship rating
- `freetime`: free time after school
- `goout`: going out with friends
- `Dalc`, `Walc`: workday and weekend alcohol-use ratings
- `health`: current health rating
- `absences`: school absences (used only when predicting final grade/pass-fail)

For absence prediction, `absences` is the target and is not included as an input feature.

## 4. Run the project on Windows

Install Python 3.10 or newer, then follow these steps in a terminal opened inside this project folder.

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open in your browser. If it does not open automatically, use the local address shown in the terminal (normally `http://localhost:8501`).

## 5. Run the project on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## 6. If the dataset does not download

1. Open the [UCI Student Performance dataset page](https://archive.ics.uci.edu/dataset/320/student+performance).
2. Download the dataset ZIP file.
3. Extract `student-mat.csv` from the ZIP.
4. Place it at `data/student-mat.csv` inside this project folder.
5. Run `streamlit run app.py` again.

## 7. Dashboard appearance

Use the circular **moon/sun button** in the top-right corner to switch between light and dark themes. The moon switches to dark mode; the sun switches back to light mode. The selected theme is applied to the page background, text, metric cards, charts, and result tables. The default theme is light.

## 8. Project workflow

1. Load the Mathematics dataset.
2. Check the columns and remove rows with missing values in the fields used by the models.
3. Explore absence and grade distributions.
4. Split data into training (80%) and testing (20%) sets.
5. Train Poisson and Negative Binomial models for absences.
6. Train Linear Regression for the final grade.
7. Create a Pass/Fail label and train LDA and QDA.
8. Compare the models using the test metrics shown in the dashboard.

## 9. Important limitations

- This is a relatively small dataset from two Portuguese schools, not a representative sample of every school or country.
- A single train/test split can give slightly different results from another split.
- The project uses a compact set of numeric features to keep the code and explanation easy to follow.
- The models show associations, not causes.
- The dashboard is for learning and demonstration, not for making decisions about real students.

## 10. Folder structure

```text
student_absence_project/
│
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── README.md
└── data/
    └── README.txt
```
