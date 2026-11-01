# Simple Project Explanation

## 1. What does the project do?

This project checks whether information about a student's lifestyle and study habits can help us estimate school absences, final marks, and pass/fail results.

It has three prediction tasks and a Streamlit dashboard with a light/dark mode switch in the left sidebar. The switch updates the page, charts, cards, and tables together.

It has three prediction tasks:

- **Absences:** predict a number, so we use count regression.
- **Final grade:** predict a mark from 0 to 20, so we use Linear Regression.
- **Pass or Fail:** predict one of two classes, so we use classification.

## 2. Why did we choose this dataset?

The UCI Student Performance dataset already contains study time, past failures, parents' education, travel time, family relationships, free time, going out, alcohol-use ratings, health, absences, and final marks. This lets us demonstrate all the required methods using one dataset.

We use the Mathematics file, which contains 395 student records.

## 3. Which models are used?

| Model | Simple explanation | Used for |
|---|---|---|
| Poisson Regression | Predicts count values such as 0, 1, 2, or more absences | Absences |
| Negative Binomial Regression | Also predicts counts; it can handle overdispersed count data | Absences |
| Linear Regression | Predicts a continuous number | Final grade |
| LDA | Separates two classes using a shared covariance assumption | Pass/Fail |
| QDA | Separates two classes while allowing different covariance patterns | Pass/Fail |

## 4. What is overdispersion?

For a Poisson model, the mean and variance of the count are expected to be similar. If the variance is much higher than the mean, the count data is overdispersed. Negative Binomial Regression is useful for this situation because it allows the variance to grow more freely than the mean.

The dashboard shows:

- Mean and variance of absences.
- Poisson Pearson dispersion ratio.
- AIC for both count models.
- Test RMSE for both count models.

A lower AIC and lower test RMSE are generally preferred, but the dispersion check should also be discussed.

## 5. What is the pass/fail rule?

The final grade is called `G3` and ranges from 0 to 20.

- `G3 >= 10`: Pass (class 1)
- `G3 < 10`: Fail (class 0)

LDA and QDA are compared using accuracy, ROC-AUC, and confusion matrices.

## 6. Why are G1 and G2 excluded?

G1 and G2 are the student's first- and second-period marks. They are not included as inputs. The project is meant to test what lifestyle and study habits can tell us before using earlier marks. Including G1 and G2 would make final-grade prediction easier, but it would not match the main problem statement as well.

## 7. What does the train/test split mean?

The data is divided into two parts:

- 80% is used to train the models.
- 20% is kept aside to test the models on data they did not train on.

The test results are more useful than reporting only training performance because they give an estimate of how the model handles unseen records.

## 8. A simple presentation script

"Our project is called Predicting Student Absences and Pass/Fail Outcome from Lifestyle and Study Habits. We used the UCI Student Performance Mathematics dataset. We first explored the data and checked the distribution of absences and final grades. We used Poisson and Negative Binomial Regression to predict absences and compare their fit. We used Linear Regression to predict the final grade. We then created a Pass/Fail column using a cutoff of 10 marks and compared LDA with QDA. We used an 80/20 train-test split and evaluated the models using AIC, RMSE, R-squared, accuracy, ROC-AUC, and confusion matrices. We excluded G1 and G2 to avoid using earlier marks as shortcuts when predicting the final grade."

## 9. Important limitation

The dataset is small and comes from two Portuguese schools. The results may not apply to other schools or countries. The models show patterns in the data; they do not prove that a habit causes a particular outcome. This project is for academic learning, not for making decisions about actual students.
