# Netflix Data Science Project

An end-to-end **Data Science project independently developed** using a Netflix titles dataset. The project covers data cleaning and preprocessing, exploratory data analysis, machine learning classification, content-based recommendation, and trend forecasting.

The project follows a structured Data Science workflow, transforming raw Netflix catalogue data into a cleaned analytical dataset and applying statistical analysis, machine learning, recommendation techniques, and forecasting methods.

---

## Project Overview

This project explores Netflix catalogue data from multiple Data Science perspectives:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Machine Learning Classification
* Content-Based Recommendation System
* Trend Analysis & Forecasting
* Data Visualization
* Model Evaluation
* Output Generation for Further Analysis and Dashboards

The project is organized as a multi-stage pipeline where the cleaned dataset produced during preprocessing is used by downstream analytical and machine learning tasks.

---

## Project Pipeline

```text
Raw Netflix Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Cleaned Dataset
        │
        ├──────────────► Exploratory Data Analysis
        │
        ├──────────────► Classification Models
        │
        ├──────────────► Recommendation System
        │
        └──────────────► Trend Forecasting
                              │
                              ▼
                       Forecast Results
```

---

## Key Components

### 1. Data Cleaning & Preprocessing

The preprocessing pipeline prepares the raw Netflix dataset for downstream analysis.

Key operations include:

* Loading the raw dataset
* Detecting missing values
* Detecting placeholder values
* Identifying duplicate records
* Removing duplicate content records
* Cleaning whitespace and text values
* Standardizing categorical labels
* Handling missing directors and countries
* Converting date fields
* Extracting year, month, and weekday information
* Transforming duration into numerical features
* Separating movie duration and TV show seasons
* Splitting multi-valued genres
* Creating a primary genre feature
* Creating audience groups from ratings
* Creating analytical features such as:

  * `years_to_netflix`
  * `num_genres`
  * `num_directors`
  * `duration_minutes`
  * `seasons`

The resulting cleaned dataset is saved as:

```text
data/processed/netflix_cleaned.csv
```

The preprocessing script also performs sanity checks before saving the final dataset.

---

### 2. Exploratory Data Analysis

The EDA stage analyzes the cleaned Netflix catalogue and generates visualizations.

Areas explored include:

* Dataset size and structure
* Movies vs TV Shows
* Titles added over time
* Top content-producing countries
* Popular genres
* Content ratings
* Audience groups
* Movie runtime
* TV show seasons
* Release-year trends
* Release-to-Netflix arrival gap
* Netflix content addition timing

The EDA script automatically saves visualizations into the `figures/` directory.

---

### 3. Machine Learning Classification

The classification component applies supervised machine learning to Netflix content.

The script supports classification of:

* Audience group
* Content type

The audience classification task uses:

* Kids
* Teens
* Adults

The model pipeline includes feature engineering from:

* Release dates
* Countries
* Director information
* Title words
* Genres
* Content type
* Duration
* Ratings

Several machine learning algorithms are compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

The models are evaluated using cross-validation and a held-out test set.

Evaluation metrics include:

* Accuracy
* Balanced Accuracy
* Macro F1
* Weighted F1
* ROC-AUC
* Classification Report
* Confusion Matrix

Hyperparameter tuning is performed using `GridSearchCV`, and the final model is selected using cross-validation macro-F1.

The best trained model is also saved as a `.joblib` file for later use.

---

### 4. Content-Based Recommendation System

The recommendation component generates Netflix title recommendations based on content similarity.

The system uses features including:

* Genres
* Director
* Rating
* Audience group
* Content type
* Release era
* Country
* Title words

Text preprocessing includes normalization, stop-word removal, tokenization, and conversion of multi-word categories into usable tokens.

The recommendation engine uses:

* TF-IDF vectorization
* Weighted feature blocks
* Cosine similarity
* Similarity-based ranking

Recommendations are generated for selected Netflix titles, with the system returning the top similar titles.

The implementation also evaluates recommendation quality using metrics such as:

* Hit Rate@10
* Recall@10
* Genre similarity
* Genre diversity
* Catalogue coverage
* Recommendation concentration

The system is explicitly **content-based** and does not rely on user viewing history or user ratings because those fields are not available in the dataset.

---

### 5. Trend Analysis & Forecasting

The trend forecasting component analyzes Netflix content release patterns over time.

The analysis includes:

* Yearly content trends
* Movie vs TV Show trends
* TV Show share of releases
* Year-over-year growth
* Release-year analysis
* Data maturity considerations
* Forecasting of future content counts

Multiple forecasting approaches are compared:

* Naive forecast
* Linear trend
* Recent linear trend
* Quadratic trend
* Log-linear / exponential trend
* Holt damped trend

The models are evaluated using rolling-origin backtesting and metrics including:

* MAPE
* MAE
* RMSE

The best-performing model is selected based on backtesting performance and used to generate future forecasts.

---

## Project Structure

```text
netflix-data-science-project/
│
├── data/
│   ├── raw/
│   │   └── Dataset.csv
│   └── processed/
│       └── netflix_cleaned.csv
│
├── figures/
│   └── EDA visualizations
│
├── classification_output/
│   ├── model comparison results
│   ├── classification reports
│   ├── confusion matrices
│   ├── feature importance
│   └── trained models
│
├── recommender_output/
│   └── recommendation results and evaluation
│
├── trend_output/
│   └── forecasting results and visualizations
│
├── data_cleaning.py
├── netflix_eda.py
├── netflix_classification.py
├── netflix_recommender.py
├── netflix_trend_forecast.py
│
├── build_dashboard.py
├── dashboard_template.py
├── netflix_dashboard.html
│
├── requirements.txt
└── README.md
```

---

## Technologies & Libraries

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Core Techniques

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* TF-IDF
* Cosine Similarity
* Classification
* Hyperparameter Tuning
* Cross-Validation
* Model Evaluation
* Time-Series Forecasting
* Backtesting

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/netflix-data-science-project.git
```

### 2. Navigate into the project

```bash
cd netflix-data-science-project
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Data Cleaning

```bash
python data_cleaning.py
```

This creates:

```text
data/processed/netflix_cleaned.csv
```

### Step 2 — Exploratory Data Analysis

```bash
python netflix_eda.py
```

EDA charts are saved to:

```text
figures/
```

### Step 3 — Classification

For audience classification:

```bash
python netflix_classification.py
```

For content-type classification:

```bash
python netflix_classification.py data/processed/netflix_cleaned.csv classification_output type
```

Classification results are saved to:

```text
classification_output/
```

### Step 4 — Recommendation System

```bash
python netflix_recommender.py
```

Recommendations and evaluation results are saved to:

```text
recommender_output/
```

### Step 5 — Trend Forecasting

```bash
python netflix_trend_forecast.py
```

Forecasting results are saved to:

```text
trend_output/
```

---

## Outputs

The project generates a variety of analytical outputs, including:

* Cleaned datasets
* EDA visualizations
* Model comparison tables
* Classification reports
* Confusion matrices
* Feature importance analysis
* Trained machine learning models
* Recommendation results
* Recommendation evaluation metrics
* Historical trend analysis
* Forecast results
* Forecast visualizations

---

## Methodology Highlights

This project emphasizes a structured and reproducible Data Science workflow.

### Data Quality

Duplicate detection, missing-value handling, placeholder detection, data type conversion, and consistency checks are performed before modeling.

### Feature Engineering

The project derives additional analytical features from dates, duration, genres, ratings, countries, directors, and titles.

### Model Validation

The classification workflow uses stratified train/test splitting, cross-validation, hyperparameter tuning, and held-out test evaluation.

### Recommendation Evaluation

The recommendation system goes beyond generating recommendations by evaluating hit rate, recall, diversity, similarity, and catalogue coverage.

### Forecast Validation

Forecasting models are compared using rolling-origin backtesting instead of relying only on a single fitted model.

---

## Important Dataset Considerations

The recommendation system is based on available catalogue metadata. The dataset does not contain:

* User viewing history
* User ratings
* Individual user preferences
* Plot descriptions
* Complete cast information

Therefore, the recommendation engine is a **content-based recommendation system**, rather than a collaborative filtering system.

The trend forecasting component also accounts for incomplete recent release-year data, since newer titles may continue to be added to Netflix after their original release year.

---

## Future Improvements

Potential future improvements include:

* Deploying the dashboard as an interactive web application
* Adding user-specific recommendation profiles
* Incorporating plot descriptions and cast information
* Adding collaborative filtering when user interaction data becomes available
* Experimenting with advanced NLP embeddings
* Adding automated model retraining
* Building an API for recommendations
* Adding interactive forecasting controls
* Containerizing the application with Docker
* Deploying the complete system to a cloud platform

---

## Project Status

**Completed**

The project currently includes:

* Data preprocessing
* Exploratory analysis
* Machine learning classification
* Content-based recommendation
* Trend forecasting
* Model evaluation
* Output generation
* Dashboard components

---

## Author

**Warda Adan**

Data Science | AI/ML

GitHub: `https://github.com/wardaadan03-netizen`

LinkedIn: `https://www.linkedin.com/in/thewardaadan-wa`

Kaggle: `https://www.kaggle.com/wardaadann`

---

## Disclaimer

This is an **independent educational Data Science project** developed for learning, experimentation, portfolio development, and analytical purposes.

This project is not affiliated with or endorsed by Netflix.

