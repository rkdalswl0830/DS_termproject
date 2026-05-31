"""
auto_ml_pipeline
================

Automated machine learning pipeline for the Mental Health in Tech Survey
classification task. Provides an end-to-end workflow from raw CSV loading
through preprocessing, hyperparameter tuning, k-fold cross validation,
holdout evaluation, and permutation-based feature importance analysis.

The module exposes a single high-level entry point, :func:`run_training_pipeline`,
which compares multiple classifiers under a unified data split and scoring
scheme and returns a dictionary of result tables and fitted models.

Authors
-------
DS Term Project — Team 10

Example
-------
>>> from auto_ml_pipeline import run_training_pipeline, plot_model_performance
>>> artifacts = run_training_pipeline(
...     csv_path="data/survey.csv",
...     target_col="treatment",
...     k_values=(5,),
...     primary_metric="f1",
... )
>>> plot_model_performance(artifacts)
>>> top5 = artifacts["holdout_df"].head(5)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Preprocessing helpers
# =============================================================================

def clean_gender(value):
    """Normalize free-form gender values into ``Male``, ``Female`` or ``Other``.

    Parameters
    ----------
    value : object
        Raw gender response from the survey. Coerced to ``str`` internally.

    Returns
    -------
    str
        One of ``"Male"``, ``"Female"`` or ``"Other"``.
    """
    gender = str(value).strip().lower()

    male_values = [
        "male", "m", "male-ish", "maile", "mal", "male (cis)",
        "make", "male ", "man", "msle", "mail", "cis male",
        "cis man", "ostensibly male, unsure what that really means",
    ]
    female_values = [
        "female", "f", "woman", "female ", "femake",
        "female (cis)", "cis female", "femail",
    ]

    if gender in male_values:
        return "Male"
    if gender in female_values:
        return "Female"
    return "Other"


def drop_unnecessary_columns(df_clean):
    """Drop columns that carry no predictive signal or have excessive missingness.

    Removes ``Timestamp`` (no predictive value), ``comments`` (~87% missing),
    and ``state`` (~41% missing, only meaningful for US respondents).

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe with unnecessary columns removed.
    """
    drop_columns = ["Timestamp", "comments", "state"]
    existing_drop_columns = [col for col in drop_columns if col in df_clean.columns]
    return df_clean.drop(columns=existing_drop_columns)


def handle_dirty_age(df_clean):
    """Coerce ``Age`` to numeric and impute biologically implausible values.

    Values outside the working-age range ``[18, 80]`` are treated as outliers,
    replaced with ``NaN``, then imputed with the column median.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Input dataframe containing an ``Age`` column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the cleaned ``Age`` column.
    """
    if "Age" not in df_clean.columns:
        return df_clean

    df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce")
    invalid_age = (df_clean["Age"] < 18) | (df_clean["Age"] > 80)
    df_clean.loc[invalid_age, "Age"] = np.nan

    age_median = df_clean["Age"].median()
    df_clean["Age"] = df_clean["Age"].fillna(age_median)
    return df_clean


def handle_dirty_gender(df_clean):
    """Apply :func:`clean_gender` to the ``Gender`` column."""
    if "Gender" in df_clean.columns:
        df_clean["Gender"] = df_clean["Gender"].apply(clean_gender)
    return df_clean


def handle_missing_values(df_clean):
    """Impute missing values in low-missing categorical columns using the mode.

    Currently handles ``self_employed`` and ``work_interfere``.
    """
    if "self_employed" in df_clean.columns:
        mode_value = df_clean["self_employed"].mode()[0]
        df_clean["self_employed"] = df_clean["self_employed"].fillna(mode_value)

    if "work_interfere" in df_clean.columns:
        mode_value = df_clean["work_interfere"].mode()[0]
        df_clean["work_interfere"] = df_clean["work_interfere"].fillna(mode_value)

    return df_clean


def encode_target_variable(df_clean, target_col="treatment"):
    """Encode binary target ``Yes``/``No`` as integers ``1``/``0``.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Input dataframe.
    target_col : str, default="treatment"
        Name of the binary target column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the target encoded as integers.
    """
    df_clean[target_col] = df_clean[target_col].map({"Yes": 1, "No": 0})
    return df_clean


def convert_country_to_continent(df_clean):
    """Replace high-cardinality ``Country`` with a low-cardinality ``Continent`` column.

    Unmapped countries fall back to ``"Other"``. The original ``Country`` column
    is dropped after mapping.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Input dataframe containing a ``Country`` column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with ``Country`` replaced by ``Continent``.
    """
    if "Country" not in df_clean.columns:
        return df_clean

    continent_map = {
        "United States": "North America", "Canada": "North America",
        "Mexico": "North America", "Costa Rica": "North America",
        "Bahamas, The": "North America", "Brazil": "South America",
        "Colombia": "South America", "Uruguay": "South America",
        "Austria": "Europe", "Belgium": "Europe",
        "Bosnia and Herzegovina": "Europe", "Bulgaria": "Europe",
        "Croatia": "Europe", "Czech Republic": "Europe",
        "Denmark": "Europe", "Finland": "Europe",
        "France": "Europe", "Georgia": "Europe", "Germany": "Europe",
        "Greece": "Europe", "Hungary": "Europe", "Ireland": "Europe",
        "Italy": "Europe", "Latvia": "Europe", "Moldova": "Europe",
        "Netherlands": "Europe", "Norway": "Europe", "Poland": "Europe",
        "Portugal": "Europe", "Romania": "Europe", "Russia": "Europe",
        "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe",
        "Switzerland": "Europe", "United Kingdom": "Europe",
        "China": "Asia", "India": "Asia", "Israel": "Asia",
        "Japan": "Asia", "Philippines": "Asia", "Singapore": "Asia",
        "Thailand": "Asia", "Australia": "Oceania", "New Zealand": "Oceania",
        "Nigeria": "Africa", "South Africa": "Africa", "Zimbabwe": "Africa",
    }

    df_clean["Continent"] = df_clean["Country"].map(continent_map).fillna("Other")
    df_clean = df_clean.drop(columns=["Country"])
    return df_clean


def encode_categorical_with_dummies(df_clean):
    """One-hot encode all object-dtype columns and cast to ``int``."""
    categorical_columns = df_clean.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df_clean, columns=categorical_columns)
    return df_encoded.astype(int)


def feature_scaling_minmax(df_encoded, target_col="treatment"):
    """Apply Min-Max scaling to all feature columns (target excluded).

    Parameters
    ----------
    df_encoded : pandas.DataFrame
        Dataframe with all features already encoded as numeric.
    target_col : str, default="treatment"
        Target column to exclude from scaling.

    Returns
    -------
    pandas.DataFrame
        Dataframe with feature columns scaled to ``[0, 1]``.
    """
    df_scaled = df_encoded.copy()
    feature_columns = df_scaled.columns.drop(target_col)
    scaler = MinMaxScaler()
    df_scaled[feature_columns] = scaler.fit_transform(df_scaled[feature_columns])
    return df_scaled


def preprocess(df, target_col="treatment"):
    """Run the full preprocessing pipeline in order.

    Steps:
        1. Drop unnecessary columns.
        2. Clean ``Age``, ``Gender`` and impute missing values.
        3. Encode target and convert ``Country`` to ``Continent``.
        4. One-hot encode categorical features.
        5. Apply Min-Max scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.
    target_col : str, default="treatment"
        Name of the binary target column.

    Returns
    -------
    pandas.DataFrame
        Fully preprocessed dataframe ready for modeling.
    """
    df_clean = df.copy()
    df_clean = drop_unnecessary_columns(df_clean)
    df_clean = handle_dirty_age(df_clean)
    df_clean = handle_dirty_gender(df_clean)
    df_clean = handle_missing_values(df_clean)
    df_clean = encode_target_variable(df_clean, target_col=target_col)
    df_clean = convert_country_to_continent(df_clean)
    df_encoded = encode_categorical_with_dummies(df_clean)
    df_scaled = feature_scaling_minmax(df_encoded, target_col=target_col)
    return df_scaled


def load_and_clean_data(csv_path, target_col="treatment"):
    """Load the survey CSV, preprocess it, and split into ``X`` and ``y``.

    Parameters
    ----------
    csv_path : str
        Path to the ``survey.csv`` file (assumed ``latin1`` encoded).
    target_col : str, default="treatment"
        Name of the binary target column.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.
    y : pandas.Series
        Integer-encoded target of shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If ``target_col`` is not in the loaded dataframe.
    """
    df = pd.read_csv(csv_path, encoding="latin1")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df_scaled = preprocess(df, target_col=target_col)
    df_scaled = df_scaled.dropna(subset=[target_col]).copy()

    X = df_scaled.drop(columns=[target_col])
    y = df_scaled[target_col].astype(int)
    return X, y


# =============================================================================
# Model space
# =============================================================================

def _get_model_space(random_state=42):
    """Return the dict of classifiers and their GridSearchCV parameter grids.

    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility, applied to every stochastic model.

    Returns
    -------
    models : dict[str, sklearn.base.BaseEstimator]
        Mapping from model name to an unfit estimator instance.
    param_grids : dict[str, dict]
        Mapping from model name to the GridSearchCV parameter grid. Parameter
        keys are prefixed with ``model__`` so they target the ``model`` step of
        the :class:`~sklearn.pipeline.Pipeline` used in
        :func:`run_training_pipeline`.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "adaboost": AdaBoostClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            random_state=random_state,
        ),
    }

    param_grids = {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 5.0],
            "model__class_weight": [None, "balanced"],
        },
        "knn": {
            "model__n_neighbors": [5, 11, 21],
            "model__weights": ["uniform", "distance"],
        },
        "decision_tree": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 10, 20],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 10],
            "model__class_weight": [None, "balanced_subsample"],
        },
        "adaboost": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.5, 1.0],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "bagging": {
            "model__n_estimators": [100, 200],
            "model__max_samples": [0.7, 1.0],
        },
    }
    return models, param_grids


# =============================================================================
# Main pipeline
# =============================================================================

def run_training_pipeline(
    csv_path="dataset/survey.csv",
    target_col="treatment",
    k_values=(3, 5, 10),
    primary_metric="f1",
    scale_method="none",
    tune_hyperparameters=True,
    test_size=0.2,
    random_state=42,
):
    """Run the end-to-end training and evaluation pipeline.

    For each classifier in the model space, this function performs:

    1. CSV load + preprocessing via :func:`load_and_clean_data`.
    2. Stratified train/test split (holdout test set).
    3. Optional hyperparameter tuning via 5-fold :class:`GridSearchCV`.
    4. Multi-k Stratified K-Fold cross validation on the training set.
    5. Holdout evaluation of the best estimator on the test set.

    Parameters
    ----------
    csv_path : str, default="dataset/survey.csv"
        Path to the survey CSV file.
    target_col : str, default="treatment"
        Name of the binary target column.
    k_values : tuple of int, default=(3, 5, 10)
        K-fold values to evaluate during cross validation.
    primary_metric : {"f1", "accuracy", "precision", "recall", "roc_auc"}, default="f1"
        Scoring metric used both as the GridSearchCV objective and as the
        sort key for the final holdout leaderboard.
    scale_method : str, default="none"
        Reserved for future scaling-strategy comparison. Currently a no-op.
    tune_hyperparameters : bool, default=True
        If ``True``, run :class:`GridSearchCV` to find the best parameters for
        each model. If ``False``, use each model's default parameters.
    test_size : float, default=0.2
        Proportion of the data reserved for the holdout test set.
    random_state : int, default=42
        Random state for reproducible splits and models.

    Returns
    -------
    artifacts : dict
        Dictionary with the following keys:

        - ``evaluation_df`` (pandas.DataFrame) : cross-validation results,
          one row per ``(model, k)`` pair.
        - ``holdout_df`` (pandas.DataFrame) : holdout test results, sorted
          by ``primary_metric`` descending. The first five rows give the
          top-5 best combinations.
        - ``trained_models`` (dict) : fitted estimator for each model name.
        - ``X_train``, ``X_test``, ``y_train``, ``y_test`` : the data split.
        - ``target_col`` (str) : echo of the input target column.
        - ``primary_metric`` (str) : echo of the input primary metric.
    """
    # 1. Load and preprocess data
    X, y = load_and_clean_data(csv_path=csv_path, target_col=target_col)

    # 2. Define model space
    models, param_grids = _get_model_space(random_state=random_state)

    scorers = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    evaluation_rows = []
    holdout_rows = []
    trained_models = {}

    # 3. Holdout split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4. Per-model tuning, CV, and holdout evaluation
    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", "passthrough"), ("model", estimator)])

        best_estimator = pipeline
        best_params = {}

        # 4-1. Hyperparameter search
        if tune_hyperparameters:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                scoring=primary_metric,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_

        # 4-2. Multi-k cross validation
        for k in sorted(set(k_values)):
            cv = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=random_state)
            cv_scores = cross_validate(
                estimator=best_estimator,
                X=X_train,
                y=y_train,
                scoring=scorers,
                cv=cv,
                n_jobs=-1,
            )

            row = {
                "model": model_name,
                "k": int(k),
                "tuned": tune_hyperparameters,
                "params": best_params,
            }
            for metric_name in scorers.keys():
                row[f"{metric_name}_mean"] = float(np.mean(cv_scores[f"test_{metric_name}"]))
                row[f"{metric_name}_std"] = float(np.std(cv_scores[f"test_{metric_name}"]))
            evaluation_rows.append(row)

        # 4-3. Refit on full train set, evaluate on holdout
        fitted_model = clone(best_estimator).fit(X_train, y_train)
        trained_models[model_name] = fitted_model

        y_pred = fitted_model.predict(X_test)
        if hasattr(fitted_model, "predict_proba"):
            y_score = fitted_model.predict_proba(X_test)[:, 1]
        elif hasattr(fitted_model, "decision_function"):
            y_score = fitted_model.decision_function(X_test)
        else:
            y_score = y_pred

        holdout_rows.append({
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "params": best_params,
        })

    # 5. Assemble result tables
    evaluation_df = (
        pd.DataFrame(evaluation_rows)
        .sort_values(by=["model", "k"])
        .reset_index(drop=True)
    )
    holdout_df = (
        pd.DataFrame(holdout_rows)
        .sort_values(by=primary_metric, ascending=False)
        .reset_index(drop=True)
    )

    return {
        "evaluation_df": evaluation_df,
        "holdout_df": holdout_df,
        "trained_models": trained_models,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_col": target_col,
        "primary_metric": primary_metric,
    }


# =============================================================================
# Reporting and analysis
# =============================================================================

def plot_model_performance(artifacts, top_n_models=9):
    """Print result tables and plot CV / holdout performance bar charts.

    Parameters
    ----------
    artifacts : dict
        Output of :func:`run_training_pipeline`.
    top_n_models : int, default=9
        Number of top holdout models to include in the holdout bar chart.
    """
    sns.set_theme(style="whitegrid")

    print("Cross-validation results")
    print(artifacts["evaluation_df"].to_string(index=False))
    print("\nHoldout results")
    print(artifacts["holdout_df"].head(top_n_models).to_string(index=False))

    # CV by model and k
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=artifacts["evaluation_df"],
        x="model",
        y=f"{artifacts['primary_metric']}_mean",
        hue="k",
        errorbar=None,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title(f"Cross-Validation {artifacts['primary_metric'].upper()} by Model and K")
    plt.tight_layout()
    plt.show()

    # Holdout by model
    holdout_top = artifacts["holdout_df"].head(top_n_models)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=holdout_top,
        x="model",
        y=artifacts["primary_metric"],
        errorbar=None,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title(f"Holdout {artifacts['primary_metric'].upper()} by Model")
    plt.tight_layout()
    plt.show()


def export_feature_importance(artifacts, top_n=20, n_repeats=8, random_state=42):
    """Compute permutation feature importance per trained model and summarize.

    Parameters
    ----------
    artifacts : dict
        Output of :func:`run_training_pipeline`.
    top_n : int, default=20
        Number of top features to print and plot per model.
    n_repeats : int, default=8
        Number of permutation repeats per feature.
    random_state : int, default=42
        Random state for the permutation procedure.

    Returns
    -------
    summary_df : pandas.DataFrame
        Long-format dataframe of the top-``top_n`` features for each model.
    """
    importance_rows = []
    per_model_frames = []

    for model_name, trained_pipeline in artifacts["trained_models"].items():
        perm = permutation_importance(
            estimator=trained_pipeline,
            X=artifacts["X_test"],
            y=artifacts["y_test"],
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=artifacts["primary_metric"],
            n_jobs=-1,
        )

        model_importance = pd.DataFrame({
            "model": model_name,
            "feature": artifacts["X_test"].columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values(by="importance_mean", ascending=False)

        top_features = model_importance.head(top_n).copy()
        print(f"\nTop {top_n} features for {model_name}")
        print(top_features.reset_index(drop=True).to_string(index=False))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x="importance_mean", y="feature", errorbar=None)
        plt.title(f"Permutation Feature Importance ({model_name})")
        plt.tight_layout()
        plt.show()

        per_model_frames.append(top_features)
        importance_rows.extend(top_features.to_dict(orient="records"))

    summary_df = pd.DataFrame(importance_rows)

    if per_model_frames:
        merged = pd.concat(per_model_frames, ignore_index=True)
        pivot = (
            merged.pivot_table(
                index="feature",
                columns="model",
                values="importance_mean",
                aggfunc="mean",
            )
            .fillna(0.0)
        )
        pivot["avg_importance"] = pivot.mean(axis=1)
        pivot = pivot.sort_values(by="avg_importance", ascending=False)
        print("\nFeature importance consensus")
        print(pivot.to_string())

    return summary_df


__all__ = [
    # Preprocessing
    "clean_gender",
    "drop_unnecessary_columns",
    "handle_dirty_age",
    "handle_dirty_gender",
    "handle_missing_values",
    "encode_target_variable",
    "convert_country_to_continent",
    "encode_categorical_with_dummies",
    "feature_scaling_minmax",
    "preprocess",
    "load_and_clean_data",
    # Pipeline
    "run_training_pipeline",
    # Reporting
    "plot_model_performance",
    "export_feature_importance",
]
