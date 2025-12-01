"""
Name: Your Name
Student ID: Your Student ID

File: spam_detector.py
Description:
    NLP spam detection project using a small SMS Spam sample dataset.
    Includes preprocessing, model training, validation, testing on unseen data,
    and a simple live demo function.
"""

import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import nltk
from nltk.corpus import stopwords


def load_dataset(filepath: str) -> pd.DataFrame:
    # // Load dataset into a pandas DataFrame. Handles common encodings
    # // and simple CSV/TSV formats used in small sample datasets.
    # // Returns DataFrame with exactly two columns: 'label' and 'text'.
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    except pd.errors.ParserError:
        df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "text"])

    df.columns = [c.lower() for c in df.columns]

    if "label" not in df.columns:
        if "v1" in df.columns:
            df.rename(columns={"v1": "label"}, inplace=True)
        else:
            raise ValueError("Could not find 'label' column in dataset.")

    if "text" not in df.columns:
        if "v2" in df.columns:
            df.rename(columns={"v2": "text"}, inplace=True)
        else:
            raise ValueError("Could not find 'text' column in dataset.")

    df = df[["label", "text"]].dropna()

    return df


def clean_text(text: str, stop_words: set) -> str:
    # // Normalize the SMS: lowercase, remove urls, numbers, punctuation.
    # // Then split into tokens and remove stopwords and single-letter tokens.
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    cleaned = " ".join(tokens)

    return cleaned


def preprocess_texts(texts: pd.Series) -> pd.Series:
    # // Prepare a stopword set. NLTK requires downloading the corpus before
    # // use; that download can fail in restricted environments. To make the
    # // script more robust, try to load NLTK stopwords and fall back to a
    # // small built-in set if unavailable.
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        # // Fallback minimal stopwords set to keep preprocessing usable.
        stop_words = {
            "the",
            "and",
            "is",
            "in",
            "it",
            "to",
            "a",
            "of",
            "for",
            "on",
            "you",
            "i",
            "that",
        }

    # // Apply cleaning to each text value in the pandas Series.
    return texts.apply(lambda x: clean_text(str(x), stop_words))


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
    - df: DataFrame with 'label' and 'text' columns.
    - test_size: Proportion of data to reserve for final testing.
    - val_size: Proportion of remaining training data for validation.
    - random_state: Random seed for reproducibility.

    Returns:
    - X_train, X_val, X_test: Text data splits.
    - y_train, y_val, y_test: Label splits.
    """
    X = df["text"]
    y = df["label"]

    # For very small sample datasets, avoid stratify and use simple splits
    if len(df) < 50:
        # For tiny sample datasets used just to verify the pipeline,
        # use a simple split: 60% train, 20% val, 20% test, no further splitting.
        # // For tiny datasets (common in examples), avoid stratified splits
        # // because stratify can fail when classes have very few examples.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=random_state,
        )
    else:
        # // For larger datasets, use stratified splitting so train/val/test
        # // keep the same class balance as the original data.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        val_ratio_of_temp = val_size / (1.0 - test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_ratio_of_temp,
            random_state=random_state,
            stratify=y_temp,
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_vectorizer_and_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
):
    """
    Fit a TF-IDF vectorizer and a Logistic Regression classifier.

    Parameters:
    - X_train: Training texts.
    - y_train: Training labels ('spam'/'ham').
    - X_val: Validation texts.
    - y_val: Validation labels.

    Returns:
    - vectorizer: Trained TfidfVectorizer.
    - model: Trained LogisticRegression model.
    """
    # // Create a TF-IDF vectorizer that converts raw text into numeric
    # // features suitable for the Logistic Regression model. We limit
    # // `max_features` to keep the example lightweight.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # // Fit the vectorizer on training texts and transform both train/val.
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # // Train a Logistic Regression classifier. `class_weight='balanced'`
    # // helps when dataset classes are imbalanced (spam vs ham counts).
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    # // Quick validation check to show how the model performs on held-out
    # // validation data while tuning. This is not the final test metric.
    val_preds = model.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    return vectorizer, model


def evaluate_model(
    model,
    vectorizer,
    X_test: pd.Series,
    y_test: pd.Series,
    title: str = "Confusion Matrix",
):
    """
    Evaluate the trained model on unseen test data and show metrics.

    Parameters:
    - model: Trained classification model.
    - vectorizer: Trained TF-IDF vectorizer.
    - X_test: Unseen test texts.
    - y_test: True labels for test data.
    - title: Title for confusion matrix plot.

    Returns:
    - None. Prints metrics and displays a confusion matrix.
    """
    # // Transform test texts and run predictions to evaluate final performance.
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (Unseen Data): {acc:.4f}\n")

    # // Print a more complete classification report (precision/recall/f1).
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # // Create and show a confusion matrix to visualise errors.
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_artifacts(
    model,
    vectorizer,
    model_path: str = "spam_model.joblib",
    vec_path: str = "tfidf_vectorizer.joblib",
) -> None:
    """
    Save the trained model and vectorizer to disk.

    Parameters:
    - model: Trained classifier.
    - vectorizer: Trained TF-IDF vectorizer.
    - model_path: Path to save the model file.
    - vec_path: Path to save the vectorizer file.

    Returns:
    - None.
    """
    # // Persist trained objects so you can load them later without
    # // retraining. This writes binary files using joblib.
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Saved model to {model_path} and vectorizer to {vec_path}")


def load_artifacts(
    model_path: str = "spam_model.joblib",
    vec_path: str = "tfidf_vectorizer.joblib",
):
    """
    Load the trained model and vectorizer from disk.

    Parameters:
    - model_path: Path to the saved model file.
    - vec_path: Path to the saved vectorizer file.

    Returns:
    - model: Loaded classifier.
    - vectorizer: Loaded TF-IDF vectorizer.
    """
    # // Load previously saved model and vectorizer from disk.
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict_single_sms(model, vectorizer, sms_text: str) -> str:
    """
    Predict whether a single SMS message is spam or ham.

    Parameters:
    - model: Trained classifier.
    - vectorizer: Trained TF-IDF vectorizer.
    - sms_text: SMS message to classify.

    Returns:
    - Predicted label as a string ('spam' or 'ham').
    """
    # // Wrap the single message in a pandas Series because the vectorizer
    # // expects an iterable of documents. Then transform and predict.
    sms_series = pd.Series([sms_text])
    sms_tfidf = vectorizer.transform(sms_series)
    pred = model.predict(sms_tfidf)[0]
    return pred


def live_demo(model, vectorizer) -> None:
    """
    Run a simple live demo allowing the user to type SMS messages
    and get spam/ham predictions in the console.

    Parameters:
    - model: Trained classifier.
    - vectorizer: Trained TF-IDF vectorizer.

    Returns:
    - None. Runs an input loop until user types 'exit'.
    """
    # // Simple interactive loop for manual testing/demo. Type 'exit' to
    # // quit. This is useful when showing the model live in a classroom.
    print("\n--- Live SMS Spam Detection Demo ---")
    print("Type an SMS message and press Enter to classify it.")
    print("Type 'exit' to quit.\n")

    while True:
        sms = input("Enter SMS: ")
        if sms.lower().strip() == "exit":
            print("Exiting demo.")
            break

        label = predict_single_sms(model, vectorizer, sms)
        print(f"Prediction: {label.upper()}\n")


def main() -> None:
    """
    Main function to execute the full NLP pipeline:
    - Load dataset
    - Preprocess texts
    - Split into train/val/test
    - Train model
    - Evaluate on unseen test data
    - Save artefacts
    - Run a sample demo prediction
    """
    # // Main pipeline orchestration. This function runs each step in order
    # // and prints helpful messages so you can follow the flow when you run
    # // the script in a terminal.
    dataset_path = "spam.csv"
    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)

    print("Original dataset size:", len(df))
    print(df.head())

    print("\nPreprocessing texts...")
    # // Try to download stopwords only if necessary. If download is not
    # // possible, preprocess_texts already falls back to a small set.
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

    df["text"] = preprocess_texts(df["text"])

    print("Splitting data into train/validation/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("Training model...")
    vectorizer, model = train_vectorizer_and_model(X_train, y_train, X_val, y_val)

    print("\nEvaluating on unseen test data...")
    evaluate_model(model, vectorizer, X_test, y_test)

    save_artifacts(model, vectorizer)

    example_sms = "Congratulations! You have won a free ticket. Call now."
    example_pred = predict_single_sms(model, vectorizer, example_sms)
    print(f"\nExample SMS: {example_sms}")
    print(f"Predicted label: {example_pred}\n")

    # // For live demo in class, keep the next line uncommented.
    # live_demo(model, vectorizer)


if __name__ == "__main__":
    main()


