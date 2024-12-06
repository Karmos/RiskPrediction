import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib


# Funzione generica per Grid Search
def train_with_grid_search(model, param_grid, X_train, y_train, scoring='f1_weighted', cv=5, n_jobs=-1, verbose=2):
    """
    Addestra un modello utilizzando la Grid Search per ottimizzare gli iperparametri.
    :param model: Istanza del modello sklearn.
    :param param_grid: Dizionario con la griglia degli iperparametri.
    :param X_train: Feature di training.
    :param y_train: Target di training.
    :param scoring: Metrica da ottimizzare (default: 'f1_weighted').
    :param cv: Numero di fold per la validazione incrociata.
    :param n_jobs: Numero di processi paralleli (default: -1 per tutti).
    :param verbose: Livello di verbosità.
    :return: Modello ottimizzato, risultati della Grid Search.
    """
    print(f"Avvio della Grid Search per {model.__class__.__name__}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs,
                               verbose=verbose)
    grid_search.fit(X_train, y_train)
    print("Grid Search completata.")
    print("Migliori parametri trovati: ", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.cv_results_


# Funzione per valutare il modello
def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello sui dati di test.
    :param model: Modello addestrato.
    :param X_test: Feature di test.
    :param y_test: Target di test.
    :return: Dizionario con le metriche di valutazione.
    """
    print(f"Valutazione del modello {model.__class__.__name__}...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    print("Accuracy:", metrics["accuracy"])
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


# Funzione per salvare i risultati della Grid Search
def save_grid_search_results(results, columns_to_save, file_name):
    """
    Salva i risultati della Grid Search in un file CSV.
    :param results: Risultati della Grid Search (cv_results_ di GridSearchCV).
    :param columns_to_save: Colonne da includere nel file.
    :param file_name: Nome del file CSV.
    """
    print(f"Salvataggio dei risultati della Grid Search in {file_name}...")
    results_df = pd.DataFrame(results)
    results_df[columns_to_save].to_csv(file_name, index=False)
    print(f"Risultati salvati con successo in: {file_name}")


# Funzione per salvare il modello
def save_model(model, file_name):
    """
    Salva il modello addestrato in un file.
    :param model: Modello addestrato.
    :param file_name: Nome del file per il modello.
    """
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print(f"Salvataggio del modello in {file_name}...")
    joblib.dump(model, file_name)
    print(f"Modello salvato con successo in: {file_name}")
