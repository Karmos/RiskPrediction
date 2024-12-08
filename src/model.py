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
    print(f"Avvio della Grid Search per {model.__class__.__name__}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs,
                               verbose=verbose)
    grid_search.fit(X_train, y_train)
    print("Grid Search completata.")
    print("Migliori parametri trovati: ", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.cv_results_


# Funzione per valutare il modello
def evaluate_model(model, results, X_test, y_test):
    """
    Valuta il modello sui dati di test.
    :param model: Modello addestrato.
    :param X_test: Feature di test.
    :param y_test: Target di test.
    :return: Dizionario con le metriche di valutazione.
    """
    print(f"Valutazione del modello {model.__class__.__name__}...")
    y_pred = model.predict(X_test)

    results = pd.DataFrame(results)
    results['precision'] = precision_score(y_test, y_pred, average='weighted')
    results['recall'] = recall_score(y_test, y_pred, average='weighted')
    results['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    results['accuracy'] = accuracy_score(y_test, y_pred)

    return results


# Funzione per salvare i risultati della Grid Search
def save_grid_search_results(results, columns_to_save, file_name):

    print(f"Salvataggio dei risultati della Grid Search in {file_name}...")
    # Seleziona le colonne principali per il CSV
    results_to_save = results[columns_to_save].copy()

    # Identifica il miglior modello basato su mean_test_score
    best_index = results['mean_test_score'].idxmax()  # Indice del miglior punteggio medio

    # Aggiungi una colonna che indica se il modello è il migliore
    results_to_save['is_best_model'] = False
    results_to_save.loc[best_index, 'is_best_model'] = True

    # Salva i risultati in un file CSV
    results_to_save.to_csv(file_name, index=False)
    print(f"Risultati salvati con successo in: {file_name}")


# Funzione per salvare il modello
def save_model(model, file_name):

    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print(f"Salvataggio del modello in {file_name}...")
    joblib.dump(model, file_name)
    print(f"Modello salvato con successo in: {file_name}")
