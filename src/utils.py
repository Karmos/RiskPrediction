import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_model(model, filename):
    filename = './models'+'/'+filename
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Modello salvato con successo in: {filename}")
    except Exception as e:
        print(f"Errore nel salvataggio del modello: {e}")

#%%
def save_grid_search_results(results, columns_to_save, filename):
    filename = './results'+'/'+filename

    # Seleziona le colonne principali per il CSV
    results_to_save = results[columns_to_save].copy()

    # Identifica il miglior modello basato su mean_test_score
    best_index = results['mean_test_score'].idxmax()  # Indice del miglior punteggio medio

    # Aggiungi una colonna che indica se il modello è il migliore
    results_to_save['is_best_model'] = False
    results_to_save.loc[best_index, 'is_best_model'] = True


    # Salva i risultati in un file CSV
    results_to_save.to_csv(filename, index=False)
    print(f"Risultati salvati con successo in: {filename}")
#%%
def plot_confusion_matrix(y_true, y_pred):
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)

    # Configura la visualizzazione
    plt.figure(figsize=(8,6))
    class_labels = sorted(set(y_true) | set(y_pred))  # Unisce le classi uniche di y_true e y_pred
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

    # Aggiungi titolo e etichette
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)

    # Mostra il grafico
    plt.show()
#%%
def plot_feature_importance(model, pca, X):
    importances = model.feature_importances_
    pca_loadings = pca.components_

    # The importance of each feature is the sum of the absolute values of its loadings across all components
    feature_importance = np.sum(np.abs(pca_loadings), axis=0)

    # Step 8: Sort the feature importances in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    top_sorted_idx = sorted_idx[:28]

    # Step 10: Plot the top 28 feature importances based on original feature names
    plt.figure(figsize=(12, 6))
    plt.bar(range(28), feature_importance[top_sorted_idx])
    plt.title(f'Top {28} Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(range(28), np.array(X.columns)[top_sorted_idx], rotation=90)  # If X is a DataFrame
    plt.show()