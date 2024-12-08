import numpy as np
import pandas as pd
import os
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import shap


def plot_confusion_matrix(y_true, y_pred, model_name):
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

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'confusion_matrix' + model_name + '.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    # Mostra il grafico
    plt.show()
#%%
def plot_feature_importance(model, X, model_name):
    feature_importance = model.feature_importances_

    # Sort the feature importances in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    top_sorted_idx = sorted_idx[:28]

    # Plot the top 28 feature importances based on original feature names
    plt.figure(figsize=(12, 6))
    plt.bar(range(28), feature_importance[top_sorted_idx])
    plt.title(f'Top {28} Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(range(28), np.array(X.columns)[top_sorted_idx], rotation=90)  # If X is a DataFrame

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'feature_importance_' + model_name + '.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    # Mostra il grafico
    plt.show()


def analyze_rf_model(X_train_copy, X_test_copy, class_names, model_name):
    # Carica il modello RandomForest
    if (model_name == 'randomForest_smartLending'):
        best_rf_model_path = 'models/smartLending/best_xgb_model.pkl'
    else:
        best_rf_model_path = 'models/creditRisk/best_rf_model.pkl'
    best_rf_model = joblib.load(open(best_rf_model_path, 'rb'))

    # Traccia l'importanza delle caratteristiche
    plot_feature_importance(best_rf_model, pd.DataFrame(X_test_copy), model_name)

    # Calcolare i valori SHAP
    explainer = shap.TreeExplainer(best_rf_model, pd.DataFrame(X_train_copy))
    shap_values = explainer(pd.DataFrame(X_test_copy), check_additivity=False)

    # Traccia i grafici di SHAP per ciascuna classe
    shap.summary_plot(shap_values, pd.DataFrame(X_test_copy).values, plot_type='bar',
                      class_names=class_names, feature_names=pd.DataFrame(X_test_copy).columns.tolist())
