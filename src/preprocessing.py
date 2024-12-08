import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def transform_skewed_features(df, threshold=0.5):
    """
    Funzione che seleziona le colonne numeriche, calcola la skewness, e applica la trasformazione Box-Cox
    alle caratteristiche skewed (asimmetriche) se i valori sono positivi.

    :param df: DataFrame contenente i dati
    :param threshold: Soglia di skewness oltre la quale si applica la trasformazione Box-Cox (default 0.5)
    :return: DataFrame con le colonne skewed trasformate
    """
    # Selezionare solo le colonne numeriche
    numeric_df = df.select_dtypes(include=[np.number])

    # Calcolare la skewness delle colonne numeriche
    skewness = numeric_df.skew()

    # Ottenere una lista delle caratteristiche skewed (con skewness maggiore della soglia)
    skewed_features = skewness[skewness > threshold].index

    # Applicare la trasformazione Box-Cox alle colonne skewed (solo se i valori sono strettamente positivi)
    for feature in skewed_features:
        if (df[feature] > 0).all():  # Verifica che tutti i valori siano positivi
            # Applicare Box-Cox e ottenere il valore di lambda
            df[feature], lambda_val = stats.boxcox(df[feature])
            print(f"Box-Cox applied to {feature} with lambda = {lambda_val:.4f}")
        else:
            pass
            #print(f"Skipping {feature} as it contains non-positive values.")

    return df

def imputing(df):
    """Function to preprocess the data, handle missing values, etc."""
    if df.isnull().any().any():
        # Initialize the IterativeImputer with optimized settings
        imputer = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=5)
        # Apply the IterativeImputer to the DataFrame
        imputed_data = imputer.fit_transform(df)
        # Convert the imputed data back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
        imputed_df.to_csv('./dataset/preprocessed_data.csv', index=False)
    return imputed_df


def encode_data(df):

    label_encoder = LabelEncoder()
    df['term'] = label_encoder.fit_transform(df['term'])
    df['initial_list_status'] = label_encoder.fit_transform(df['initial_list_status'])
    # %%
    df['verification_status'] = df['verification_status'].apply(verification_status)
    # %%
    categorical_features = df.select_dtypes(include='object').drop(columns=['loan_status'])
    encoded_features = pd.get_dummies(categorical_features, dtype=int)
    df = pd.concat([df, encoded_features], axis=1)
    # %%
    df.drop(columns=categorical_features.columns, inplace=True)
    return df


def sub_grades_encoding(df):
    # Define a dictionary mapping letter grades to numeric values
    grade_map = {
        'A': 7,
        'B': 6,
        'C': 5,
        'D': 4,
        'E': 3,
        'F': 2,
        'G': 1
    }

    # Vectorized operation for letter grade encoding
    grade_vals = df['sub_grade'].str.extract('([A-G])')[0].map(grade_map)

    # Vectorized operation for numeric adjustment based on 1-5
    numeric_adjustments = {
        '1': 0.8,
        '2': 0.6,
        '3': 0.4,
        '4': 0.2,
        '5': 0.0
    }

    # Extract the numeric suffix (1, 2, 3, etc.) and apply the corresponding adjustment
    numeric_vals = df['sub_grade'].str.extract('([1-5])')[0].map(numeric_adjustments).fillna(0)

    # Combine the grade and numeric adjustments
    df['sub_grade'] = grade_vals + numeric_vals

    return df

# verification status encoding
def verification_status(x):
    # Return 0 for 'Not Verified' and 1 for any other status
    return 0 if x == 'Not Verified' else 1


def process_loan_status(df):
    # Definire i valori di loan_status validi
    status_to_consider = ['Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)',
                          'Default', 'Charged Off']

    # Filtrare il DataFrame per mantenere solo i loan_status validi
    df = df[df['loan_status'].isin(status_to_consider)]

    # Mappatura dei livelli di rischio
    risk_mapping = {
        'Fully Paid': 0,  # basso
        'Current': 0,
        'In Grace Period': 0,  # medio
        'Late (16-30 days)': 0,
        'Late (31-120 days)': 1,  # alto
        'Default': 1,  # alto
        'Charged Off': 1
    }

    # Creare una nuova colonna 'risk_level' basata sulla mappatura di loan_status
    df['risk_level'] = df['loan_status'].map(risk_mapping)

    # Rimuovere la colonna 'loan_status'
    df.drop(columns=['loan_status'], inplace=True)

    # Restituire il DataFrame con la colonna 'risk_level'
    return df


def scale_data(X_train, X_test, X):
    """Function to scale the features."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    return X_train, X_test


def pca_analysis(X_train, X_test, variance_threshold=0.9):
    # Esegui la PCA per determinare la varianza spiegata cumulativa
    pca = PCA(n_components=len(X_train.columns))
    pca.fit(X_train)

    # Calcolare la varianza spiegata cumulativa
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Creare un DataFrame per la varianza spiegata cumulativa
    cvr = pd.DataFrame({
        'Number of Principal Components': range(1, len(cumulative_variance) + 1),
        'Cumulative Explained Variance Ratio': cumulative_variance
    })

    # Creare il grafico della varianza spiegata cumulativa
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cvr,
                 x='Number of Principal Components',
                 y='Cumulative Explained Variance Ratio',
                 marker='o')  # Aggiungere i marcatori per chiarezza
    plt.title('Cumulative Explained Variance Ratio by Number of Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # Determinare il numero di componenti principali da mantenere in base alla soglia di varianza
    components_to_retain = np.argmax(cumulative_variance >= variance_threshold) + 1

    print(components_to_retain)

    # Esegui la PCA con il numero di componenti scelto
    pca = PCA(n_components=components_to_retain)

    X_train_reduced = pca.fit_transform(X_train)

    X_test_reduced = pca.transform(X_test)

    return X_train_reduced, X_test_reduced
