import pandas as pd
import os
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from matplotlib import gridspec

def load_data(dataset_path):
    """Function to load data from CSV"""
    dataset_url = 'https://drive.google.com/uc?id=1-7fBYTt8mSm-LYcSRsYnCbBiQhXSDx38'

    if not os.path.exists(dataset_path):
        print('Downloading dataset...')
        gdown.download(dataset_url, dataset_path, quiet=False)
    else:
        print('Dataset already downloaded.')

    df = pd.read_csv(dataset_path)

    return df

def plot_barplot(df, col_name):
    """Function to perform basic EDA on a DataFrame."""
    col_name_counts = df[col_name].value_counts()
    print(col_name_counts)
    # Set up the figure
    plt.figure(figsize=(8, 5))

    # Create the pie chart
    sns.barplot(x=col_name_counts.index, y=col_name_counts.values,
                palette='bright')

    # Add labels and title (optional)
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.title(f'{col_name} Distribution')

    # Customize plot and background color
    plt.gca().set_facecolor('white')
    # Show the plot

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'application_type.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()

def print_missing_data(df):
    """Print missing data patterns."""
    null_percentages = df.isnull().mean().sort_values(ascending=False) * 100
    for column, percentage in null_percentages.items():
        if percentage:
            print({column: [round(percentage, 2), df[column].dtype]})


def plot_missing_data(df, threshold):
    """Plot missing data patterns."""

    # Filter columns with more than 5% missing values
    columns_with_na = df.isna().mean() * 100
    columns_with_na = columns_with_na[columns_with_na > threshold]

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Create the bar plot using Seaborn
    sns.barplot(x=columns_with_na.values, y=columns_with_na.index, palette='bright')

    # Add labels and title
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Column')
    plt.title('Percentage of Missing Values')

    # Customize plot and background color
    plt.gca().set_facecolor('white')

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'missing_data.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()

def print_uniques(df):
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")
        if df[col].nunique() < 10:
            print(df[col].value_counts(), "\n")


# Define a function to group emp_length into the desired categories
def group_emp_length(emp_length):
    if emp_length == '< 1 year':
        return '<1'
    elif emp_length in ['1 year', '2 years', '3 years', '4 years']:
        return '1-4'
    elif emp_length in ['5 years', '6 years', '7 years', '8 years']:
        return '5-8'
    else:
        return '>8'

def plot_feature_distributions(df):
    """
    Funzione che seleziona le caratteristiche con più di 2 valori unici
    e plottano gli istogrammi per ciascuna di queste caratteristiche.

    :param df: DataFrame contenente i dati
    """
    # Selezionare le caratteristiche con più di 2 valori unici
    features = [col for col in df.columns if df[col].nunique() > 2]

    # Determinare il numero di righe e colonne per la griglia dei subplot
    num_features = len(features)
    rows = (num_features + 2) // 3  # Calcolare automaticamente il numero di righe
    cols = 3  # Numero fisso di colonne

    # Configurare la figura e gli assi per i subplot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # Appiattire gli assi per iterarci facilmente sopra

    # Impostare il colore di sfondo dell'intera figura
    fig.patch.set_facecolor('lightgray')

    # Definire una palette di colori per gli istogrammi
    palette = sns.color_palette('bright', num_features)

    # Creare una copia del dataframe e aggiungere una nuova caratteristica 'diff_loan_funded'
    plotted_df = df.copy()
    plotted_df['diff_loan_funded'] = plotted_df['loan_amnt'] - plotted_df['funded_amnt']

    # Creare gli istogrammi per ciascuna caratteristica
    for i, feature in enumerate(features):
        sns.histplot(x=plotted_df[feature], kde=False, ax=axes[i], color=palette[i], alpha=1, bins=8)
        axes[i].set_title(feature)

    # Rimuovere i subplot inutilizzati, se ci sono più assi di quelli necessari
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    # Impostare il titolo principale e aggiustare il layout
    fig.suptitle("Distributions of Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Lasciare spazio per il titolo principale

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'distributions.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()

def plot_boxplot_interestrate(df):
    # Set up the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot with a custom color palette
    sns.boxplot(x='int_rate', y='loan_status', data=df, ax=ax, palette='bright')

    # Set the title and labels with improved font sizes
    ax.set_title('Loan Status by Interest Rate', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Interest Rate (%)', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel('Loan Status', fontsize=14, weight='bold', labelpad=10)

    # Improve gridlines and aesthetics for readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Display the plot with tight layout to avoid overlapping
    plt.tight_layout()

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'Loan_status_by_Interest_Rate.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()


def plot_boxplot_dti(df):
    # Creare la figura con dimensioni personalizzate
    plt.figure(figsize=(10, 6))

    # Creiamo il boxplot con un colore personalizzato usando la palette 'bright'
    sns.boxplot(x='dti', y='loan_status', data=df, palette='bright')

    # Impostiamo il titolo e le etichette con miglioramenti nella leggibilità
    plt.title('Loan Status by Debt-to-Income Ratio (DTI)', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Debt-to-Income Ratio (DTI)', fontsize=14, weight='bold', labelpad=10)
    plt.ylabel('Loan Status', fontsize=14, weight='bold', labelpad=10)

    # Aggiungiamo le linee della griglia per migliorare la leggibilità
    plt.grid(True, linestyle='--', alpha=0.7)

    # Aggiustiamo il layout per evitare sovrapposizioni
    plt.tight_layout()

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'Loan_status_by_DTI.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()


def plot_loan_status_vs_median_income(df):
    """
    Funzione che crea un grafico a barre che mostra la relazione tra lo stato del prestito (loan_status)
    e il reddito annuo mediano (annual_inc) per ogni stato, con miglioramenti estetici.

    :param df: DataFrame contenente le colonne 'loan_status' e 'annual_inc'
    """
    # Calcolare il reddito annuo mediano per ogni stato del prestito
    avg_income = df.groupby('loan_status')['annual_inc'].median().reset_index()

    # Creare la figura con dimensioni personalizzate
    plt.figure(figsize=(10, 6))

    # Creiamo il grafico a barre con una palette di colori vivaci
    sns.barplot(y='annual_inc', x='loan_status', data=avg_income, palette='bright')

    # Impostiamo il titolo e le etichette con miglioramenti nella leggibilità
    plt.title('Loan Status by Median Annual Income', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Loan Status', fontsize=14, weight='bold', labelpad=10)
    plt.ylabel('Median Annual Income', fontsize=14, weight='bold', labelpad=10)

    # Ruotiamo le etichette dell'asse X per migliorare la visibilità e l'allineamento
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Aggiungiamo le linee della griglia sull'asse Y per facilitare il confronto
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Aggiustiamo il layout per una visione più pulita
    plt.tight_layout()

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'loan_status_vs_median_income.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()

def plot_loans_per_term_and_status(df):
    """
    Funzione che crea due grafici a barre:
    1. Un grafico che mostra il numero di prestiti per termine del prestito.
    2. Un grafico che mostra la percentuale di prestiti per stato, suddivisi per termine del prestito.

    :param df: DataFrame contenente le colonne 'term', 'loan_status'.
    """
    # Calcolare il numero di prestiti per termine
    loans_per_term = df['term'].value_counts().reset_index()

    # Raggruppare per 'term' e 'loan_status' per calcolare il conteggio e la percentuale
    count_data = df.groupby(['term', 'loan_status']).size().reset_index(name='count')
    count_data['percentage'] = count_data['count'] / count_data.groupby('term')['count'].transform('sum') * 100

    # Impostiamo la figura con dimensioni maggiori
    fig = plt.figure(figsize=(16, 6))

    # Creare un layout a griglia per i grafici
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    # Primo subplot: grafico a barre per il numero di prestiti per termine
    ax1 = plt.subplot(gs[0])
    sns.barplot(x='term', y='count', data=loans_per_term, ax=ax1, width=0.4, palette='bright')
    ax1.set_title('Loans per Term', fontsize=16, weight='bold', pad=20)
    ax1.set_xlabel('Loan Term', fontsize=12, weight='bold', labelpad=10)
    ax1.set_ylabel('Number of Loans', fontsize=12, weight='bold', labelpad=10)

    # Secondo subplot: grafico a barre per la percentuale di prestiti per stato, suddivisi per termine
    ax2 = plt.subplot(gs[1])
    sns.barplot(x='term', y='percentage', hue='loan_status', data=count_data, ax=ax2, width=0.8, palette='bright')

    # Aggiungere etichette con la percentuale sopra le barre
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.text(
                p.get_x() + p.get_width() / 2.,
                height,
                f'{height:.2f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                weight='bold'
            )

    # Personalizzare i titoli e le etichette del secondo subplot
    ax2.set_title('Loan Status by Loan Term', fontsize=16, weight='bold', pad=20)
    ax2.set_xlabel('Loan Term', fontsize=12, weight='bold', labelpad=10)
    ax2.set_ylabel('Percentage per Loan Status (%)', fontsize=12, weight='bold', labelpad=10)

    # Migliorare la leggenda per una migliore leggibilità
    ax2.legend(title='Loan Status', fontsize=10, loc='upper left')

    # Aggiustare il layout per evitare sovrapposizioni e assicurarsi che tutto si adatti
    plt.tight_layout()

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'loans_per_term_and_status.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()

def plot_loans_by_subgrade(df):
    """
    Funzione che crea due grafici:
    1. Un grafico a barre che mostra il numero di prestiti per ogni sottogruppo ('sub_grade').
    2. Un grafico a barre impilate che mostra la proporzione di stato del prestito per ogni sottogruppo.

    :param df: DataFrame contenente le colonne 'sub_grade' e 'loan_status'.
    """
    # Calcolare il numero di prestiti per ogni sottogruppo ('sub_grade')
    subgrade_counts = df['sub_grade'].value_counts().sort_index()

    # Calcolare le proporzioni dello stato del prestito per ogni sottogruppo
    subgrade_proportions = df.groupby(['sub_grade', 'loan_status']).size().unstack(fill_value=0)
    subgrade_proportions = subgrade_proportions.div(subgrade_proportions.sum(axis=1), axis=0)

    # Creare la figura e gli assi per i sottogruppi
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax1, ax2 = axes[0], axes[1]

    # Plot 1: Numero di prestiti per sottogruppo ('sub_grade')
    sns.barplot(x=subgrade_counts.index, y=subgrade_counts.values, ax=ax1, width=0.6, palette='bright')
    ax1.set_title('Number of Loans per Subgrade', fontsize=16, weight='bold', pad=20)
    ax1.set_xlabel('Subgrade', fontsize=12, weight='bold', labelpad=10)
    ax1.set_ylabel('Number of Loans', fontsize=12, weight='bold', labelpad=10)
    ax1.tick_params(axis='x', rotation=45)  # Ruotare le etichette sull'asse x per leggibilità

    # Plot 2: Proporzioni di stato del prestito per ogni sottogruppo (grafico a barre impilate)
    subgrade_proportions.plot(kind='bar', stacked=True, ax=ax2, color=sns.color_palette('bright', n_colors=subgrade_proportions.shape[1]))
    ax2.set_title('Loan Status by Subgrade', fontsize=16, weight='bold', pad=20)
    ax2.set_xlabel('Subgrade', fontsize=12, weight='bold', labelpad=10)
    ax2.set_ylabel('Proportion', fontsize=12, weight='bold', labelpad=10)
    ax2.tick_params(axis='x', rotation=45)  # Ruotare le etichette sull'asse x per leggibilità
    ax2.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title_fontsize='large')

    # Migliorare la disposizione per evitare sovrapposizioni
    plt.tight_layout()

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'loans_by_subgrade.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()


def loan_status_by_purpose(df):
    # Raggruppa per 'purpose' e calcola il numero di prestiti
    purpose_status_counts = df.groupby(['purpose']).size().sort_values(ascending=False)

    # Stampa i risultati
    print("Loan Status by Purpose (Counts):")
    print(purpose_status_counts)


# Definisci la funzione per raggruppare 'emp_length' nelle categorie desiderate
def group_emp_length(emp_length):
    if emp_length == '< 1 year':
        return '<1'
    elif emp_length in ['1 year', '2 years', '3 years', '4 years']:
        return '1-4'
    elif emp_length in ['5 years', '6 years', '7 years', '8 years']:
        return '5-8'
    else:
        return '>8'

def plot_emp_length_vs_loan_status(df):
    # Definisci l'ordine desiderato per i gruppi di emp_length
    emp_length_groups = ['<1', '1-4', '5-8', '>8']

    # Crea una copia del DataFrame per evitare modifiche all'originale
    df_copy = df.copy()

    # Applica la funzione di raggruppamento per creare una nuova colonna in df_copy
    df_copy['emp_length_group'] = df_copy['emp_length'].apply(group_emp_length)

    # Converti la colonna 'emp_length_group' in un tipo categorico con l'ordine specificato
    df_copy['emp_length_group'] = pd.Categorical(df_copy['emp_length_group'], categories=emp_length_groups, ordered=True)

    # Imposta la figura per il grafico
    plt.figure(figsize=(10, 6))

    # Crea l'istogramma impilato con i dati raggruppati di 'emp_length'
    sns.histplot(data=df_copy, x="emp_length_group", hue="loan_status", multiple="stack", palette='bright')

    # Aggiungi titolo e etichette agli assi
    plt.title("Relationship between Employment Length and Loan Status")
    plt.xlabel("Employment Length")
    plt.ylabel("Count")

    folder = 'images'
    if not os.path.exists(folder):  # Se la cartella non esiste, la crea
        os.makedirs(folder)

    filename = 'emp_length_vs_loan_status.png'

    path = os.path.join(folder, filename)

    plt.savefig(path)

    plt.show()

    return df