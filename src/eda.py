import pandas as pd
import os
import gdown
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Function to load data from CSV"""
    dataset_url = 'https://drive.google.com/uc?id=1-7fBYTt8mSm-LYcSRsYnCbBiQhXSDx38'
    dataset_path = os.path.abspath('../dataset/dataset.csv')

    if not os.path.exists(dataset_path):
        print('Downloading dataset...')
        gdown.download(dataset_url, dataset_path, quiet=False)

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

    # Show the plot
    plt.show()

def pippo(df):
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


def grade(df):
    grades = sorted(df['grade'].unique())
    print("Possible grades: ", grades)
    sub_grades = sorted(df['sub_grade'].unique())
    print("Possible subgrades: ", sub_grades)
    df.drop(columns=['grade'], inplace=True)
    emp_length_groups = ['<1', '1-4', '5-8', '>8']

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Apply the grouping function to create a new column in df_copy
    df_copy['emp_length_group'] = df_copy['emp_length'].apply(group_emp_length)

    # Convert the new 'emp_length_group' column to a categorical type with the specified order
    df_copy['emp_length_group'] = pd.Categorical(df_copy['emp_length_group'], categories=emp_length_groups,
                                                 ordered=True)

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Create the stacked histogram with the grouped emp_length data
    sns.histplot(data=df_copy, x="emp_length_group", hue="loan_status", multiple="stack", palette='bright')

    # Add title and axis labels
    plt.title("Relationship between Employment Length and Loan Status")
    plt.xlabel("Employment Length")
    plt.ylabel("Count")

    # Show the plot
    plt.show()


def plot_boxplot(df):
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

    # Show the plot
    plt.show()


def plot_feature_distribution(data, feature):
    """TO DO"""
    sns.histplot(data[feature])
    plt.show()