# ======================
# IMPORT LIBRARIES
# ======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.utils import resample
from itertools import combinations

# ======================
# HELPER FUNCTIONS
# ======================
def bootstrap_ci(data, n_iterations=1000):
    """Calculate 95% CI using bootstrap resampling"""
    stats = []
    for _ in range(n_iterations):
        sample = resample(data)
        stats.append(sample.mean())
    return np.percentile(stats, [2.5, 97.5])

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2)/2)
    return diff / pooled_std

def rank_correlations(data, target_vars):
    """Rank variables by absolute correlation with targets"""
    corr_matrix = data.corr()
    return pd.concat([
        corr_matrix[target].abs().sort_values(ascending=False)
        for target in target_vars
    ], axis=1, keys=target_vars)

def standardized_model(formula, data):
    """Run regression with standardized coefficients"""
    df_std = data.copy()
    for col in df_std.select_dtypes(include=[np.number]):
        df_std[col] = (df_std[col] - df_std[col].mean())/df_std[col].std()
    return smf.ols(formula, data=df_std).fit()

# ======================
# DATA PREPARATION
# ======================
def load_and_clean(filepath):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(filepath)[
            ['Age', 'Gender', 'Employment_Status', 'Financial_Stress',
             'Sleep_Hours', 'Physical_Activity_Hrs', 'Anxiety_Score',
             'Depression_Score', 'Stress_Level', 'Social_Support_Score']
        ].dropna()

        # Clean outliers using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR)))]  # Fix outlier removal

        # Encode categorical variables
        df['Gender'] = df['Gender'].map({'Male':0, 'Female':1, 'Non-Binary':2})
        df['Employment_Status'] = df['Employment_Status'].map({
            'Employed':0, 'Unemployed':1, 'Retired':2, 'Student':3})
        
        # Create binned versions for visualization
        df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 75], 
                                labels=['18-29', '30-44', '45-59', '60+'])
        df['Sleep_Quality'] = pd.cut(df['Sleep_Hours'], bins=[0, 5, 7, 24],
                                    labels=['Poor (<5h)', 'Average (5-7h)', 'Good (>7h)'])
        df['Activity_Level'] = pd.qcut(df['Physical_Activity_Hrs'], q=3,
                                     labels=['Low', 'Medium', 'High'])
        df['Support_Level'] = pd.qcut(df['Social_Support_Score'], q=2,
                                    labels=['Low Support', 'High Support'])
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please check the path.")
        exit()

# ======================
# VISUALIZATION SETTINGS
# ======================
def set_visual_style():
    """Configure consistent visual style for all plots"""
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titlepad'] = 15

# ======================
# ANALYSIS FUNCTIONS
# ======================
def describe_data(data):
    """Generate comprehensive descriptive statistics"""
    print("\n=== Sample Characteristics ===")
    print(f"Final sample size: {len(data)}")
    
    print("\n=== Gender Distribution ===")
    gender_counts = data['Gender'].value_counts().rename({0:'Male',1:'Female',2:'Non-Binary'})
    print(gender_counts)
    
    print("\n=== Employment Status ===")
    emp_counts = data['Employment_Status'].value_counts().rename(
        {0:'Employed',1:'Unemployed',2:'Retired',3:'Student'})
    print(emp_counts)
    
    print("\n=== Mental Health Scores ===")
    print(data[['Anxiety_Score', 'Depression_Score', 'Stress_Level']].describe())
    
    print("\n=== Lifestyle Factors ===")
    print(data[['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score']].describe())
    
    return gender_counts, emp_counts

def stratified_analysis(data):
    """Analyze relationships across subgroups"""
    # 1. Key correlations
    key_corr = data[['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score',
                    'Depression_Score', 'Stress_Level', 'Financial_Stress']].corr()
    print("\n=== Key Variable Correlations ===")
    print(key_corr.loc[['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score'],
                      ['Depression_Score', 'Stress_Level', 'Financial_Stress']])
    
    # 2. Effect sizes by employment status
    print("\n=== Effect Sizes by Employment Status ===")
    for var in ['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score']:
        employed = data[data['Employment_Status']==0][var]
        unemployed = data[data['Employment_Status']==1][var]
        print(f"\n{var} difference (Employed vs Unemployed):")
        print(f"Cohen's d: {cohens_d(employed, unemployed):.2f}")
    
    return key_corr

# ======================
# CORRELATION MATRICES
# ======================
def plot_correlation_matrices(data):
    """Generate upper triangle correlation matrices for Pearson and Spearman"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_data = data[numeric_cols]
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
    
    # Pearson correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data.corr(method='pearson'), mask=mask, annot=True, 
                cmap='coolwarm', center=0, fmt='.2f', square=True,
                cbar_kws={'shrink': 0.7})
    plt.title('Pearson Correlation Matrix (Upper Triangle)')
    plt.tight_layout()
    plt.savefig('pearson_correlation_matrix.png')
    plt.close()
    print("Saved pearson_correlation_matrix.png")
    
    # Spearman correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data.corr(method='spearman'), mask=mask, annot=True, 
                cmap='viridis', center=0, fmt='.2f', square=True,
                cbar_kws={'shrink': 0.7})
    plt.title('Spearman Correlation Matrix (Upper Triangle)')
    plt.tight_layout()
    plt.savefig('spearman_correlation_matrix.png')
    plt.close()
    print("Saved spearman_correlation_matrix.png")

# ======================
# REGRESSION ANALYSIS
# ======================
def perform_regression(data):
    """Perform multiple regression analysis for anxiety and depression"""
    # Define common predictors
    predictors = ['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score', 
                 'Financial_Stress', 'Stress_Level', 'Age', 'Gender', 'Employment_Status']
    
    print("\n=== Regression Analysis ===")
    
    # Anxiety Score Regression
    print("\n=== Anxiety Score Regression ===")
    anxiety_formula = 'Anxiety_Score ~ ' + ' + '.join(predictors)
    anxiety_model = smf.ols(anxiety_formula, data=data).fit()
    print(anxiety_model.summary())
    
    # Standardized Anxiety Model
    print("\n=== Standardized Anxiety Model ===")
    std_anxiety_model = standardized_model(anxiety_formula, data)
    print(std_anxiety_model.summary())
    
    # Depression Score Regression
    print("\n=== Depression Score Regression ===")
    depression_formula = 'Depression_Score ~ ' + ' + '.join(predictors)
    depression_model = smf.ols(depression_formula, data=data).fit()
    print(depression_model.summary())
    
    # Standardized Depression Model
    print("\n=== Standardized Depression Model ===")
    std_depression_model = standardized_model(depression_formula, data)
    print(std_depression_model.summary())
    
    return {
        'anxiety_model': anxiety_model,
        'std_anxiety_model': std_anxiety_model,
        'depression_model': depression_model,
        'std_depression_model': std_depression_model
    }

# ======================
# INDIVIDUAL VISUALIZATION FUNCTIONS
# ======================

# Sleep vs Depression by Age
def plot_sleep_depression_age(data):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Sleep_Quality', y='Depression_Score', hue='Age_Group', 
                    data=data, palette='coolwarm')
    plt.title('Depression Scores by Sleep Quality and Age Group')
    plt.xlabel('Sleep Quality')
    plt.ylabel('Depression Score')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('sleep_depression_age.png')
    plt.close()
    print("Saved sleep_depression_age.png")

# Activity vs Stress by Employment
def plot_activity_stress_employment(data):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Activity_Level', y='Stress_Level', hue='Employment_Status',
                    data=data.assign(Employment_Status=lambda x: x['Employment_Status'].map(
                        {0:'Employed',1:'Unemployed',2:'Retired',3:'Student'})),
                    palette='viridis')
    plt.title('Stress Levels by Activity Level and Employment Status')
    plt.xlabel('Physical Activity Level')
    plt.ylabel('Stress Level')
    plt.legend(title='Employment', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('activity_stress_employment.png')
    plt.close()
    print("Saved activity_stress_employment.png")

# Social Support vs Financial Stress by Gender
def plot_support_financial_gender(data):
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x='Social_Support_Score', y='Financial_Stress', hue='Gender',
                        data=data.assign(Gender=lambda x: x['Gender'].map(
                            {0:'Male',1:'Female',2:'Non-Binary'})),
                        palette='Set2', alpha=0.7, s=100)
    plt.title('Financial Stress vs Social Support by Gender')
    plt.xlabel('Social Support Score')
    plt.ylabel('Financial Stress Level')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig('support_financial_gender.png')
    plt.close()
    print("Saved support_financial_gender.png")

# Lifestyle Factors Correlation with Depression
def plot_lifestyle_depression_corr(data):
    plt.figure(figsize=(8, 6))
    corr_data = data[['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score',
                     'Depression_Score']].corr()[['Depression_Score']].sort_values(
                         'Depression_Score', ascending=False)
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
               cbar_kws={'label': 'Correlation with Depression'})
    plt.title('Correlation of Lifestyle Factors with Depression')
    plt.tight_layout()
    plt.savefig('lifestyle_depression_corr.png')
    plt.close()
    print("Saved lifestyle_depression_corr.png")

# Sleep × Activity Interaction on Depression
def plot_sleep_activity_interaction(data):
    plt.figure(figsize=(10, 6))
    ax = sns.pointplot(x='Sleep_Quality', y='Depression_Score', hue='Activity_Level',
                      data=data, palette='magma', dodge=0.2, markers=['o', 's', 'D'])
    plt.title('Interaction of Sleep Quality and Activity Level on Depression')
    plt.xlabel('Sleep Quality')
    plt.ylabel('Depression Score')
    plt.legend(title='Activity Level')
    plt.tight_layout()
    plt.savefig('sleep_activity_interaction.png')
    plt.close()
    print("Saved sleep_activity_interaction.png")

# Employment × Support Interaction on Stress
def plot_employment_support_interaction(data):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Employment_Status', y='Stress_Level', hue='Support_Level',
                    data=data.assign(Employment_Status=lambda x: x['Employment_Status'].map(
                        {0:'Employed',1:'Unemployed',2:'Retired',3:'Student'})))
    plt.title('Interaction of Employment and Social Support on Stress')
    plt.xlabel('Employment Status')
    plt.ylabel('Stress Level')
    plt.legend(title='Support Level')
    plt.tight_layout()
    plt.savefig('employment_support_interaction.png')
    plt.close()
    print("Saved employment_support_interaction.png")

# Gender × Activity on Financial Stress
def plot_gender_activity_financial(data):
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Gender', y='Financial_Stress', hue='Activity_Level',
                       data=data.assign(Gender=lambda x: x['Gender'].map(
                           {0:'Male',1:'Female',2:'Non-Binary'})),
                       palette='Set3', split=True)
    plt.title('Financial Stress by Gender and Activity Level')
    plt.xlabel('Gender')
    plt.ylabel('Financial Stress')
    plt.legend(title='Activity Level')
    plt.tight_layout()
    plt.savefig('gender_activity_financial.png')
    plt.close()
    print("Saved gender_activity_financial.png")

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("=== Mental Health Data Analysis ===")
    
    # Set visual style
    set_visual_style()
    
    # Load and clean data
    data_file = 'Test_anxiety_depression_data.csv'
    df = load_and_clean(data_file)
    
    # Run analyses
    gender_counts, emp_counts = describe_data(df)
    key_corr = stratified_analysis(df)
    
    # Generate correlation matrices
    plot_correlation_matrices(df)
    
    # Perform regression analysis
    regression_results = perform_regression(df)
    
    # Generate all individual visualizations
    plot_sleep_depression_age(df)
    plot_activity_stress_employment(df)
    plot_support_financial_gender(df)
    plot_lifestyle_depression_corr(df)
    plot_sleep_activity_interaction(df)
    plot_employment_support_interaction(df)
    plot_gender_activity_financial(df)
    
    print("\n=== Analysis Complete ===")
    print("All visualizations saved as individual PNG files:")
    print("1. pearson_correlation_matrix.png")
    print("2. spearman_correlation_matrix.png")
    print("3. sleep_depression_age.png")
    print("4. activity_stress_employment.png")
    print("5. support_financial_gender.png")
    print("6. lifestyle_depression_corr.png")
    print("7. sleep_activity_interaction.png")
    print("8. employment_support_interaction.png")
    print("9. gender_activity_financial.png")