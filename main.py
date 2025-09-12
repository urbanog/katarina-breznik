import pandas as pd
from IPython.display import display
from scipy.stats import f_oneway
from scipy.stats import shapiro
import pingouin as pg
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as stats  # Import scipy.stats


# Path to your Excel file
excel_path = 'statistike.xlsx'

# Če želite uvoziti dodatne podatke iz drugega Excel lista, uporabite naslednje:
# Primer: Branje lista "statistike vseh obravnav pacien" iz iste datoteke
df_all = pd.read_excel(excel_path, sheet_name="Statistike vseh obravnav pacien")
print(df_all.head())

pivot_df = df_all.pivot_table(
    index=['ID'],
    columns='način',
    values=[
        'Kanal 1 Povprečje delovanja',
        'Kanal 2 Povprečje delovanja'
    ],
    aggfunc='first'
)
print(pivot_df.head())

def calculate_rms(group):
    """Calculate RMS for each column in the group."""
    rms_values = {}
    for column in group.columns:
        # Ensure the column exists before attempting to calculate RMS
        if column in group.columns:
            # Calculate the squared values
            squared_values = group[column] ** 2
            # Calculate the mean of the squared values
            mean_squared = squared_values.mean()
            # Calculate the square root of the mean squared values
            rms = np.sqrt(mean_squared)
            rms_values[column] = rms
        else:
            rms_values[column] = None  # Or np.nan, or any other placeholder
    return pd.Series(rms_values)

# Group by 'način' and calculate RMS for 'Kanal 1 Povprečje delovanja' and 'Kanal 2 Povprečje delovanja'
rms_by_nacin = df_all.groupby('način')[['Kanal 1 Povprečje delovanja', 'Kanal 2 Povprečje delovanja']].apply(calculate_rms)

print("\nRMS po načinu:")
print(rms_by_nacin)

# Preveri normalnost porazdelitve za vsako skupino v 'način' za Kanal 1 in Kanal 2
for kanal in ['Kanal 1 Povprečje delovanja', 'Kanal 2 Povprečje delovanja']:
    print(f"\nNormalnost za {kanal}:")
    for name, group in df_all.groupby('način'):
        stat, p = shapiro(group[kanal].dropna())
        print(f"Shapiro-Wilk test za skupino '{name}': stat={stat:.4f}, p-value={p:.4f}")

# # ANOVA test za Kanal 1
# kanal1_data = []
# for name, group in df_all.groupby('način'):
#     kanal1_data.append(group['Kanal 1 Povprečje delovanja'].dropna())

# # Perform ANOVA test
# f_statistic, p_value = f_oneway(*kanal1_data)

# print("\nANOVA test za Kanal 1:")
# print(f"F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")

# Friedmanov test za Kanal 1
data_kanal1 = pivot_df['Kanal 1 Povprečje delovanja'].dropna()
print("\nFriedmanov test za Kanal 1:")
friedman_test = pg.friedman(data=df_all, dv='Kanal 1 Povprečje delovanja', within='način', subject='ID')
print(friedman_test)

# Friedmanov test za Kanal 2
print("\nFriedmanov test za Kanal 2:")
friedman_test_kanal2 = pg.friedman(data=df_all, dv='Kanal 2 Povprečje delovanja', within='način', subject='ID')
print(friedman_test_kanal2)

# Post-hoc test (Dunn's test with Bonferroni correction) for Kanal 1
print("\nPost-hoc Dunn's test with Bonferroni correction for Kanal 1:")
posthoc_kanal1 = pg.pairwise_tukey(data=df_all, dv='Kanal 1 Povprečje delovanja', between='način')
print(posthoc_kanal1)

# Post-hoc test (Dunn's test with Bonferroni correction) for Kanal 2
print("\nPost-hoc Dunn's test with Bonferroni correction for Kanal 2:")
posthoc_kanal2 = pg.pairwise_tukey(data=df_all, dv='Kanal 2 Povprečje delovanja', between='način')
print(posthoc_kanal2)




# Calculate RMS for specific 'način' values: 'nos', 'ŠŠŠ', and 'ustnična pripora'
selected_nacins = ['nos', 'ŠŠŠ', 'ustnična pripora']
rms_selected = df_all[df_all['način'].isin(selected_nacins)].groupby('način')[['Kanal 1 Povprečje delovanja', 'Kanal 2 Povprečje delovanja']].apply(calculate_rms)

print("\nRMS for selected načins:")
print(rms_selected)


# Normality test for RMS values
print("\nNormality test for RMS values:")
for column in rms_by_nacin.columns:
    stat, p = shapiro(rms_by_nacin[column].dropna())
    print(f"Shapiro-Wilk test for {column}: stat={stat:.4f}, p-value={p:.4f}")

    # Paired t-tests or Wilcoxon signed-rank tests with Bonferroni correction

    alpha = 0.05  # significance level

    # Define pairs to compare
    pairs = [('nos', 'ŠŠŠ'), ('ŠŠŠ', 'ustnična pripora')]

    # Results storage
    ttest_results = {}
    wilcoxon_results = {}

    # Perform tests for Kanal 1 and Kanal 2
    for kanal in ['Kanal 1 Povprečje delovanja', 'Kanal 2 Povprečje delovanja']:
        print(f"\nPaired tests for {kanal}:")
        ttest_results[kanal] = {}
        wilcoxon_results[kanal] = {}
        
        # Collect p-values for Bonferroni correction
        p_values_ttest = []
        p_values_wilcoxon = []
        
        for pair in pairs:
            group1 = df_all[df_all['način'] == pair[0]][kanal].dropna()
            group2 = df_all[df_all['način'] == pair[1]][kanal].dropna()
            
            # Check if there are enough data points for the tests
            if len(group1) < 2 or len(group2) < 2:
                print(f"Warning: Insufficient data for {kanal}, pair {pair}. Skipping test.")
                ttest_results[kanal][pair] = {'statistic': None, 'pvalue': None}
                wilcoxon_results[kanal][pair] = {'statistic': None, 'pvalue': None}
                p_values_ttest.append(1.0)  # Append a neutral p-value for correction
                p_values_wilcoxon.append(1.0)  # Append a neutral p-value for correction
                continue
            
            # Normality test for each group
            normality_group1 = shapiro(group1)[1] > alpha
            normality_group2 = shapiro(group2)[1] > alpha
            
            # Paired t-test if both groups are normally distributed
            if normality_group1 and normality_group2:
                ttest_result = ttest_rel(group1, group2)
                ttest_results[kanal][pair] = {'statistic': ttest_result.statistic, 'pvalue': ttest_result.pvalue}
                p_values_ttest.append(ttest_result.pvalue)
                wilcoxon_results[kanal][pair] = {'statistic': None, 'pvalue': None}  # Set Wilcoxon result to None
            else:
                # Wilcoxon signed-rank test if not normally distributed
                wilcoxon_result = wilcoxon(group1, group2)
                wilcoxon_results[kanal][pair] = {'statistic': wilcoxon_result.statistic, 'pvalue': wilcoxon_result.pvalue}
                p_values_wilcoxon.append(wilcoxon_result.pvalue)
                ttest_results[kanal][pair] = {'statistic': None, 'pvalue': None}  # Set t-test result to None
        
        # Bonferroni correction
        reject_ttest, pvals_corrected_ttest, _, _ = multipletests(p_values_ttest, alpha=alpha, method='bonferroni')
        reject_wilcoxon, pvals_corrected_wilcoxon, _, _ = multipletests(p_values_wilcoxon, alpha=alpha, method='bonferroni')
        
        # Print results with Bonferroni correction
        pair_idx = 0
        for pair in pairs:
            if ttest_results[kanal][pair]['pvalue'] is not None:
                print(f"Paired t-test for {kanal}, pair {pair}: statistic={ttest_results[kanal][pair]['statistic']:.4f}, p-value={pvals_corrected_ttest[pair_idx]:.4f} (Bonferroni corrected)")
            else:
                stat = wilcoxon_results[kanal][pair]['statistic']
                pval = pvals_corrected_wilcoxon[pair_idx]
                stat_str = f"{stat:.4f}" if stat is not None else "NA"
                pval_str = f"{pval:.4f}" if pval is not None else "NA"
                print(f"Wilcoxon signed-rank test for {kanal}, pair {pair}: statistic={stat_str}, p-value={pval_str} (Bonferroni corrected)")
            pair_idx += 1

    # Effect size calculation (Cohen's d for t-tests, r for Wilcoxon)
    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    def wilcoxon_r(z, n):
        """
        Calculates the effect size r for the Wilcoxon signed-rank test.
        """
        r = z / np.sqrt(n)
        return r

    print("\nEffect sizes:")
    for kanal in ['Kanal 1 Povprečje delovanja', 'Kanal 2 Povprečje delovanja']:
        print(f"\nEffect sizes for {kanal}:")
        for pair in pairs:
            group1 = df_all[df_all['način'] == pair[0]][kanal].dropna()
            group2 = df_all[df_all['način'] == pair[1]][kanal].dropna()
            
             # Check if there are enough data points for the tests
            if len(group1) < 2 or len(group2) < 2:
                print(f"Warning: Insufficient data for {kanal}, pair {pair}. Skipping effect size calculation.")
                continue
            
            # Normality test for each group
            normality_group1 = shapiro(group1)[1] > alpha
            normality_group2 = shapiro(group2)[1] > alpha
            
            if normality_group1 and normality_group2:
                # Calculate Cohen's d
                effect_size = cohens_d(group1, group2)
                print(f"Cohen's d for {kanal}, pair {pair}: {effect_size:.4f}")
            else:
                # Calculate r for Wilcoxon
                wilcoxon_result = wilcoxon(group1, group2)
                z_statistic = wilcoxon_result.statistic
                n = len(group1)  # Number of pairs
                effect_size = wilcoxon_r(z_statistic, n)
                print(f"Effect size r for Wilcoxon for {kanal}, pair {pair}: {effect_size:.4f}")