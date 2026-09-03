"""
Notebook 03: Statistical Analysis & Hypothesis Testing
Performs t-tests, Mann-Whitney U tests, Chi-Square independence tests, and ANOVA with effect sizes.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import process_and_clean_data
from src.analysis import run_all_statistical_tests

_, _, _, df_master, _ = process_and_clean_data()
results = run_all_statistical_tests(df_master)

print("==================================================")
print(" RESEARCH QUESTION 1 & 3: READING SPEED BY FORMAT ")
print("==================================================")
speed_res = results['speed_by_format']
print(f"Group 1 ({speed_res['group_1_name']}): Mean = {speed_res['group_1_mean']} PPM, 95% CI = {speed_res['group_1_ci']}")
print(f"Group 2 ({speed_res['group_2_name']}): Mean = {speed_res['group_2_mean']} PPM, 95% CI = {speed_res['group_2_ci']}")
print(f"Welch's t-statistic: {speed_res['t_statistic']}, p-value: {speed_res['p_value_ttest']:.4e}")
print(f"Mann-Whitney U statistic: {speed_res['u_statistic']}, p-value: {speed_res['p_value_mannwhitney']:.4e}")
print(f"Cohen's d effect size: {speed_res['cohens_d']}")
print(f"Interpretation: {speed_res['practical_interpretation']}")

print("\n==================================================")
print(" RESEARCH QUESTION 2: COMPLETION RATE BY FORMAT   ")
print("==================================================")
comp_res = results['completion_by_format']
print("Contingency Table:")
print(comp_res['contingency_table'])
print(f"Chi-square statistic: {comp_res['chi2_statistic']}, p-value: {comp_res['p_value']:.4e}")
print(f"Cramér's V effect size: {comp_res['cramers_v']}")
print(f"Interpretation: {comp_res['practical_interpretation']}")
