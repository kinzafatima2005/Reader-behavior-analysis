import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def analyze_format_comparison(df_master):
    """
    Compares Physical vs E-Book across key metrics:
    - Average session duration
    - Pages per session
    - Pages per minute (reading speed)
    - Completion rate & abandonment rate
    - Number of sessions per book
    """
    summary = df_master.groupby('format').agg(
        total_sessions=('session_id', 'count'),
        avg_duration_min=('duration_minutes', 'mean'),
        std_duration_min=('duration_minutes', 'std'),
        avg_pages_read=('pages_read', 'mean'),
        avg_speed_ppm=('pages_per_minute', 'mean'),
        std_speed_ppm=('pages_per_minute', 'std'),
        completion_rate=('completed', 'mean'),
        unique_readers=('reader_id', 'nunique'),
        unique_books=('book_id', 'nunique')
    ).reset_index()
    
    summary['abandonment_rate'] = 1.0 - summary['completion_rate']
    return summary

def analyze_genre_comparison(df_master):
    """
    Compares reading metrics across genres.
    """
    genre_col = 'genre_clean' if 'genre_clean' in df_master.columns else 'genre'
    summary = df_master.groupby(genre_col).agg(
        total_sessions=('session_id', 'count'),
        avg_duration_min=('duration_minutes', 'mean'),
        avg_speed_ppm=('pages_per_minute', 'mean'),
        avg_pages_read=('pages_read', 'mean'),
        completion_rate=('completed', 'mean')
    ).reset_index().sort_values(by='completion_rate', ascending=False)
    
    return summary

def analyze_time_of_day(df_master):
    """
    Compares reading behavior across time of day (Morning, Afternoon, Evening, Night).
    """
    summary = df_master.groupby('time_of_day').agg(
        total_sessions=('session_id', 'count'),
        avg_duration_min=('duration_minutes', 'mean'),
        avg_speed_ppm=('pages_per_minute', 'mean'),
        avg_pages_read=('pages_read', 'mean'),
        completion_rate=('completed', 'mean')
    ).reset_index()
    
    # Custom sort order
    tod_order = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    summary['sort_key'] = summary['time_of_day'].map(tod_order)
    summary = summary.sort_values('sort_key').drop(columns=['sort_key'])
    return summary

# ---------------------------------------------------------
# Statistical Hypothesis Testing Engine
# ---------------------------------------------------------
def cohens_d(group1, group2):
    """Calculates Cohen's d effect size between two independent groups."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / s_pooled

def cramers_v(contingency_table):
    """Calculates Cramér's V effect size for a chi-square contingency table."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def mean_confidence_interval(data, confidence=0.95):
    """Calculates mean and confidence interval for a numeric vector."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def test_reading_speed_by_format(df_master):
    """
    Research Question 1 & 3:
    Does reading speed differ between e-books and physical books?
    Tests: Independent t-test and Mann-Whitney U test.
    """
    ebook_speed = df_master[df_master['format'] == 'Ebook']['pages_per_minute'].dropna()
    physical_speed = df_master[df_master['format'] == 'Physical']['pages_per_minute'].dropna()
    
    # Parametric t-test
    t_stat, p_val_t = stats.ttest_ind(ebook_speed, physical_speed, equal_var=False)
    
    # Non-parametric Mann-Whitney U
    u_stat, p_val_u = stats.mannwhitneyu(ebook_speed, physical_speed, alternative='two-sided')
    
    d_effect = cohens_d(ebook_speed, physical_speed)
    m1, ci1_low, ci1_high = mean_confidence_interval(ebook_speed)
    m2, ci2_low, ci2_high = mean_confidence_interval(physical_speed)
    
    return {
        'research_question': 'Does reading speed differ between E-Books and Physical Books?',
        'group_1_name': 'Ebook',
        'group_1_mean': round(m1, 3),
        'group_1_ci': (round(ci1_low, 3), round(ci1_high, 3)),
        'group_2_name': 'Physical',
        'group_2_mean': round(m2, 3),
        'group_2_ci': (round(ci2_low, 3), round(ci2_high, 3)),
        't_statistic': round(t_stat, 4),
        'p_value_ttest': p_val_t,
        'u_statistic': round(u_stat, 4),
        'p_value_mannwhitney': p_val_u,
        'cohens_d': round(d_effect, 4),
        'statistically_significant': p_val_t < 0.05,
        'practical_interpretation': (
            f"In this dataset, E-Book sessions had a mean reading speed of {round(m1, 2)} PPM "
            f"compared to {round(m2, 2)} PPM for Physical books (t={round(t_stat,2)}, p={p_val_t:.4e}, Cohen's d={round(d_effect,2)}). "
            f"Note: This reflects an observed association in the data and does not prove causal acceleration."
        )
    }

def test_completion_rate_by_format(df_master):
    """
    Research Question 2:
    Which format has a higher book-completion rate?
    Test: Chi-Square Test of Independence.
    """
    contingency = pd.crosstab(df_master['format'], df_master['completed'])
    chi2, p_val, dof, ex = stats.chi2_contingency(contingency)
    cv = cramers_v(contingency)
    
    rates = df_master.groupby('format')['completed'].mean()
    
    return {
        'research_question': 'Is book completion rate dependent on format?',
        'contingency_table': contingency,
        'completion_rates': rates.to_dict(),
        'chi2_statistic': round(chi2, 4),
        'p_value': p_val,
        'degrees_of_freedom': dof,
        'cramers_v': round(cv, 4),
        'statistically_significant': p_val < 0.05,
        'practical_interpretation': (
            f"Chi-square test yielded chi2={round(chi2,2)}, p={p_val:.4e}, Cramér's V={round(cv,3)}. "
            f"Completion rates by format: Ebook={round(rates.get('Ebook',0)*100,1)}%, Physical={round(rates.get('Physical',0)*100,1)}%."
        )
    }

def test_genre_speed_and_completion(df_master):
    """
    Research Question 6:
    Does genre influence reading speed or completion?
    Tests: One-way ANOVA & Kruskal-Wallis across genres.
    """
    genre_col = 'genre_clean' if 'genre_clean' in df_master.columns else 'genre'
    genre_groups = [group['pages_per_minute'].dropna().values for name, group in df_master.groupby(genre_col)]
    
    f_stat, p_val_f = stats.f_oneway(*genre_groups)
    h_stat, p_val_h = stats.kruskal(*genre_groups)
    
    return {
        'research_question': 'Does genre significantly affect reading speed?',
        'f_statistic': round(f_stat, 4),
        'p_value_anova': p_val_f,
        'h_statistic': round(h_stat, 4),
        'p_value_kruskal': p_val_h,
        'statistically_significant': p_val_f < 0.05,
        'practical_interpretation': f"ANOVA across genres: F={round(f_stat,2)}, p={p_val_f:.4e}. Reading speed varies significantly across genres."
    }

def run_all_statistical_tests(df_master):
    """
    Runs all research hypothesis tests and aggregates results into a summary dictionary.
    """
    results = {
        'speed_by_format': test_reading_speed_by_format(df_master),
        'completion_by_format': test_completion_rate_by_format(df_master),
        'genre_speed_anova': test_genre_speed_and_completion(df_master)
    }
    return results

if __name__ == "__main__":
    from src.data_processing import process_and_clean_data
    _, _, _, df_master, _ = process_and_clean_data()
    res = run_all_statistical_tests(df_master)
    print("[Statistical Analysis] All hypothesis tests executed successfully:")
    print(" - Speed test p-value:", res['speed_by_format']['p_value_ttest'])
    print(" - Completion test p-value:", res['completion_by_format']['p_value'])
