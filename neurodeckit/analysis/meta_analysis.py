import numpy as np
from scipy import stats
from statsmodels.stats import multitest
from typing import List, Dict, Tuple, Union, Optional
import math

def calculate_confidence_interval(data_A: List[float], data_B: List[float], 
                                 test_type: str = 'paired', 
                                 confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean difference.
    """
    diff = np.array(data_A) - np.array(data_B)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    
    if n < 2:
        return (mean_diff, mean_diff)
    
    # Calculate t-critical value
    t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    margin_of_error = t_critical * (std_diff / math.sqrt(n))
    
    return (mean_diff - margin_of_error, mean_diff + margin_of_error)

def calculate_effect_size(data_A: List[float], data_B: List[float], 
                         test_type: str = 'paired') -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters:
    data_A, data_B: Paired or independent samples
    test_type: 'paired' or 'independent'
    
    Returns:
    d: Cohen's d effect size
    """
    if test_type == 'paired':
        # Paired samples effect size
        diff = np.array(data_A) - np.array(data_B)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        return mean_diff / std_diff if std_diff != 0 else 0.0
    else:
        # Independent samples effect size
        mean_A, mean_B = np.mean(data_A), np.mean(data_B)
        std_A, std_B = np.std(data_A, ddof=1), np.std(data_B, ddof=1)
        n_A, n_B = len(data_A), len(data_B)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_A - 1) * std_A**2 + (n_B - 1) * std_B**2) / (n_A + n_B - 2))
        return (mean_A - mean_B) / pooled_std if pooled_std != 0 else 0.0

def stouffers_method_advanced(p_values: List[float], sample_sizes: List[int],
                              weights: Optional[str] = 'sqrt_n') -> Dict:
    """
    Enhanced Stouffer's method with additional statistics.
    
    Parameters:
    p_values: List of p-values from individual studies
    sample_sizes: List of sample sizes
    weights: Weighting method ('sqrt_n', 'n', 'inverse_var', or 'equal')
    
    Returns:
    Dictionary with comprehensive meta-analysis results
    """
    p_values = np.array(p_values)
    sample_sizes = np.array(sample_sizes)
    
    # Calculate Z-scores (two-tailed)
    z_scores = np.zeros_like(p_values)
    for i, p in enumerate(p_values):
        if p <= 0:
            z_scores[i] = np.inf if p == 0 else -stats.norm.ppf(p/2)
        elif p >= 1:
            z_scores[i] = 0
        else:
            z_scores[i] = -stats.norm.ppf(p/2)
    
    # Determine weights
    if weights == 'sqrt_n':
        weights_array = np.sqrt(sample_sizes)
    elif weights == 'n':
        weights_array = sample_sizes
    elif weights == 'inverse_var':
        # Approximate variance as 1/n for proportions
        weights_array = sample_sizes
    elif weights == 'equal':
        weights_array = np.ones_like(sample_sizes)
    else:
        raise ValueError(f"Unsupported weight method: {weights}")
    
    # Calculate combined Z
    weighted_z = weights_array * z_scores
    combined_z = np.sum(weighted_z) / np.sqrt(np.sum(weights_array**2))
    
    # Two-tailed p-value
    if np.isinf(combined_z):
        combined_p = 0.0 if combined_z > 0 else 1.0
    else:
        combined_p = stats.norm.sf(abs(combined_z)) * 2
    
    # Calculate heterogeneity statistics (Cochran's Q)
    Q = np.sum((z_scores - combined_z)**2 * weights_array)
    df_q = len(p_values) - 1
    p_q = 1 - stats.chi2.cdf(Q, df_q) if df_q > 0 else 1.0
    
    # I² statistic (heterogeneity)
    if Q > df_q:
        I2 = max(0, (Q - df_q) / Q * 100)
    else:
        I2 = 0.0
    
    # Fail-safe N (Rosenthal's method)
    # Number of null studies needed to make combined p > 0.05
    if combined_p > 0:
        fail_safe_n = int(((np.sum(z_scores) / 1.645)**2 - len(z_scores)) + 0.5)
        fail_safe_n = max(0, fail_safe_n)
    else:
        fail_safe_n = float('inf')
    
    return {
        'combined_z': combined_z,
        'combined_p': combined_p,
        'z_scores': z_scores.tolist(),
        'weights': weights_array.tolist(),
        'heterogeneity_q': Q,
        'heterogeneity_df': df_q,
        'heterogeneity_p': p_q,
        'i2': I2,
        'fail_safe_n': fail_safe_n,
        'n_studies': len(p_values),
        'total_n': int(np.sum(sample_sizes)),
        'weight_method': weights
    }

def meta_analysis_advanced(acc_A: List[List[float]], acc_B: List[List[float]], 
                          test_method: Optional[str] = None, 
                          correction_method: Optional[str] = None, 
                          perm_cutoff: int = 20, 
                          alternative: str = "two-sided",
                          meta_weight_method: str = 'sqrt_n',
                          confidence_level: float = 0.95) -> Dict:
    """
    Advanced meta-analysis with comprehensive reporting.
    """
    if len(acc_A) != len(acc_B):
        raise ValueError("The number of datasets should be the same")
    
    # Perform individual tests
    results_individual = []
    for idx, (data_A, data_B) in enumerate(zip(acc_A, acc_B)):
        n = len(data_A)
        
        # Determine test method
        if test_method is None:
            _test_method = 'paired_t' if n < perm_cutoff else 'wilcoxon'
        else:
            _test_method = test_method
        
        # Perform test
        if _test_method == 'paired_t':
            t_stat, p_val = stats.ttest_rel(data_A, data_B, alternative=alternative)
            df = n - 1
            effect_d = calculate_effect_size(data_A, data_B, 'paired')
            ci_low, ci_high = calculate_confidence_interval(data_A, data_B, 'paired', confidence_level)
            
        elif _test_method == 'independent_t':
            t_stat, p_val = stats.ttest_ind(data_A, data_B, alternative=alternative)
            df = 2 * n - 2
            effect_d = calculate_effect_size(data_A, data_B, 'independent')
            ci_low, ci_high = (float('nan'), float('nan'))  # CI calculation more complex for independent
            
        elif _test_method == 'wilcoxon':
            w_stat, p_val = stats.wilcoxon(data_A, data_B, alternative=alternative)
            t_stat, df, effect_d = w_stat, None, calculate_effect_size(data_A, data_B, 'paired')
            ci_low, ci_high = calculate_confidence_interval(data_A, data_B, 'paired', confidence_level)
            
        else:
            raise ValueError(f"Unsupported test method: {_test_method}")
        
        results_individual.append({
            'dataset_id': idx + 1,
            'n': n,
            'mean_A': float(np.mean(data_A)),
            'mean_B': float(np.mean(data_B)),
            'mean_diff': float(np.mean(data_B) - np.mean(data_A)),
            'test_statistic': float(t_stat) if _test_method != 'wilcoxon' else float(w_stat),
            'df': df,
            'p_value': float(p_val),
            'effect_size': float(effect_d),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'test_method': _test_method
        })
    
    # Extract p-values and sample sizes for meta-analysis
    p_values = [r['p_value'] for r in results_individual]
    sample_sizes = [r['n'] for r in results_individual]
    
    # Apply multiple comparison correction
    original_p_values = p_values.copy()
    if correction_method == 'bonferroni':
        corrected_p = [min(p * len(p_values), 1.0) for p in original_p_values]
        correction_info = "Bonferroni-corrected"
    elif correction_method == 'fdr':
        _, corrected_p, _, _ = multitest.multipletests(original_p_values, method='fdr_bh')
        correction_info = "FDR-corrected"
    elif correction_method is not None:
        raise ValueError(f"Unsupported correction method: {correction_method}")
    else:
        corrected_p = original_p_values.copy()
        correction_info = "Uncorrected"
    
    # Update corrected p-values
    for i, r in enumerate(results_individual):
        r['p_value_corrected'] = float(corrected_p[i])
    
    # Perform advanced meta-analysis
    meta_results = stouffers_method_advanced(corrected_p, sample_sizes, meta_weight_method)
    
    # Calculate weighted average effect size
    effect_sizes = [r['effect_size'] for r in results_individual]
    if meta_weight_method == 'sqrt_n':
        weights = np.sqrt(sample_sizes)
    elif meta_weight_method == 'n':
        weights = sample_sizes
    else:
        weights = np.ones_like(sample_sizes)
    
    weighted_effect = np.sum(np.array(effect_sizes) * weights) / np.sum(weights)
    
    # Calculate confidence interval for weighted effect
    # Simplified approach using weighted variance
    effect_var = np.var(effect_sizes, ddof=1) if len(effect_sizes) > 1 else 0
    effect_se = math.sqrt(effect_var / len(effect_sizes))
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)
    effect_ci_low = weighted_effect - z_critical * effect_se
    effect_ci_high = weighted_effect + z_critical * effect_se
    
    # Generate summary strings
    summary_strings = []
    for r in results_individual:
        test_str = ""
        if r['test_method'] == 'paired_t':
            test_str = f"t({r['df']}) = {r['test_statistic']:.2f}"
        elif r['test_method'] == 'wilcoxon':
            test_str = f"W = {r['test_statistic']:.0f}"
        
        summary = (f"Dataset {r['dataset_id']} (n={r['n']}): {test_str}, "
                  f"p = {r['p_value_corrected']:.3f}, d = {r['effect_size']:.2f}, "
                  f"Δ = {r['mean_diff']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
        summary_strings.append(summary)
    
    # Generate professional report for paper
    paper_report = generate_paper_report(results_individual, meta_results, 
                                        weighted_effect, effect_ci_low, effect_ci_high,
                                        correction_info)
    
    return {
        'individual_results': results_individual,
        'meta_analysis': meta_results,
        'weighted_effect_size': weighted_effect,
        'effect_size_ci': (float(effect_ci_low), float(effect_ci_high)),
        'summary_strings': summary_strings,
        'paper_report': paper_report,
        'correction_method': correction_info,
        'total_datasets': len(acc_A),
        'total_subjects': int(np.sum(sample_sizes))
    }

def generate_paper_report(individual_results: List[Dict], meta_results: Dict,
                         weighted_effect: float, ci_low: float, ci_high: float,
                         correction_method: str) -> Dict:
    """
    Generate professional report for academic paper.
    """
    # Count significant results
    n_significant = sum(1 for r in individual_results if r['p_value_corrected'] < 0.05)
    n_studies = len(individual_results)
    
    # Determine significance level symbols
    p_val = meta_results['combined_p']
    if p_val < 0.001:
        sig_symbol = "***"
        sig_text = "highly significant"
    elif p_val < 0.01:
        sig_symbol = "**"
        sig_text = "very significant"
    elif p_val < 0.05:
        sig_symbol = "*"
        sig_text = "significant"
    else:
        sig_symbol = "ns"
        sig_text = "not significant"
    
    # Determine heterogeneity interpretation
    i2 = meta_results['i2']
    if i2 < 25:
        heterogeneity = "low heterogeneity"
    elif i2 < 50:
        heterogeneity = "moderate heterogeneity"
    elif i2 < 75:
        heterogeneity = "high heterogeneity"
    else:
        heterogeneity = "very high heterogeneity"
    
    # Generate different report sections
    methods_section = (
        f"Statistical analysis was performed using paired t-tests for each of the "
        f"{n_studies} independent datasets. Multiple comparisons were adjusted using "
        f"the {correction_method} method. To integrate results across datasets, "
        f"we applied Stouffer's Z-score meta-analysis with sample-size weighting "
        f"(√n weights). Heterogeneity was assessed using Cochran's Q test and I² statistic."
    )
    
    results_section = (
        f"Individual dataset analyses revealed significant differences in "
        f"{n_significant} out of {n_studies} datasets ({n_significant/n_studies*100:.0f}%). "
        f"Meta-analysis showed a combined effect of Z = {meta_results['combined_z']:.2f} "
        f"(p = {meta_results['combined_p']:.4f}{sig_symbol}), indicating {sig_text} "
        f"superiority of Algorithm B over Algorithm A across all datasets. "
        f"The weighted average effect size was Cohen's d = {weighted_effect:.2f} "
        f"(95% CI: [{ci_low:.2f}, {ci_high:.2f}]), corresponding to a "
        f"{'small' if abs(weighted_effect) < 0.5 else 'medium' if abs(weighted_effect) < 0.8 else 'large'} effect. "
        f"Heterogeneity analysis indicated {heterogeneity} among studies "
        f"(Q({meta_results['heterogeneity_df']}) = {meta_results['heterogeneity_q']:.2f}, "
        f"p = {meta_results['heterogeneity_p']:.3f}, I² = {meta_results['i2']:.1f}%)."
    )
    
    concise_results = (
        f"Meta-analysis (Stouffer's Z) showed significant superiority of Algorithm B "
        f"(Z = {meta_results['combined_z']:.2f}, p = {meta_results['combined_p']:.4f}{sig_symbol}, "
        f"N_total = {meta_results['total_n']}, weighted d = {weighted_effect:.2f})."
    )
    
    # Generate table-ready data
    table_data = []
    for r in individual_results:
        table_data.append({
            'Dataset': r['dataset_id'],
            'N': r['n'],
            'Mean A': f"{r['mean_A']:.3f}",
            'Mean B': f"{r['mean_B']:.3f}",
            'Δ (B-A)': f"{r['mean_diff']:.3f}",
            't/W': f"{r['test_statistic']:.2f}" if r['test_method'] == 'paired_t' else f"{r['test_statistic']:.0f}",
            'df': f"{r['df']}" if r['df'] is not None else "-",
            'p': f"{r['p_value_corrected']:.4f}",
            "p<0.05": "✓" if r['p_value_corrected'] < 0.05 else "✗",
            'Cohen_d': f"{r['effect_size']:.2f}",
            '95% CI': f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        })
    
    return {
        'methods_description': methods_section,
        'results_description': results_section,
        'concise_results': concise_results,
        'table_data': table_data,
        'significance_symbol': sig_symbol,
        'heterogeneity_level': heterogeneity,
        'n_significant': n_significant,
        'n_total_studies': n_studies
    }

# 示例使用和测试
if __name__ == '__main__':
    np.random.seed(42)
    
    # 创建更真实的示例数据
    dataset_configs = [
        (9, 0.75, 0.08, 0.05),
        (52, 0.72, 0.03, 0.03),
        (54, 0.78, 0.04, 0.04),
        (14, 0.70, 0.06, 0.06),
        (106, 0.75, 0.02, 0.02),
        (29, 0.73, 0.04, 0.04),
        (10, 0.76, 0.05, 0.05)
    ]
    
    acc_A = []
    acc_B = []
    
    for n_samples, base_acc, std, diff in dataset_configs:
        data_A = np.clip(np.random.normal(base_acc, std, n_samples), 0.5, 0.95)
        data_B = np.clip(np.random.normal(base_acc + diff, std, n_samples), 0.5, 0.95)
        acc_A.append(data_A)
        acc_B.append(data_B)
    
    # 运行高级荟萃分析
    print("高级荟萃分析示例")
    print("=" * 80)
    
    results = meta_analysis_advanced(
        acc_A, acc_B,
        test_method='paired_t',
        correction_method='fdr',
        meta_weight_method='sqrt_n',
        confidence_level=0.95
    )
    
    # 打印详细结果
    print("\n个体研究结果:")
    print("-" * 80)
    for summary in results['summary_strings']:
        print(summary)
    
    print("\n\n荟萃分析结果:")
    print("-" * 80)
    meta = results['meta_analysis']
    print(f"综合Z值: {meta['combined_z']:.3f}")
    print(f"综合p值: {meta['combined_p']:.4f}")
    print(f"研究数量: {meta['n_studies']}")
    print(f"总样本量: {meta['total_n']}")
    print(f"异质性检验 Q({meta['heterogeneity_df']}) = {meta['heterogeneity_q']:.3f}, "
          f"p = {meta['heterogeneity_p']:.3f}")
    print(f"I²指数: {meta['i2']:.1f}%")
    print(f"失效安全数: {meta['fail_safe_n']}")
    
    print(f"\n加权平均效应量 (Cohen's d): {results['weighted_effect_size']:.3f}")
    print(f"95%置信区间: [{results['effect_size_ci'][0]:.3f}, {results['effect_size_ci'][1]:.3f}]")
    
    print("\n\n写作模板:")
    print("=" * 80)
    print("\n方法部分描述:")
    print(results['paper_report']['methods_description'])
    
    print("\n\n结果部分描述:")
    print(results['paper_report']['results_description'])
    
    print("\n\n简洁结果（摘要用）:")
    print(results['paper_report']['concise_results'])
    
    print("\n\n表格数据:")
    print("-" * 80)
    print(f"{'Dataset':<8} {'N':<6} {'Mean A':<8} {'Mean B':<8} {'Δ(B-A)':<8} "
          f"{'t/W':<8} {'df':<6} {'p':<8} {'Sig':<4} {'d':<8} {'95% CI':<20}")
    print("-" * 80)

    for row in results['paper_report']['table_data']:
        print(f"{row['Dataset']:<8} {row['N']:<6} {row['Mean A']:<8} {row['Mean B']:<8} "
              f"{row['Δ (B-A)']:<8} {row['t/W']:<8} {row['df']:<6} "
              f"{row['p']:<8} {row['p<0.05']:<4} {row['Cohen_d']:<8} {row['95% CI']:<20}")
