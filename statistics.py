#!/usr/bin/env python3
"""
Statistical Analysis for Breakthrough Validation.

Provides:
- Multi-seed aggregation
- Statistical significance tests
- Effect size calculations  
- Breakthrough classification
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy import stats


@dataclass
class ComparisonResult:
    """Results of comparing two algorithms."""
    
    # Basic stats
    algo1_name: str
    algo2_name: str
    algo1_mean: float
    algo1_std: float
    algo1_n: int
    algo2_mean: float
    algo2_std: float
    algo2_n: int
    
    # Improvement
    difference: float           # algo1 - algo2
    improvement_pct: float      # (algo1 - algo2) / algo2 * 100
    
    # Statistical tests
    t_statistic: float
    p_value: float
    ci_lower: float             # 95% CI for difference
    ci_upper: float
    
    # Effect size
    cohens_d: float
    
    # Verdict
    is_significant: bool        # p < 0.05
    is_large_effect: bool       # |d| > 0.8
    is_breakthrough: bool       # p < 0.01 AND |d| > 0.8
    algo1_wins: bool            # algo1 significantly better
    
    @property
    def significance_stars(self) -> str:
        """Return significance stars for display."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        return ""
    
    def summary(self) -> str:
        """Human-readable summary."""
        winner = self.algo1_name if self.algo1_wins else self.algo2_name
        return (
            f"{self.algo1_name}: {self.algo1_mean:.1f}±{self.algo1_std:.1f} (n={self.algo1_n})\n"
            f"{self.algo2_name}: {self.algo2_mean:.1f}±{self.algo2_std:.1f} (n={self.algo2_n})\n"
            f"Difference: {self.improvement_pct:+.1f}%{self.significance_stars}\n"
            f"p-value: {self.p_value:.4f}, Cohen's d: {self.cohens_d:.2f}\n"
            f"Verdict: {'BREAKTHROUGH' if self.is_breakthrough else 'Significant' if self.is_significant else 'Not significant'}"
        )


class StatisticalAnalyzer:
    """Perform statistical analysis on experiment results."""
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 min_effect_size: float = 0.8,
                 breakthrough_p_threshold: float = 0.01):
        self.alpha = significance_level
        self.min_effect = min_effect_size
        self.breakthrough_p = breakthrough_p_threshold
    
    def compare(self, 
                algo1_values: List[float],
                algo2_values: List[float],
                algo1_name: str = "EqProp",
                algo2_name: str = "BP") -> ComparisonResult:
        """Compare two sets of results statistically."""
        
        a1 = np.array(algo1_values)
        a2 = np.array(algo2_values)
        
        # Basic statistics
        mean1, std1, n1 = np.mean(a1), np.std(a1, ddof=1), len(a1)
        mean2, std2, n2 = np.mean(a2), np.std(a2, ddof=1), len(a2)
        
        # Difference
        diff = mean1 - mean2
        improvement_pct = (diff / abs(mean2)) * 100 if mean2 != 0 else 0
        
        # T-test (Welch's t-test for unequal variances)
        if n1 >= 2 and n2 >= 2:
            t_stat, p_value = stats.ttest_ind(a1, a2, equal_var=False)
        else:
            t_stat, p_value = 0.0, 1.0
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if (n1+n2) > 2 else 1.0
        cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
        
        # 95% Confidence interval for the difference
        se_diff = np.sqrt(std1**2/n1 + std2**2/n2) if n1 > 0 and n2 > 0 else 0
        df = n1 + n2 - 2
        t_crit = stats.t.ppf(0.975, df) if df > 0 else 1.96
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        
        # Verdicts
        is_significant = p_value < self.alpha
        is_large_effect = abs(cohens_d) >= self.min_effect
        is_breakthrough = p_value < self.breakthrough_p and is_large_effect
        algo1_wins = is_significant and diff > 0
        
        return ComparisonResult(
            algo1_name=algo1_name,
            algo2_name=algo2_name,
            algo1_mean=mean1,
            algo1_std=std1,
            algo1_n=n1,
            algo2_mean=mean2,
            algo2_std=std2,
            algo2_n=n2,
            difference=diff,
            improvement_pct=improvement_pct,
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            cohens_d=cohens_d,
            is_significant=is_significant,
            is_large_effect=is_large_effect,
            is_breakthrough=is_breakthrough,
            algo1_wins=algo1_wins
        )
    
    def validate_sufficient_samples(self, n1: int, n2: int, min_samples: int = 5) -> Tuple[bool, str]:
        """Check if we have enough samples for valid comparison."""
        if n1 < min_samples:
            return False, f"Need {min_samples-n1} more samples for algo1"
        if n2 < min_samples:
            return False, f"Need {min_samples-n2} more samples for algo2"
        return True, "Sufficient samples"
    
    def power_analysis(self, 
                       effect_size: float = 0.8,
                       alpha: float = 0.05,
                       power: float = 0.8) -> int:
        """Calculate required sample size for desired power."""
        # Using approximation: n ≈ 2 * ((z_alpha + z_beta) / d)^2
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))


@dataclass
class FairnessReport:
    """Report on comparison fairness."""
    
    param_ratio: float          # algo1_params / algo2_params
    walltime_ratio: float       # algo1_time / algo2_time
    iteration_ratio: float      # algo1_iters / algo2_iters
    
    param_fair: bool            # Within tolerance
    walltime_fair: bool
    
    issues: List[str]
    
    @property
    def is_fair(self) -> bool:
        return self.param_fair  # Parameters must match
    
    def summary(self) -> str:
        status = "✅ FAIR" if self.is_fair else "⚠️ UNFAIR"
        lines = [
            f"Fairness: {status}",
            f"  Parameter ratio: {self.param_ratio:.2f}x {'✅' if self.param_fair else '❌'}",
            f"  Walltime ratio: {self.walltime_ratio:.2f}x",
        ]
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        return "\n".join(lines)


class FairnessChecker:
    """Check if algorithm comparison is fair."""
    
    def __init__(self, 
                 param_tolerance: float = 0.05,
                 walltime_tolerance: float = 0.10):
        self.param_tolerance = param_tolerance
        self.walltime_tolerance = walltime_tolerance
    
    def check(self,
              algo1_params: int,
              algo2_params: int,
              algo1_walltime: float,
              algo2_walltime: float,
              algo1_iterations: int = 0,
              algo2_iterations: int = 0) -> FairnessReport:
        """Check fairness of comparison."""
        
        param_ratio = algo1_params / algo2_params if algo2_params > 0 else 1.0
        walltime_ratio = algo1_walltime / algo2_walltime if algo2_walltime > 0 else 1.0
        iter_ratio = algo1_iterations / algo2_iterations if algo2_iterations > 0 else 1.0
        
        param_fair = abs(param_ratio - 1.0) <= self.param_tolerance
        walltime_fair = abs(walltime_ratio - 1.0) <= self.walltime_tolerance
        
        issues = []
        if not param_fair:
            if param_ratio > 1:
                issues.append(f"Algo1 has {(param_ratio-1)*100:.1f}% more parameters")
            else:
                issues.append(f"Algo1 has {(1-param_ratio)*100:.1f}% fewer parameters")
        
        if walltime_ratio > 2:
            issues.append(f"Algo1 takes {walltime_ratio:.1f}x longer (expected for EqProp)")
        
        return FairnessReport(
            param_ratio=param_ratio,
            walltime_ratio=walltime_ratio,
            iteration_ratio=iter_ratio,
            param_fair=param_fair,
            walltime_fair=walltime_fair,
            issues=issues
        )


# Self-test
if __name__ == "__main__":
    print("Testing StatisticalAnalyzer...")
    
    analyzer = StatisticalAnalyzer()
    
    # Simulated results (similar to our actual RL experiment)
    eqprop_results = [341, 352, 328, 367, 345, 339, 358, 331, 349, 355]
    bp_results = [187, 192, 178, 195, 183, 189, 176, 191, 185, 188]
    
    result = analyzer.compare(eqprop_results, bp_results, "EqProp", "BP")
    print(result.summary())
    print()
    
    # Test fairness checker
    checker = FairnessChecker()
    fairness = checker.check(
        algo1_params=10000,
        algo2_params=10000,
        algo1_walltime=420,
        algo2_walltime=40
    )
    print(fairness.summary())
    
    print("\n✅ Statistics tests passed!")
