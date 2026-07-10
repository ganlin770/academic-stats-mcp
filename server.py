# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.12.2,<2"]
# ///
"""
academic-stats-advisor — an MCP server that lets an AI (ChatGPT / Claude / any
MCP client) DIRECTLY CALL a statistician's decision logic:

  • recommend_test        — pick the correct statistical test from your design
  • check_assumptions     — assumptions of a test, how to check them, what to do if violated
  • interpret_result      — turn a p-value / effect size into a correct, APA-style conclusion
  • plan_sample_size      — a priori power analysis (required n) with a normal approximation
  • normality_guide       — how to decide & report normality the right way
  • list_supported_tests  — everything this server knows

Author: Gan Lin (github.com/ganlin770). Dependency-light (only `mcp`) so it runs
anywhere: `uv run server.py` (stdio) or `MCP_HTTP=1 uv run server.py` (remote HTTP).
Guidance only — always verify assumptions against your own data.
"""
from __future__ import annotations
import math, os, sys
from typing import Literal
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("academic-stats-advisor")

# --------------------------------------------------------------------------- #
# reference tables
# --------------------------------------------------------------------------- #
TESTS: dict[str, dict] = {
    "one_sample_t": {
        "name": "One-sample t-test",
        "spss": "Analyze ▸ Compare Means ▸ One-Sample T Test",
        "r": 't.test(x, mu = <test_value>)',
        "assumptions": ["normality", "independence", "no_outliers"],
        "reporting": "A one-sample t-test showed the mean (M = __, SD = __) differed from <value>, t(df) = __, p = __, Cohen's d = __.",
    },
    "wilcoxon_signed_one": {
        "name": "One-sample Wilcoxon signed-rank test",
        "spss": "Analyze ▸ Nonparametric Tests ▸ One Sample",
        "r": 'wilcox.test(x, mu = <test_value>)',
        "assumptions": ["independence", "symmetry_of_differences"],
        "reporting": "A one-sample Wilcoxon signed-rank test indicated the median differed from <value>, Z = __, p = __, r = __.",
    },
    "independent_t": {
        "name": "Independent-samples t-test (Student)",
        "spss": "Analyze ▸ Compare Means ▸ Independent-Samples T Test (read the 'equal variances assumed' row)",
        "r": 't.test(y ~ group, var.equal = TRUE)',
        "assumptions": ["normality", "equal_variance", "independence", "no_outliers"],
        "reporting": "An independent-samples t-test showed group A (M = __, SD = __) differed from group B (M = __, SD = __), t(df) = __, p = __, Cohen's d = __.",
    },
    "welch_t": {
        "name": "Welch's t-test (unequal variances)",
        "spss": "Analyze ▸ Compare Means ▸ Independent-Samples T Test (read the 'equal variances NOT assumed' row)",
        "r": 't.test(y ~ group)  # var.equal = FALSE is the default',
        "assumptions": ["normality", "independence", "no_outliers"],
        "reporting": "Welch's t-test (equal variances not assumed): t(df_adj) = __, p = __, Cohen's d = __.",
    },
    "mann_whitney": {
        "name": "Mann–Whitney U test",
        "spss": "Analyze ▸ Nonparametric Tests ▸ Independent Samples (2 groups)",
        "r": 'wilcox.test(y ~ group)',
        "assumptions": ["independence", "similar_distribution_shape"],
        "reporting": "A Mann–Whitney U test indicated a difference, U = __, Z = __, p = __, r = __ (report medians/IQR).",
    },
    "paired_t": {
        "name": "Paired-samples t-test",
        "spss": "Analyze ▸ Compare Means ▸ Paired-Samples T Test",
        "r": 't.test(x1, x2, paired = TRUE)',
        "assumptions": ["normality_of_differences", "independence_of_pairs", "no_outliers"],
        "reporting": "A paired-samples t-test showed a change from time 1 (M = __, SD = __) to time 2 (M = __, SD = __), t(df) = __, p = __, Cohen's dz = __.",
    },
    "wilcoxon_signed": {
        "name": "Wilcoxon signed-rank test",
        "spss": "Analyze ▸ Nonparametric Tests ▸ Related Samples (2 conditions)",
        "r": 'wilcox.test(x1, x2, paired = TRUE)',
        "assumptions": ["independence_of_pairs", "symmetry_of_differences"],
        "reporting": "A Wilcoxon signed-rank test indicated a change, Z = __, p = __, r = __ (report medians).",
    },
    "one_way_anova": {
        "name": "One-way ANOVA",
        "spss": "Analyze ▸ Compare Means ▸ One-Way ANOVA (or GLM Univariate); request Levene + Tukey",
        "r": 'aov(y ~ group); then TukeyHSD()',
        "assumptions": ["normality_of_residuals", "equal_variance", "independence", "no_outliers"],
        "reporting": "A one-way ANOVA showed a group effect, F(df1, df2) = __, p = __, η² = __. Tukey HSD post hoc: ...",
    },
    "welch_anova": {
        "name": "Welch's ANOVA (unequal variances)",
        "spss": "One-Way ANOVA ▸ Options ▸ Welch; post hoc: Games-Howell",
        "r": 'oneway.test(y ~ group)  # var.equal = FALSE; post hoc rstatix::games_howell_test',
        "assumptions": ["normality_of_residuals", "independence", "no_outliers"],
        "reporting": "A Welch's ANOVA showed a group effect, F(df1, df2_adj) = __, p = __. Games-Howell post hoc: ...",
    },
    "kruskal_wallis": {
        "name": "Kruskal–Wallis H test",
        "spss": "Analyze ▸ Nonparametric Tests ▸ Independent Samples (>2 groups)",
        "r": 'kruskal.test(y ~ group); post hoc: dunn.test / rstatix::dunn_test',
        "assumptions": ["independence", "similar_distribution_shape"],
        "reporting": "A Kruskal–Wallis test showed a difference, H(df) = __, p = __. Dunn's post hoc (Bonferroni): ...",
    },
    "rm_anova": {
        "name": "Repeated-measures ANOVA",
        "spss": "Analyze ▸ General Linear Model ▸ Repeated Measures (check Mauchly's sphericity)",
        "r": 'ez::ezANOVA or afex::aov_ez',
        "assumptions": ["normality_of_residuals", "sphericity", "independence_of_subjects"],
        "reporting": "A repeated-measures ANOVA (Greenhouse-Geisser corrected if sphericity violated) showed an effect, F(df1, df2) = __, p = __, η²p = __.",
    },
    "friedman": {
        "name": "Friedman test",
        "spss": "Analyze ▸ Nonparametric Tests ▸ Related Samples (>2 conditions)",
        "r": 'friedman.test(y ~ condition | subject); post hoc: pairwise Wilcoxon (Bonferroni)',
        "assumptions": ["independence_of_subjects", "ordinal_or_continuous_outcome"],
        "reporting": "A Friedman test showed a difference across conditions, χ²(df) = __, p = __. Post hoc: pairwise Wilcoxon, Bonferroni.",
    },
    "pearson": {
        "name": "Pearson correlation",
        "spss": "Analyze ▸ Correlate ▸ Bivariate ▸ Pearson",
        "r": 'cor.test(x, y, method = "pearson")',
        "assumptions": ["linearity", "bivariate_normality", "no_outliers", "homoscedasticity"],
        "reporting": "A Pearson correlation was found, r(df) = __, p = __, 95% CI [__, __].",
    },
    "spearman": {
        "name": "Spearman rank correlation",
        "spss": "Analyze ▸ Correlate ▸ Bivariate ▸ Spearman",
        "r": 'cor.test(x, y, method = "spearman")',
        "assumptions": ["monotonic_relationship", "ordinal_or_continuous"],
        "reporting": "A Spearman correlation was found, r_s = __, p = __.",
    },
    "chi_square_independence": {
        "name": "Chi-square test of independence",
        "spss": "Analyze ▸ Descriptive Statistics ▸ Crosstabs ▸ Statistics ▸ Chi-square",
        "r": 'chisq.test(table(x, y))',
        "assumptions": ["independence", "expected_freq_5", "categorical_variables"],
        "reporting": "A chi-square test of independence was significant, χ²(df, N = __) = __, p = __, Cramér's V = __.",
    },
    "fisher_exact": {
        "name": "Fisher's exact test",
        "spss": "Crosstabs ▸ Statistics ▸ Chi-square (SPSS prints Fisher's Exact for 2×2)",
        "r": 'fisher.test(table(x, y))',
        "assumptions": ["independence", "categorical_variables"],
        "reporting": "Fisher's exact test: p = __ (use when expected counts < 5, or small samples).",
    },
    "mcnemar": {
        "name": "McNemar's test",
        "spss": "Crosstabs ▸ Statistics ▸ McNemar (paired 2×2)",
        "r": 'mcnemar.test(table(before, after))',
        "assumptions": ["paired_binary_data", "independence_of_pairs"],
        "reporting": "McNemar's test on paired proportions: χ²(1) = __, p = __.",
    },
    "chi_square_gof": {
        "name": "Chi-square goodness-of-fit",
        "spss": "Analyze ▸ Nonparametric Tests ▸ Legacy Dialogs ▸ Chi-square",
        "r": 'chisq.test(table(x), p = <expected_proportions>)',
        "assumptions": ["independence", "expected_freq_5", "categorical_variable"],
        "reporting": "A chi-square goodness-of-fit test compared observed vs expected, χ²(df, N = __) = __, p = __.",
    },
    "count_regression": {
        "name": "Poisson / Negative-binomial regression",
        "spss": "Analyze ▸ Generalized Linear Models (Poisson; use neg-binomial if overdispersed)",
        "r": 'glm(y ~ ..., family = poisson) or MASS::glm.nb(...)',
        "assumptions": ["counts_outcome", "independence", "mean_variance_relationship"],
        "reporting": "Poisson regression (or negative binomial if overdispersion, i.e. variance ≫ mean): IRR = __, 95% CI [__, __], p = __.",
    },
}

ASSUMPTION_HELP: dict[str, dict[str, str]] = {
    "normality": {
        "check": "Shapiro-Wilk (best for n<50, still fine to ~2000); Q-Q plot; skewness/kurtosis. For large n, trust the Q-Q plot over the test — significance tests over-reject with big samples.",
        "if_violated": "Use the nonparametric equivalent, transform the data, or rely on the Central Limit Theorem if n per group is large (~30+).",
    },
    "normality_of_residuals": {
        "check": "Check normality of the MODEL RESIDUALS (not the raw outcome): Q-Q plot of residuals, Shapiro-Wilk on residuals.",
        "if_violated": "With large groups ANOVA is robust; otherwise use Kruskal-Wallis (independent) or Friedman (repeated), or transform.",
    },
    "normality_of_differences": {
        "check": "Test normality of the paired DIFFERENCE scores (d = x1 − x2), not each condition separately: Shapiro-Wilk / Q-Q on the differences.",
        "if_violated": "Use the Wilcoxon signed-rank test.",
    },
    "equal_variance": {
        "check": "Levene's test (robust; SPSS prints it automatically). p > .05 → variances homogeneous.",
        "if_violated": "Use Welch's t-test / Welch's ANOVA (with Games-Howell post hoc) — do NOT need equal variances.",
    },
    "independence": {"check": "By design: each observation from a different, unrelated unit; no clustering/repeated measures unless modeled.", "if_violated": "Use a paired/repeated-measures or mixed/multilevel model."},
    "independence_of_pairs": {"check": "Pairs come from independent subjects.", "if_violated": "Use a mixed-effects model."},
    "independence_of_subjects": {"check": "Each subject independent of others.", "if_violated": "Use a mixed-effects model with the clustering factor."},
    "no_outliers": {"check": "Boxplots, z-scores (|z|>3.29), Cook's distance for regression.", "if_violated": "Investigate; use robust methods, a nonparametric test, or report with/without the outlier."},
    "expected_freq_5": {"check": "In Crosstabs, no cell should have expected count < 5 (SPSS warns; ≤20% of cells <5 is the common rule).", "if_violated": "Use Fisher's exact test (2×2) or combine categories."},
    "similar_distribution_shape": {"check": "Overlay group histograms/boxplots. If shapes match, Mann-Whitney/Kruskal compares medians; if not, it compares stochastic dominance.", "if_violated": "Interpret as a difference in distributions, not strictly medians."},
    "symmetry_of_differences": {"check": "Histogram of the difference scores should be roughly symmetric.", "if_violated": "Use the sign test."},
    "sphericity": {"check": "Mauchly's test of sphericity (SPSS prints it).", "if_violated": "Apply Greenhouse-Geisser or Huynh-Feldt correction to the df."},
    "linearity": {"check": "Scatterplot of x vs y should look linear.", "if_violated": "Use Spearman, or transform / model the nonlinearity."},
    "monotonic_relationship": {"check": "Scatterplot shows a consistently increasing or decreasing (not necessarily straight) relationship.", "if_violated": "Reconsider the measure of association."},
    "bivariate_normality": {"check": "Both variables ~normal; no severe outliers.", "if_violated": "Use Spearman."},
    "homoscedasticity": {"check": "Residual spread constant across x (residual plot).", "if_violated": "Use robust SEs or transform."},
}


def _recommend(outcome_type, design, n_groups, normality, equal_variance):
    ot, dz = outcome_type.lower(), design.lower()
    nonparam = (ot == "ordinal") or (normality == "non_normal")
    note = None

    if ot == "count" and dz in ("independent", "paired", "one_sample"):
        return "count_regression", "Count outcomes are usually modeled with Poisson/negative-binomial regression rather than a t-test/ANOVA."

    if dz == "correlation":
        if ot in ("nominal",):
            return "chi_square_independence", "Two categorical variables → test of association, not correlation."
        return ("spearman" if (nonparam or ot == "ordinal") else "pearson"), None

    if dz == "association" or ot == "nominal":
        # categorical outcome — route by design
        if dz == "paired":
            return "mcnemar", ("Paired/repeated categorical data. McNemar applies to a 2×2 table; for >2 categories "
                               "use Stuart–Maxwell, and for >2 repeated binary measures use Cochran's Q.")
        if dz == "one_sample":
            return "chi_square_gof", "One categorical variable compared to expected proportions → goodness-of-fit."
        return "chi_square_independence", "Two categorical variables → test of association. Use Fisher's exact test if any expected cell count < 5 (esp. 2×2)."

    if dz == "one_sample":
        return ("wilcoxon_signed_one" if nonparam else "one_sample_t"), None

    if dz == "paired":
        if n_groups and n_groups > 2:
            return ("friedman" if nonparam else "rm_anova"), None
        return ("wilcoxon_signed" if nonparam else "paired_t"), None

    if dz == "independent":
        if n_groups and n_groups > 2:
            if nonparam:
                return "kruskal_wallis", None
            if equal_variance == "unequal":
                return "welch_anova", None
            return "one_way_anova", "Check Levene's test; if variances are unequal switch to Welch's ANOVA + Games-Howell."
        # two independent groups
        if nonparam:
            return "mann_whitney", None
        if equal_variance == "equal":
            return "independent_t", None
        # unequal OR unknown → Welch is the safer default
        return "welch_t", ("Welch's t-test is the recommended default when equal variances are not confirmed; "
                           "if Levene's test is non-significant you may report Student's t instead.")
    return None, "Could not map this design — check the parameters."


def _assumption_block(test_id):
    out = []
    for a in TESTS[test_id]["assumptions"]:
        h = ASSUMPTION_HELP.get(a)
        if h:
            out.append({"assumption": a, "how_to_check": h["check"], "if_violated": h["if_violated"]})
        else:
            out.append({"assumption": a, "how_to_check": "(design / measurement requirement)", "if_violated": "reconsider design"})
    return out


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def recommend_test(
    outcome_type: Literal["continuous", "ordinal", "nominal", "count"],
    design: Literal["one_sample", "independent", "paired", "correlation", "association"],
    n_groups: int = 2,
    normality: Literal["normal", "non_normal", "unknown"] = "unknown",
    equal_variance: Literal["equal", "unequal", "unknown"] = "unknown",
) -> dict:
    """Recommend the correct statistical test for a study design.

    Use this to answer "what statistical test should I use?". Describe the design:
    outcome_type (continuous/ordinal/nominal/count), design (one_sample = compare one
    group to a value; independent = compare separate groups; paired = same subjects over
    time/conditions; correlation = relationship between two variables; association = two
    categorical variables), how many groups, whether the outcome is ~normal, and whether
    group variances are equal. Returns the test, why, assumptions, SPSS path, R code,
    an APA reporting template, and fallbacks if assumptions fail.
    """
    test_id, note = _recommend(outcome_type, design, n_groups, normality, equal_variance)
    if test_id is None:
        return {"error": note, "hint": "Try recommend_test(outcome_type='continuous', design='independent', n_groups=2)."}
    info = TESTS[test_id]
    return {
        "recommended_test": info["name"],
        "test_id": test_id,
        "why": note or f"Given a {outcome_type} outcome, {'an' if design[0] in 'aeiou' else 'a'} {design} design"
                       + (f" with {n_groups} groups" if design in ("independent", "paired") else "")
                       + f", normality={normality}, equal_variance={equal_variance}.",
        "assumptions": _assumption_block(test_id),
        "spss_path": info["spss"],
        "r_code": info["r"],
        "reporting_template": info["reporting"],
        "report_effect_size": True,
        "caveat": "Guidance based on the classic decision tree. Always verify assumptions on YOUR data; "
                  "with large samples parametric tests are robust to mild non-normality (CLT).",
    }


@mcp.tool()
def check_assumptions(test_id: str) -> dict:
    """List the assumptions of a specific test, how to check each, and what to do if violated.

    Pass a test_id from recommend_test / list_supported_tests (e.g. 'independent_t',
    'one_way_anova', 'pearson', 'chi_square_independence').
    """
    if test_id not in TESTS:
        return {"error": f"Unknown test_id '{test_id}'.", "supported": sorted(TESTS.keys())}
    return {"test": TESTS[test_id]["name"], "test_id": test_id, "assumptions": _assumption_block(test_id)}


@mcp.tool()
def interpret_result(
    p_value: float,
    alpha: float = 0.05,
    test_name: str = "the test",
    effect_size: float | None = None,
    effect_size_type: str = "",
) -> dict:
    """Interpret a p-value correctly and produce a defensible, APA-style conclusion.

    Guards against the classic mistakes: a non-significant result does NOT prove the null,
    and statistical significance is not practical importance (report effect size + CI).
    """
    if not 0.0 <= p_value <= 1.0:
        return {"error": "p_value must be between 0 and 1."}
    if not 0.0 < alpha < 1.0:
        return {"error": "alpha must be strictly between 0 and 1 (e.g. 0.05)."}
    sig = p_value < alpha
    verdict = (f"p = {p_value:g} < α = {alpha:g}: statistically significant — reject the null hypothesis."
               if sig else
               f"p = {p_value:g} ≥ α = {alpha:g}: NOT statistically significant — fail to reject the null. "
               f"This does NOT prove the null is true (absence of evidence ≠ evidence of absence).")
    cautions = [
        "Significance ≠ importance: always report and interpret the effect size and its 95% CI.",
        "p is not the probability the null is true, nor the probability of replication.",
    ]
    if abs(p_value - alpha) <= 0.01:
        cautions.append(f"p is close to α — avoid a hard 'significant/not' dichotomy; report the exact p and effect size.")
    es_note = None
    if effect_size is not None:
        es_note = _interpret_effect_size(effect_size, effect_size_type)
    return {
        "significant": sig,
        "interpretation": verdict,
        "effect_size_interpretation": es_note,
        "cautions": cautions,
        "reporting_tip": f"Report as: '{test_name} was {'significant' if sig else 'not significant'}, "
                         f"p = {p_value:g}' plus the test statistic, df, and effect size with 95% CI.",
    }


def _interpret_effect_size(value: float, kind: str) -> str:
    k = (kind or "").lower()
    v = abs(value)
    if k in ("d", "cohen_d", "cohens_d", "dz", "g"):
        band = "small" if v < 0.5 else "medium" if v < 0.8 else "large"
        return f"Cohen's d = {value:g} → {band} effect (benchmarks 0.2/0.5/0.8)."
    if k in ("r", "pearson", "correlation", "spearman"):
        band = "small" if v < 0.3 else "medium" if v < 0.5 else "large"
        return f"r = {value:g} → {band} effect (benchmarks 0.1/0.3/0.5); r² = {value*value:.3f} variance explained."
    if k in ("eta2", "eta_squared", "η2"):
        band = "small" if v < 0.06 else "medium" if v < 0.14 else "large"
        return f"η² = {value:g} → {band} effect (benchmarks 0.01/0.06/0.14)."
    if k in ("h", "cohen_h"):
        band = "small" if v < 0.5 else "medium" if v < 0.8 else "large"
        return f"Cohen's h = {value:g} → {band} effect."
    return f"Effect size = {value:g} (specify effect_size_type as d/r/eta2/h for a benchmarked interpretation)."


@mcp.tool()
def plan_sample_size(
    comparison: Literal["two_means", "paired_means", "two_proportions", "correlation"],
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> dict:
    """A priori power analysis: the required sample size for a target power.

    effect_size is Cohen's d for two_means/paired_means, Cohen's h for two_proportions,
    and the correlation r for correlation. Uses a normal approximation — treat the result
    as a close lower bound and confirm exact numbers in G*Power for t-based tests.
    """
    if effect_size == 0:
        return {"error": "effect_size cannot be 0 (an effect of zero needs infinite n)."}
    if not (0 < alpha < 1 and 0 < power < 1):
        return {"error": "alpha and power must be in (0,1)."}
    za = _norm_ppf(1 - alpha / (2 if two_sided else 1))
    zb = _norm_ppf(power)
    es = abs(effect_size)
    out = {"alpha": alpha, "power": power, "two_sided": two_sided,
           "effect_size": effect_size, "comparison": comparison}
    if comparison == "two_means":
        n = math.ceil(2 * ((za + zb) / es) ** 2)
        out.update(n_per_group=n, total_n=2 * n)
    elif comparison == "paired_means":
        n = math.ceil(((za + zb) / es) ** 2)
        out.update(n_pairs=n, total_n=n)
    elif comparison == "two_proportions":
        n = math.ceil(((za + zb) / es) ** 2)
        out.update(n_per_group=n, total_n=2 * n, note_effect="effect_size is interpreted as Cohen's h.")
    elif comparison == "correlation":
        if es >= 1:
            return {"error": "correlation effect_size must be < 1."}
        C = 0.5 * math.log((1 + es) / (1 - es))
        n = math.ceil(((za + zb) / C) ** 2) + 3
        out.update(total_n=n)
    out["caveat"] = ("Normal-approximation power analysis (close lower bound). For t-based tests add ~1–2 per group "
                     "and confirm in G*Power. Benchmarks — d: 0.2/0.5/0.8, r: 0.1/0.3/0.5, h: 0.2/0.5/0.8.")
    return out


@mcp.tool()
def normality_guide() -> dict:
    """How to decide and report normality correctly — the #1 thing students get wrong."""
    return {
        "what_to_test": "For t-tests: the outcome within each group. For paired: the DIFFERENCE scores. For ANOVA/regression: the MODEL RESIDUALS. Never the whole dataset ignoring groups.",
        "tests": {
            "Shapiro-Wilk": "Preferred; most powerful. Ideal n < 50, usable to ~2000.",
            "Kolmogorov-Smirnov (Lilliefors)": "SPSS 'Explore' prints it; weaker than Shapiro-Wilk.",
        },
        "visual": "Q-Q plot and histogram are the real evidence. With large n, normality TESTS over-reject trivial deviations — trust the Q-Q plot.",
        "descriptives": "Skewness and kurtosis roughly within ±1 (some accept ±2) suggest approximate normality.",
        "clt": "With large groups (~30+), t-tests and ANOVA are robust to non-normality by the Central Limit Theorem — you often do not need a nonparametric test.",
        "if_non_normal": "Use the nonparametric equivalent (Mann-Whitney / Wilcoxon / Kruskal-Wallis / Friedman / Spearman), transform (log, sqrt), or use robust methods.",
        "reporting": "e.g. 'Shapiro-Wilk indicated the differences were approximately normal (W = __, p = __)' or note the Q-Q inspection.",
    }


@mcp.tool()
def list_supported_tests() -> dict:
    """List every test this advisor knows, with its SPSS menu path."""
    return {tid: {"name": t["name"], "spss": t["spss"]} for tid, t in TESTS.items()}


# --------------------------------------------------------------------------- #
def main() -> None:
    use_http = ("--http" in sys.argv) or os.environ.get("MCP_HTTP")
    if use_http:
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.host = os.environ.get("HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("PORT", "8000"))
        # FastMCP's DNS-rebinding protection defaults to localhost-only and returns
        # HTTP 421 for any other Host header — which silently breaks every hosted deploy.
        public = os.environ.get("PUBLIC_HOST", "").strip()  # e.g. myservice.onrender.com
        if public:
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[public, f"{public}:*", "localhost", "localhost:*",
                               "127.0.0.1", "127.0.0.1:*"],
                allowed_origins=[f"https://{public}", f"http://{public}"],
            )
        else:
            # No fixed public host set — disable the host check so it works behind any proxy.
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False)
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio (default) — for Claude Desktop / Claude Code / any local MCP client


if __name__ == "__main__":
    main()
