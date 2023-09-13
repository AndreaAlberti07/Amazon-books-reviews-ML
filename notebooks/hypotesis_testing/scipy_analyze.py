# Definition of a function to calculate many statistics of a given dataset
def scipy_analize(variable_1, varname_1, variable_2, varname_2, analysis_list):
    """
    This function takes two variables and performs the following analysis:
    1. Check if the data are normally distributed
    2. Check the correlation between the two variables
    3. Check if the two variables follow same distribution
    4. Check if the two variables are independent
    5. Check if the mean of the two variables are equal using Wilcoxon

    Parameters:
    variable_1: data (i.e. list, array, series, etc.)
    varname_1: string (name of the variable)
    variable_2: data (i.e. list, array, series, etc.)
    varname_2: string (name of the variable)
    analysis_list: list of integers (1-5) to indicate which analysis to perform

    Returns:
    None
    """
    import scipy.stats as stats
    import pandas as pd

    if 1 in analysis_list:

        # Check if the data are normally distributed
        score_ntest, p_value_score_ntest = stats.normaltest(variable_1)
        help_ntest, p_value_help_ntest = stats.normaltest(variable_2)

        # Print the conclusion
        if p_value_score_ntest < 0.05:
            print(
                f"The p-value for normality test of {varname_1} is {p_value_score_ntest}. The data is not normally distributed.")
        else:
            print(
                f"The p-value for normality test of {varname_1} is {p_value_score_ntest}. The data is normally distributed.")

        if p_value_help_ntest < 0.05:
            print(
                f"The p-value for normality test of {varname_2} is {p_value_help_ntest}. The data is not normally distributed.")
        else:
            print(
                f"The p-value for normality test of {varname_2} is {p_value_help_ntest}. The data is normally distributed.")

    if 2 in analysis_list:
        if p_value_score_ntest < 0.05 or p_value_help_ntest < 0.05:
            print("Since at least one of the variables is not normally distributed, we will use Spearman's correlation.")
            # Check the correlation between the two variables
            spearman_coeff, p_value_corr = stats.spearmanr(
                variable_1, variable_2)

            # Print the conclusion
            if p_value_corr < 0.05:
                print(
                    f"The p-value for correlation value: {spearman_coeff} between {varname_1} and {varname_2} is {p_value_corr}. The correlation is significant.")
            else:
                print(
                    f"The p-value for correlation value: {spearman_coeff} between {varname_1} and {varname_2} is {p_value_corr}. The correlation is not significant.")

        if p_value_score_ntest > 0.05 and p_value_help_ntest > 0.05:
            print(
                "Since both the variables are normally distributed, we will use Pearson's correlation.")
            # Check the correlation between the two variables
            person_coeff, p_value_corr = stats.pearsonr(variable_1, variable_2)

            # Print the conclusion
            if p_value_corr < 0.05:
                print(
                    f"The p-value for correlation value: {person_coeff} between {varname_1} and {varname_2} is {p_value_corr}. The correlation is significant.")
            else:
                print(
                    f"The p-value for correlation value: {person_coeff} between {varname_1} and {varname_2} is {p_value_corr}. The correlation is not significant.")

    if 3 in analysis_list:
        # Check if the two variables follow same distribution
        ks_stat, p_value_ks = stats.ks_2samp(variable_1, variable_2)

        # Print the conclusion
        if p_value_ks < 0.05:
            print(
                f"The p-value for Kolmogorov-Smirnov test between {varname_1} and {varname_2} is {p_value_ks}. The two variables do not follow the same distribution.")
        else:
            print(
                f"The p-value for Kolmogorov-Smirnov test between {varname_1} and {varname_2} is {p_value_ks}. The two variables follow the same distribution.")

    if 4 in analysis_list:
        # Check if the two variables are independent
        chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(
            pd.crosstab(variable_1, variable_2))

        # Print the conclusion
        if p_value_chi2 < 0.05:
            print(
                f"The p-value for Chi-Square test between {varname_1} and {varname_2} is {p_value_chi2}. The two variables are not independent.")
        else:
            print(
                f"The p-value for Chi-Square test between {varname_1} and {varname_2} is {p_value_chi2}. The two variables are independent.")

    if 5 in analysis_list:
        if p_value_score_ntest < 0.05 or p_value_help_ntest < 0.05:
            print(
                "Since at least one of the variables is not normally distributed, we will use Wilcoxon.")
            # Check if the mean of the two variables are equal using Wilcoxon
            wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(
                variable_1, variable_2)

            # Print the conclusion
            if p_value_wilcoxon < 0.05:
                print(
                    f"The p-value for Wilcoxon test between {varname_1} and {varname_2} is {p_value_wilcoxon}. The two variables do not have the same mean.")
            else:
                print(
                    f"The p-value for Wilcoxon test between {varname_1} and {varname_2} is {p_value_wilcoxon}. The two variables have the same mean.")

        if p_value_score_ntest > 0.05 and p_value_help_ntest > 0.05:
            print(
                "Since both the variables are normally distributed, we will use t-test.")
            # Check if the mean of the two variables are equal using t-test
            t_stat, p_value_t = stats.ttest_ind(variable_1, variable_2)

            # Print the conclusion
            if p_value_t < 0.05:
                print(
                    f"The p-value for t-test between {varname_1} and {varname_2} is {p_value_t}. The two variables do not have the same mean.")
            else:
                print(
                    f"The p-value for t-test between {varname_1} and {varname_2} is {p_value_t}. The two variables have the same mean.")
