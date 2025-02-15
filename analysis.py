def calc_hedges_g(mu1: float, mu2: float, sd_pooled: float) -> float:
    """Calculate Hedges G metric: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
    Args:
        mu1 (float): mean of group 1
        mu2 (float): mean of group 2
    2 (float): mean of group 2
        sd_pooled (float): pooled standard deviation of both groups
        
    Returns:
        float: Hedges G metric
    """
    return (mu1 - mu2) / sd_pooled


def calc_sd_pooled(n1: int, n2: int, sd1: float, sd2: float) -> float:
    """Calculate pooled standard deviation: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
    Args:
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        
    Returns:
        float: pooled standard deviation
    """
    return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))