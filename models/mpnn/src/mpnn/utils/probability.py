import numpy as np


def sample_bernoulli_rv(p):
    """
    Given a probability p, representing the success probability of a Bernoulli
    distribution, sample X ~ Bernoulli(p).

    Arguments:
        p (float): a float between 0 and 1, representing the success probability
            of a Bernoulli distribution.

    Returns:
        x (int): the result of sampling the random variable X ~ Bernoulli(p).
            P(X = 1) = p
            P(X = 0) = 1 - p.
    """
    # Check that 0 <= p <= 1.
    if p < 0 or p > 1:
        raise ValueError("The success probability p must be between 0 and 1 inclusive.")

    # Handle the edge cases, otherwise utilize the numpy uniform distribution.
    if p == 0:
        x = 0
    elif p == 1:
        x = 1
    else:
        # Sample the Y ~ Uniform(0, 1) distribution.
        uniform_sample = np.random.uniform(0.0, 1.0)

        # P(Y < p) = p.
        if uniform_sample < p:
            x = 1
        else:
            x = 0

    return x
