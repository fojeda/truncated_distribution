`Pytorch` implementation of a Truncated distribution for most common base distributions. Two implementation versions are provided.

1. Direct implementation:
   - Definition of a new class `TruncatedDistribution` that subclasses `torch.distributions.Distribution`.
   - This class represent a truncated probability distribution, meaning that the probability mass of the base distribution outside specified lower and upper limits will be re-normalized to only exist within those limits.
   - The TruncatedDistribution includes:
     - An __init__ method to set up the base distribution and the truncation bounds. It will also calculate the normalization constant required for the `log_prob` method.
     - A `log_prob` method that computes the log probability density (or mass) for a given value. If the value falls outside the truncation bounds, its log probability will be negative infinity. Otherwise, it will be the base distribution's log probability, adjusted by the pre-calculated normalization constant.
     - A `cdf` method that computes the cumulative distribution function (CDF) for the truncated distribution.
     - A `sample` method that generates samples from the truncated distribution using **rejection sampling**, ensuring all samples fall within the defined bounds.
     - An `expand` method for proper broadcasting behavior within PyTorch.
2. Transformed implementation:
  - Definition of q new class `TruncatedDistribution` by subclassing `torch.distributions.TransformedDistribution`.
  - This approach leverages **PyTorch's transforms framework** and, crucially, avoids the need for **rejection sampling** during the sample method, which can be more efficient for complex distributions or very narrow truncation ranges.
  - The core idea behind the `TransformedDistribution` approach for truncation is to:
      1. Use a `Uniform(0,1)` distribution as the base distribution.
      2. Define a custom `Transform` that maps values from this `Uniform(0,1)` distribution to the desired truncated distribution's support.
      3. This mapping is achieved using the Inverse Cumulative Distribution Function (ICDF) of the base distribution, scaled to the truncated interval.
      4. The `log_prob` calculation is handled automatically by `TransformedDistribution` using the change of variables formula, which requires the `log_abs_det_jacobian` of the transformation.
  - This version offers a potentially more performant sampling strategy if the icdf of the base distribution is available or can be efficiently computed
