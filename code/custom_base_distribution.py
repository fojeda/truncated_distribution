"""Implement Custom example distribution when icdf method might not be exposed."""

class CustomWeibull(Weibull):
  """# Create the original Weibull distribution
      # NOTE: Weibull distribution in torch.distributions might NOT implement icdf directly.
      # To make TransformedTruncatedDistribution work, we either need to add icdf to Weibull
      # or use a distribution that natively supports it (e.g., Normal, by converting from QuantileFunction).
      # For this example, if Weibull is desired, you would typically need to patch it or use a custom Weibull.
      # For demonstration purposes, I'll rely on a hypothetical Weibull.icdf or a patched version if it exists.
      # If you run this with a standard torch.distributions.Weibull, it will now raise the NotImplementedError.
      # To make it runnable for Weibull *specifically*, a custom Weibull or patch is needed.
      # For now, let's proceed assuming base_weibull.icdf would be available if this were a custom class.
      # For demonstration, I will use a dummy class that *does* have icdf, or rely on future PyTorch updates.
      # If Weibull does not have icdf, this will error.
      # For a practical solution, you'd implement a custom Weibull distribution with icdf or a wrapper.
      
      # Example of how you *might* make Weibull work if it's external or patched:"""
    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        # This is the ICDF logic for Weibull
        # F(x; lambda, k) = 1 - exp(-(x/lambda)^k)
        # x = lambda * (-log(1 - F))^(1/k)
        value = value.clamp(torch.finfo(value.dtype).tiny, 1.0 - torch.finfo(value.dtype).tiny)
        term = -torch.log(1 - value)
        icdf_val = self.scale * (term ** (1 / self.concentration))
        return icdf_val
