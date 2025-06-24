"""Truncated distribution implemented using TransformedDistribution."""

import torch
from torch.distributions import Distribution, Uniform, TransformedDistribution
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.distributions.utils import broadcast_all

class TransformedTruncatedDistribution(TransformedDistribution):
    """
    Truncated distribution implemented using TransformedDistribution.
    """
    def __init__(self, base_distribution: Distribution, lower_bound: torch.Tensor, upper_bound: torch.Tensor, validate_args=None):
        # Infer parameter dtype and device from the original base_distribution
        param_dtype = torch.float32
        param_device = 'cpu'
        for param_name in ['loc', 'scale', 'rate', 'concentration1', 'concentration0', 'probs', 'shape', 'concentration']:
            if hasattr(base_distribution, param_name):
                param_tensor = getattr(base_distribution, param_name)
                if isinstance(param_tensor, torch.Tensor):
                    param_dtype = param_tensor.dtype
                    param_device = param_tensor.device
                    break 

        # Ensure bounds are tensors with correct dtype and device
        lower_bound = torch.as_tensor(lower_bound, dtype=param_dtype, device=param_device)
        upper_bound = torch.as_tensor(upper_bound, dtype=param_dtype, device=param_device)

        # Broadcast base_distribution and bounds to a common batch_shape
        # The base_distribution for TransformedDistribution will be Uniform(0,1)
        # We need to ensure the transformation object knows the correct batched bounds.
        
        # Determine the common batch shape including the base_distribution's original batch_shape
        # and the bounds' shapes.
        common_batch_shape = base_distribution.batch_shape
        try:
            common_batch_shape = torch.broadcast_shapes(common_batch_shape, lower_bound.shape, upper_bound.shape)
        except RuntimeError as e:
            raise ValueError(f"Bounds {lower_bound.shape} and {upper_bound.shape} are not broadcastable "
                             f"with base_distribution batch_shape {base_distribution.batch_shape}. Error: {e}")

        # Expand the original base_distribution (e.g., Weibull) to this common batch shape
        # This expanded base_distribution will be used *inside* the TruncationTransform.
        expanded_base_dist = base_distribution.expand(common_batch_shape)
        expanded_lower_bound = lower_bound.expand(common_batch_shape)
        expanded_upper_bound = upper_bound.expand(common_batch_shape)

        # The actual base for TransformedDistribution is Uniform(0,1)
        # It needs to have the same batch_shape as the final truncated distribution.
        uniform_base = Uniform(torch.zeros(common_batch_shape, dtype=param_dtype, device=param_device), 
                               torch.ones(common_batch_shape, dtype=param_dtype, device=param_device))

        # Create the custom truncation transform
        transform = TruncationTransform(expanded_base_dist, expanded_lower_bound, expanded_upper_bound)
        
        super().__init__(uniform_base, [transform], validate_args=validate_args)
        self._truncated_lower = expanded_lower_bound # Store for easy access in plotting
        self._truncated_upper = expanded_upper_bound

    @property
    def lower_bound(self):
        return self._truncated_lower
    
    @property
    def upper_bound(self):
        return self._truncated_upper

    # Need to override log_prob and cdf to handle values outside truncation range
    # TransformedDistribution's log_prob will return -inf for values where transform._inverse is not valid
    # or if the base_distribution.log_prob of the inverse transformed value is -inf.
    # The current `log_abs_det_jacobian` and `_inverse` in `TruncationTransform`
    # implicitly handles values outside the domain for the original base_distribution.
    # However, `TransformedDistribution` does not inherently know about the "outer" truncation.
    # We must explicitly set log_prob to -inf for values outside [lower_bound, upper_bound]
    # This also applies to cdf.

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Determine the correct dtype and device from the internal base_distribution's parameters.
        if hasattr(self.base_dist.low, 'dtype'): # Access Uniform(0,1)'s parameters
            data_dtype = self.base_dist.low.dtype
            data_device = self.base_dist.low.device
        else: # Fallback
            data_dtype = torch.float32
            data_device = 'cpu'

        value = torch.as_tensor(value, dtype=data_dtype, device=data_device)

        log_probs = super().log_prob(value) # Call TransformedDistribution's log_prob

        # Apply mask for values strictly outside the truncated interval
        # broadcast value with self.lower_bound and self.upper_bound
        value_b, lower_b, upper_b = broadcast_all(value, self.lower_bound, self.upper_bound)
        outside_bounds_mask = (value_b < lower_b) | (value_b > upper_b)
        
        log_probs = torch.where(outside_bounds_mask, torch.full_like(log_probs, -float('inf')), log_probs)
        return log_probs

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        # Determine the correct dtype and device
        if hasattr(self.base_dist.low, 'dtype'):
            data_dtype = self.base_dist.low.dtype
            data_device = self.base_dist.low.device
        else:
            data_dtype = torch.float32
            data_device = 'cpu'
            
        value = torch.as_tensor(value, dtype=data_dtype, device=data_device)
        
        # Broadcast value with self.lower_bound and self.upper_bound
        value_b, lower_b, upper_b = broadcast_all(value, self.lower_bound, self.upper_bound)

        cdf_val = torch.zeros_like(value_b)

        # Values below lower_bound have CDF 0
        cdf_val = torch.where(value_b < lower_b, torch.zeros_like(cdf_val), cdf_val)

        # Values above upper_bound have CDF 1
        cdf_val = torch.where(value_b > upper_b, torch.ones_like(cdf_val), cdf_val)

        # Values within [lower_bound, upper_bound] use the inverse transformation to get Uniform CDF
        # This is essentially applying the _inverse method of our custom transform
        within_bounds_mask = (value_b >= lower_b) & (value_b <= upper_b)
        
        # Apply the inverse transform which yields a probability in [0,1]
        # Then, this is directly the CDF of the TransformedTruncatedDistribution.
        # Ensure that 'transform' is accessible.
        # self.transforms is a list, so self.transforms[0] is our TruncationTransform
        inverse_transformed_value = self.transforms[0]._inverse(value_b[within_bounds_mask])
        cdf_val[within_bounds_mask] = inverse_transformed_value

        return cdf_val


class TruncationTransform(Transform):
    """
    A custom transform for creating a truncated distribution from a Uniform(0,1) base.
    This transform applies the inverse CDF method to map Uniform(0,1) samples
    to samples from the truncated base distribution.
    """
    domain = constraints.unit_interval
    codomain = constraints.real
    bijective = True # The transformation is bijective within its domain/codomain
    # This transform can be used with Weibull, Normal, etc., if their ICDF is provided.
    # Here, we specialize it for Weibull, but it can be generalized.
    event_dim = 0 # Applies to scalar events

    def __init__(self, base_distribution: Distribution, lower_bound: torch.Tensor, upper_bound: torch.Tensor, validate_args=None):
        # FIX: Removed 'event_dim' from super().__init__ as it's a class attribute, not an init argument for Transform
        super().__init__(cache_size=0)
        self.base_distribution = base_distribution
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # **IMPORTANT IMPROVEMENT:** Check if base_distribution provides an `icdf` method
        if not hasattr(base_distribution, 'icdf'):
            raise NotImplementedError(
                f"Base distribution {type(base_distribution).__name__} must implement "
                "`icdf` method to be used with TruncationTransform."
            )

        # Pre-calculate constants for the transformation
        self.cdf_lower = self.base_distribution.cdf(lower_bound)
        self.cdf_upper = self.base_distribution.cdf(upper_bound)
        self.interval_prob = (self.cdf_upper - self.cdf_lower).clamp_min(torch.finfo(self.cdf_upper.dtype).tiny)
        self.log_interval_prob = self.interval_prob.log()

    def _call(self, value: torch.Tensor) -> torch.Tensor:
        """
        Maps a Uniform(0,1) `value` to the truncated distribution's value.
        This is F_base_inv(value * (F_base(upper) - F_base(lower)) + F_base(lower))
        """
        # Linear scaling of the uniform probability to the effective CDF range
        transformed_prob = value * self.interval_prob + self.cdf_lower

        # Apply the inverse CDF of the base distribution directly
        # The check in __init__ ensures that self.base_distribution.icdf exists.
        transformed_value = self.base_distribution.icdf(transformed_prob)
        
        return transformed_value

    def _inverse(self, value: torch.Tensor) -> torch.Tensor:
        """
        Maps a value from the truncated distribution back to a Uniform(0,1) value.
        This is (F_base(value) - F_base(lower)) / (F_base(upper) - F_base(lower))
        """
        # Clamp values to the truncation bounds to handle potential floating point
        # inaccuracies or values slightly outside due to numerical issues,
        # ensuring the CDF is within the expected range for the inverse transformation.
        value = value.clamp(min=self.lower_bound, max=self.upper_bound)

        cdf_at_value = self.base_distribution.cdf(value)
        inverse_value = (cdf_at_value - self.cdf_lower) / self.interval_prob
        return inverse_value

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the log absolute determinant of the Jacobian for the transformation.
        Here, x is the input (from Uniform(0,1)), y is the output (from truncated distribution).
        We need log |dy/dx| = log |d(F_base_inv(...))/du|
        Derived as: log(interval_prob) - log(pdf_base(y))
        """
        log_pdf_base_at_y = self.base_distribution.log_prob(y)
        return self.log_interval_prob - log_pdf_base_at_y
