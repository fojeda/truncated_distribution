"""TruncatedDistribution class implementation"""
import torch
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

class TruncatedDistribution(Distribution):
    """
    A generic truncated distribution class for PyTorch.
    Subclasses `torch.distributions.Distribution` directly.
    """
    arg_constraints = {
        'lower_bound': constraints.real,
        'upper_bound': constraints.real
    }

    def __init__(self, base_distribution: Distribution, lower_bound: torch.Tensor, upper_bound: torch.Tensor, validate_args=None):
        if not isinstance(base_distribution, Distribution):
            raise TypeError("base_distribution must be an instance of torch.distributions.Distribution")
        
        if not (hasattr(base_distribution, 'cdf') and hasattr(base_distribution, 'log_prob')):
             raise NotImplementedError(
                 f"Base distribution {type(base_distribution).__name__} must implement "
                 "`cdf` and `log_prob` methods for truncation."
             )

        param_dtype = torch.float32
        param_device = 'cpu'
        # Iterate through common parameter names to find dtype and device
        for param_name in ['loc', 'scale', 'rate', 'concentration1', 'concentration0', 'probs', 'shape', 'concentration']:
            if hasattr(base_distribution, param_name):
                param_tensor = getattr(base_distribution, param_name)
                if isinstance(param_tensor, torch.Tensor):
                    param_dtype = param_tensor.dtype
                    param_device = param_tensor.device
                    break 

        lower_bound = torch.as_tensor(lower_bound, dtype=param_dtype, device=param_device)
        upper_bound = torch.as_tensor(upper_bound, dtype=param_dtype, device=param_device)

        try:
            common_shape = base_distribution.batch_shape
            common_shape = torch.broadcast_shapes(common_shape, lower_bound.shape, upper_bound.shape)
            self.base_distribution = base_distribution.expand(common_shape)
            self.lower_bound, self.upper_bound = broadcast_all(lower_bound, upper_bound)
            
        except RuntimeError as e:
            raise ValueError(
                f"Bounds {lower_bound.shape} and {upper_bound.shape} are not broadcastable "
                f"with base_distribution batch_shape {base_distribution.batch_shape}. Error: {e}"
            )

        if not torch.all(self.lower_bound < self.upper_bound):
            raise ValueError("lower_bound must be strictly less than upper_bound for a valid truncated interval.")

        cdf_upper = self.base_distribution.cdf(self.upper_bound)
        cdf_lower = self.base_distribution.cdf(self.lower_bound)
        
        self.normalization_constant = (cdf_upper - cdf_lower).clamp_min(torch.finfo(cdf_upper.dtype).tiny)
        self.log_normalization_constant = self.normalization_constant.log()

        super().__init__(batch_shape=self.base_distribution.batch_shape,
                         event_shape=self.base_distribution.event_shape,
                         validate_args=validate_args)

    @property
    def support(self):
        return constraints.real 

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Determine the correct dtype and device from base_distribution's parameters.
        # This generic approach handles various distribution types more robustly.
        if hasattr(self.base_distribution, 'scale'): # Common for Weibull, Normal, Cauchy
            data_dtype = self.base_distribution.scale.dtype
            data_device = self.base_distribution.scale.device
        elif hasattr(self.base_distribution, 'rate'): # For Exponential
            data_dtype = self.base_distribution.rate.dtype
            data_device = self.base_distribution.rate.device
        elif hasattr(self.base_distribution, 'concentration1'): # For Beta
            data_dtype = self.base_distribution.concentration1.dtype
            data_device = self.base_distribution.concentration1.device
        else: # Fallback
            data_dtype = torch.float32
            data_device = 'cpu'

        value = torch.as_tensor(value, dtype=data_dtype, device=data_device)

        try:
            value_b, lower_b, upper_b = broadcast_all(value, self.lower_bound, self.upper_bound)
        except RuntimeError as e:
            raise ValueError(f"Observed value shape {value.shape} not broadcastable with distribution bounds shapes. Error: {e}")

        log_probs = torch.full_like(value_b, -float('inf'))

        within_bounds_mask = (value_b >= lower_b) & (value_b <= upper_b)
        
        log_prob_base = self.base_distribution.log_prob(value_b)
        
        log_norm_const_expanded, _ = broadcast_all(self.log_normalization_constant, log_prob_base)

        log_probs_within_bounds = log_prob_base - log_norm_const_expanded

        log_probs = torch.where(within_bounds_mask, log_probs_within_bounds, log_probs)

        return log_probs

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative distribution function (CDF) for the truncated distribution.

        Args:
            value (torch.Tensor): The value(s) for which to compute the CDF.

        Returns:
            torch.Tensor: The CDF value for each input.
        """
        # Determine the correct dtype and device from base_distribution's parameters.
        if hasattr(self.base_distribution, 'scale'):
            data_dtype = self.base_distribution.scale.dtype
            data_device = self.base_distribution.scale.device
        elif hasattr(self.base_distribution, 'rate'):
            data_dtype = self.base_distribution.rate.dtype
            data_device = self.base_distribution.rate.device
        elif hasattr(self.base_distribution, 'concentration1'):
            data_dtype = self.base_distribution.concentration1.dtype
            data_device = self.base_distribution.concentration1.device
        else:
            data_dtype = torch.float32
            data_device = 'cpu'

        value = torch.as_tensor(value, dtype=data_dtype, device=data_device)

        # Broadcast value with self.lower_bound and self.upper_bound
        value_b, lower_b, upper_b = broadcast_all(value, self.lower_bound, self.upper_bound)

        cdf_val = torch.zeros_like(value_b)

        # Mask for values within the truncation bounds [lower_b, upper_b]
        within_bounds_mask = (value_b >= lower_b) & (value_b <= upper_b)

        # Calculate base CDFs at the value and lower bound
        cdf_at_value_base = self.base_distribution.cdf(value_b)
        cdf_at_lower_base = self.base_distribution.cdf(lower_b)

        # The truncated CDF formula: (F_B(x) - F_B(lower)) / (F_B(upper) - F_B(lower))
        # Use normalization_constant which is (F_B(upper) - F_B(lower))
        truncated_cdf_within_bounds = (cdf_at_value_base - cdf_at_lower_base) / self.normalization_constant

        # Apply results based on masks
        # If value < lower_b, cdf_val remains 0 (initialized as such)
        # If value > upper_b, cdf_val is 1
        cdf_val = torch.where(value_b > upper_b, torch.ones_like(cdf_val), cdf_val)
        # For values within bounds, apply the calculated truncated_cdf_within_bounds
        cdf_val = torch.where(within_bounds_mask, truncated_cdf_within_bounds, cdf_val)

        return cdf_val

    def sample(self, sample_shape=torch.Size()):
        full_sample_shape = sample_shape + self.base_distribution.batch_shape + self.base_distribution.event_shape
        
        # Determine the correct dtype and device for samples
        if hasattr(self.base_distribution, 'scale'):
            sample_dtype = self.base_distribution.scale.dtype
            sample_device = self.base_distribution.scale.device
        elif hasattr(self.base_distribution, 'rate'):
            sample_dtype = self.base_distribution.rate.dtype
            sample_device = self.base_distribution.rate.device
        elif hasattr(self.base_distribution, 'concentration1'):
            sample_dtype = self.base_distribution.concentration1.dtype
            sample_device = self.base_distribution.concentration1.device
        else:
            sample_dtype = torch.float32
            sample_device = 'cpu'

        samples = torch.empty(full_sample_shape, dtype=sample_dtype, device=sample_device)
        
        num_remaining = samples.numel()
        mask_remaining = torch.ones(full_sample_shape, dtype=torch.bool, device=samples.device)
        
        max_attempts = 10000 
        attempts = 0

        while num_remaining > 0 and attempts < max_attempts:
            current_samples_full_batch = self.base_distribution.sample(sample_shape)
            
            expanded_lower_bound = self.lower_bound.expand(current_samples_full_batch.shape)
            expanded_upper_bound = self.upper_bound.expand(current_samples_full_batch.shape)

            within_bounds = (current_samples_full_batch >= expanded_lower_bound) & \
                            (current_samples_full_batch <= expanded_upper_bound)

            samples[mask_remaining & within_bounds] = current_samples_full_batch[mask_remaining & within_bounds]
            
            mask_remaining[mask_remaining & within_bounds] = False
            
            num_remaining = mask_remaining.sum().item()
            attempts += 1
        
        if num_remaining > 0:
            print(f"Warning (TruncatedDistribution Direct): Could not sample all {num_remaining} values within bounds after {max_attempts} attempts. "
                  f"Filling remaining samples with lower_bound. Consider increasing max_attempts or checking bounds/base_distribution.")
            # For remaining elements, assign the lower bound to indicate truncation failure/edge case
            # This handles cases where the sample fails to be generated within bounds.
            # We need to expand lower_bound to the shape of the elements it's being assigned to.
            # `samples[mask_remaining]` already gives a flat view, so expand `lower_bound` to match that.
            # This assumes lower_bound is scalar or broadcastable.
            if self.lower_bound.numel() == 1: # Scalar lower bound
                samples[mask_remaining] = self.lower_bound.item()
            else: # Batched lower bound, needs to match elements in mask_remaining
                # This is tricky for batched scenarios if mask_remaining is not aligned with batch dimensions.
                # A robust way is to re-evaluate the elements of lower_bound that correspond to `mask_remaining`
                # For simplicity here, if batched, we'll try to apply it.
                # A more robust solution might involve creating a tensor of appropriate shape from lower_bound
                # and indexing that.
                # For now, let's assume `mask_remaining` allows direct broadcast-like assignment or
                # that we sample 1-D from a 1-D batched dist.
                samples[mask_remaining] = self.lower_bound.view(-1)[mask_remaining.view(-1)] # This is not general
                # A better approach for batched `lower_bound` and `mask_remaining`:
                # Create a temporary tensor filled with `lower_bound` and mask it.
                temp_lower_bounds = self.lower_bound.expand(full_sample_shape)
                samples[mask_remaining] = temp_lower_bounds[mask_remaining]

        return samples


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncatedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        
        new.base_distribution = self.base_distribution.expand(batch_shape)
        
        new.lower_bound = self.lower_bound.expand(batch_shape)
        new.upper_bound = self.upper_bound.expand(batch_shape)

        cdf_upper = new.base_distribution.cdf(new.upper_bound)
        cdf_lower = new.base_distribution.cdf(new.lower_bound)
        new.normalization_constant = (cdf_upper - cdf_lower).clamp_min(torch.finfo(cdf_upper.dtype).tiny)
        new.log_normalization_constant = new.normalization_constant.log()
        
        super(TruncatedDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args 
        return new
