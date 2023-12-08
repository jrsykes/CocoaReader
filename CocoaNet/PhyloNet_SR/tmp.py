import torch
import torch.nn as nn

class PolynomialLayer2D(nn.Module):
    def __init__(self, degree):
        super(PolynomialLayer2D, self).__init__()
        # Initialize coefficients for the polynomial of the given degree
        self.coefficients = nn.Parameter(torch.randn(degree + 1))

    def forward(self, x):
        # Ensure input x is a 2D tensor
        if x.ndim != 2:
            raise ValueError("Input must be a 2D tensor")

        # Create a tensor of powers of x
        # [x^0, x^1, x^2, ..., x^n]
        powers = torch.stack([x ** i for i in range(len(self.coefficients))], dim=-1)
        
        # Apply the polynomial transformation
        # This multiplies each coefficient with the corresponding power of x
        # and sums across the last dimension to get the final output
        y = torch.sum(self.coefficients * powers, dim=-1)
        return y

# Example usage
degree = 2  # For a quadratic polynomial
poly_layer = PolynomialLayer2D(degree)

# Example 2D tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = poly_layer(x)

print(y)
