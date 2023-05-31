
import torch

class CrossTalkColorGrading(torch.nn.Module):
    def __init__(self, matrix=None):
        super().__init__()
        self.matrix = matrix

    def forward(self, img):
        # The forward method calls the _transform method on the input image.
        return self._transform(img)

    def _transform(self, img):
        # If no matrix was provided during initialization, create a random 3x3 matrix.
        if self.matrix is None:
            matrix = torch.randn(3, 3)
        else:
            matrix = self.matrix

        # Convert the image to a tensor and add an extra dimension at the beginning.
        img_tensor = img.unsqueeze(0)

        # Apply the color grading
        # Permute the dimensions of the image tensor to (batch_size, height, width, channels).
        img_tensor = img_tensor.permute(0, 2, 3, 1)
        # Apply the color grading by performing a matrix multiplication with the color grading matrix.
        img_tensor = img_tensor @ matrix
        # Permute the dimensions back to (batch_size, channels, height, width).      
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        # Clip the values to be between 0 and 1
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Remove the extra dimension added at the beginning and return the image tensor.
        img = img_tensor.squeeze(0)

        return img


