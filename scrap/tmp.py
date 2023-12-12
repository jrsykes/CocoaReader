#%%
import torch
import torch
import torchvision.models as models

# Load the pretrained ResNet-18 model
model = models.resnet18(pretrained=True)

# Define a toy input tensor
input_tensor = torch.randn(4, 3, 224, 224)

# Forward the input through the network up to the last convolutional layer
with torch.no_grad():  # Assuming you're just doing inference
    # Layer 4 is the last layer before the avgpool and fc layers in ResNet-18
    conv_out = model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(input_tensor))))))))

# Apply Adaptive Average Pooling
avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
avgpool_out = avgpool(conv_out)
#%%

print(conv_out.shape)
print(avgpool_out.shape)
# %%
