
import torch
from model import HiggsClassifier

# Create the model
model = HiggsClassifier()

# Create some dummy input data (5 events, 41 features each)
sample_input = torch.randn(5, 41)

# Pass the data through the model
output = model(sample_input)

print("Output shape:", output.shape)
print("Output values:", output)
