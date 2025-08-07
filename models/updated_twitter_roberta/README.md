# Calibrated Twitter RoBERTa Model

This model has been calibrated using:

1. **Monte Carlo Dropout**
   - Keeps dropout active during inference
   - Uses 5 forward passes and averages results
   - Reduces prediction variance

2. **Temperature Scaling**
   - Uses temperature T=1.5
   - Softens probability distributions
   - Improves calibration without changing prediction order

## Usage Example

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load base model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("path_to_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_model")

# Apply MC Dropout
class MC_DropoutWrapper(nn.Module):
    def __init__(self, model, num_forward_passes=5):
        super(MC_DropoutWrapper, self).__init__()
        self.model = model
        self.num_forward_passes = num_forward_passes
        
        # Enable dropout layers in eval mode
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, input_ids, attention_mask):
        outputs = []
        for _ in range(self.num_forward_passes):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            outputs.append(output.logits)
        return torch.stack(outputs).mean(dim=0)

# Apply Temperature Scaling
class TemperatureScaling(nn.Module):
    def __init__(self, model, temp=1.5):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = temp
    
    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return logits / self.temperature

# Apply calibration techniques
model.eval()
model = MC_DropoutWrapper(model)
model = TemperatureScaling(model)

# Example inference
inputs = tokenizer("Example text", return_tensors="pt")
outputs = model(**inputs)
```
