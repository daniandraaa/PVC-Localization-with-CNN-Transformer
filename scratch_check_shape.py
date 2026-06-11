import torch
import torch.nn as nn
import shap
import numpy as np

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(12, 16, 3)
        self.fc = nn.Linear(16 * 98, 2)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = DummyModel()
bg = torch.randn(10, 12, 100)
exp = torch.randn(5, 12, 100)

explainer = shap.GradientExplainer(model, bg)
shap_values = explainer.shap_values(exp)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

print("Shape of shap_vals:", shap_vals.shape)

lead_imp = np.abs(shap_vals).mean(axis=(0, 2))
print("Shape of lead_imp:", lead_imp.shape)
