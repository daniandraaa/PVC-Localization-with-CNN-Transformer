import numpy as np

# Simulate what happened:
shap_vals = np.zeros((5, 12, 100, 2))
print("Original shape:", shap_vals.shape)

if len(shap_vals.shape) == 4:
    shap_vals = shap_vals[..., 1] # Select class 1
print("Selected shape:", shap_vals.shape)

lead_imp = np.abs(shap_vals).mean(axis=(0, 2))
print("lead_imp shape:", lead_imp.shape)
