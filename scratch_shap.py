import numpy as np
import shap
import matplotlib.pyplot as plt

# Dummy data
N = 50
shap_vals = np.random.randn(N, 12, 1000) * 0.1 # N, 12, L
exp_data_np = np.random.randn(N, 12, 1000)

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

shap_vals_2d = shap_vals.sum(axis=2) # Net impact
features_2d = np.std(exp_data_np, axis=2) # Feature value (RMS/Std)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_2d, features=features_2d, feature_names=LEADS, show=False)
plt.savefig('scratch_shap_beeswarm.png', bbox_inches='tight', dpi=150)
print("Saved dummy SHAP plot")
