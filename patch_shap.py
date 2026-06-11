import re

with open('02_Baseline_CNN_Transformer.py', 'r') as f:
    content = f.read()

target = """    shap_vals = exp_data_grad.grad.cpu().numpy()
    print("✅ Gradient-based importance computed!")

# %%
# Analisis per-lead importance
lead_imp = np.abs(shap_vals).mean(axis=(0, 2))
lead_imp_df = pd.DataFrame({'Lead': LEADS, 'Importance': lead_imp}).sort_values('Importance', ascending=False)

print(f"\\n{'Lead':<8} {'Importance':<12}")
print("─" * 20)
for _, row in lead_imp_df.iterrows():
    bar = '█' * int(row['Importance'] / lead_imp_df['Importance'].max() * 30)
    print(f"{row['Lead']:<8} {row['Importance']:<12.6f} {bar}")

# %% [markdown]
# ---
# ## 12. Ringkasan & Simpan Hasil"""

replacement = """    shap_vals = exp_data_grad.grad.cpu().numpy()
    print("✅ Gradient-based importance computed!")

# %%
# Analisis per-lead importance (Bar Chart Sederhana)
lead_imp = np.abs(shap_vals).mean(axis=(0, 2))
lead_imp_df = pd.DataFrame({'Lead': LEADS, 'Importance': lead_imp}).sort_values('Importance', ascending=False)

print(f"\\n{'Lead':<8} {'Importance':<12}")
print("─" * 20)
for _, row in lead_imp_df.iterrows():
    bar = '█' * int(row['Importance'] / lead_imp_df['Importance'].max() * 30)
    print(f"{row['Lead']:<8} {row['Importance']:<12.6f} {bar}")

# %%
# Visualisasi SHAP Beeswarm Plot (Summary Plot)
print("\\n📊 Generating SHAP Beeswarm Summary Plot...")

# Untuk merangkum fitur time-series (N, 12, L) menjadi (N, 12) untuk beeswarm plot:
# 1. SHAP Impact (X-axis): Total net impact dari lead tersebut pada prediksi (sum over time)
shap_vals_2d = shap_vals.sum(axis=2) 

# 2. Feature Value (Color): Kekuatan sinyal aktual dari lead tersebut (menggunakan standar deviasi / amplitudo)
exp_data_np = exp_data.cpu().numpy()
features_2d = np.std(exp_data_np, axis=2)

fig = plt.figure(figsize=(12, 8))
# Tampilkan beeswarm plot dari library SHAP
shap.summary_plot(shap_vals_2d, features=features_2d, feature_names=LEADS, show=False)
plt.title(f'SHAP Summary Plot (Beeswarm) - {best_model_name}\\n', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 12. Ringkasan & Simpan Hasil"""

if target in content:
    with open('02_Baseline_CNN_Transformer.py', 'w') as f:
        f.write(content.replace(target, replacement))
    print("Patch applied successfully.")
else:
    print("Target string not found!")

