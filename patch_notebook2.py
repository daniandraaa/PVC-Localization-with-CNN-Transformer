import re

with open('02_Baseline_CNN_Transformer.py', 'r') as f:
    content = f.read()

# Find the start of Section 8
idx_sec8 = content.find('# ## 8. Perbandingan 3 Model')
if idx_sec8 == -1:
    print("Section 8 not found!")
    exit(1)

content_before = content[:idx_sec8]

new_content = """# ## 8. Perbandingan 3 Model — K-Fold Results

# %%
print("=" * 70)
print("📊 PERBANDINGAN K-FOLD CV — 3 MODEL TRANSFORMER")
print("=" * 70)

comparison_data = []
for model_name in MODEL_NAMES:
    df_r = pd.DataFrame(all_results[model_name])
    row = {'Model': model_name}
    for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']:
        row[f'{m}_mean'] = df_r[m].mean()
        row[f'{m}_std'] = df_r[m].std()
    
    # Calculate average best epoch for final training
    avg_best_epoch = int(np.round(df_r['best_epoch'].mean()))
    row['avg_best_epoch'] = avg_best_epoch
    
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# Print table
metrics_list = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
print(f"\\n{'Model':<28}", end="")
for m in metrics_list:
    print(f" {m.capitalize():<16}", end="")
print(f" {'AvgBestEp':<10}")
print("─" * 120)
for _, row in df_comparison.iterrows():
    print(f"{row['Model']:<28}", end="")
    for m in metrics_list:
        print(f" {row[f'{m}_mean']:.3f}±{row[f'{m}_std']:.3f}     ", end="")
    print(f" {row['avg_best_epoch']:<10.0f}")

# Best model
best_model_name = df_comparison.loc[df_comparison['f1_mean'].idxmax(), 'Model']
best_avg_epoch = df_comparison.loc[df_comparison['f1_mean'].idxmax(), 'avg_best_epoch']
best_idx = MODEL_NAMES.index(best_model_name)
print(f"\\n🏆 Best Model (by F1): {best_model_name} (Avg Best Epoch: {best_avg_epoch})")

# %%
# Visualisasi perbandingan Training vs Validation selama K-Fold
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
fig.suptitle('K-Fold Cross Validation: Training vs Validation Accuracy', fontsize=18, fontweight='bold', y=1.05)

for i, model_name in enumerate(MODEL_NAMES):
    ax = axes[i]
    color = MODEL_COLORS[i]
    
    # Ambil history dari 5 fold
    histories = all_histories[model_name]
    max_len = max(len(h['val_acc']) for h in histories)
    
    # Pad array
    val_accs = np.array([h['val_acc'] + [h['val_acc'][-1]]*(max_len - len(h['val_acc'])) for h in histories])
    
    mean_val = np.mean(val_accs, axis=0)
    std_val = np.std(val_accs, axis=0)
    
    epochs = np.arange(max_len)
    ax.plot(epochs, mean_val, color=color, linewidth=2, label='Validation Acc (Mean)')
    ax.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2, color=color)
    
    ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'kfold_train_val_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 9. Final Model Training
# 
# **PENTING:** Melatih ulang model terbaik ({best_model_name}) menggunakan **seluruh 80% data training** sebanyak rata-rata best epoch ({best_avg_epoch}), tanpa validation split lagi.

# %%
print("=" * 70)
print(f"🔥 FINAL MODEL TRAINING ({best_model_name})")
print(f"   Menggunakan 100% dari data training (N={len(X_train_scaled)})")
print(f"   Target Epochs: {best_avg_epoch}")
print("=" * 70)

# Initialize best model architecture fresh
final_model = model_classes[best_idx](CONFIG).to(DEVICE)
criterion_final = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer_final = optim.AdamW(final_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

# DataLoader untuk seluruh data training 80%
final_train_loader = DataLoader(
    ECGDataset(X_train_scaled, y_train_all), 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    drop_last=True
)

final_history = {'train_loss': [], 'train_acc': []}

for epoch in range(best_avg_epoch):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(final_model, final_train_loader, criterion_final, optimizer_final, DEVICE)
    
    final_history['train_loss'].append(tr_loss)
    final_history['train_acc'].append(tr_acc)
    
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == best_avg_epoch - 1:
        print(f"   Ep {epoch+1:3d}/{best_avg_epoch} | TrLoss {tr_loss:.4f} | TrAcc {tr_acc:.3f} | {time.time()-t0:.1f}s")

print("\\n✅ Final Model Training Selesai!")

# Visualisasi Final Training
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(final_history['train_acc'], color='#059669', label='Train Accuracy', linewidth=2)
ax.set_title(f'Final Training Progress - {best_model_name}', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 10. Final Testing (20% DWT Data)
# 
# Menggunakan **Final Model** yang baru saja dilatih pada 20% data testing yang tidak pernah dilihat sebelumnya.

# %%
print("=" * 70)
print(f"🧪 FINAL TESTING — 20% DWT DATA ({len(X_test_scaled)} segments)")
print("=" * 70)

final_model.eval()

# Test DataLoader
test_loader = DataLoader(ECGDataset(X_test_scaled, y_test_all), batch_size=CONFIG['batch_size'], shuffle=False)

# Evaluasi tingkat segmen
test_metrics, test_preds, test_probs, test_labels = evaluate(
    final_model, test_loader, nn.CrossEntropyLoss(), DEVICE
)

print("\\n📊 Segment-Level Metrics:")
for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']:
    print(f"   {m.capitalize():<12}: {test_metrics[m]:.3f}")

# %%
# Patient-level majority voting
print("\\n📊 Patient-Level Majority Voting:")

patient_preds = {}
patient_probs = {}
patient_true = {}

for i, hid in enumerate(groups_test):
    if hid not in patient_preds:
        patient_preds[hid] = []
        patient_probs[hid] = []
        patient_true[hid] = test_labels[i]
    patient_preds[hid].append(test_preds[i])
    patient_probs[hid].append(test_probs[i])

patient_final_preds = {}
patient_final_probs = {}
for hid in patient_preds:
    votes = np.array(patient_preds[hid])
    patient_final_preds[hid] = int(np.round(votes.mean()))
    patient_final_probs[hid] = np.mean(patient_probs[hid])

y_true_patient = np.array([patient_true[hid] for hid in patient_preds])
y_pred_patient = np.array([patient_final_preds[hid] for hid in patient_preds])
y_prob_patient = np.array([patient_final_probs[hid] for hid in patient_preds])

patient_metrics = compute_metrics(y_true_patient, y_pred_patient, y_prob_patient)

print(f"   Total test patients: {len(y_true_patient)}")
for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']:
    print(f"   {m.capitalize():<12}: {patient_metrics[m]:.3f}")

print(f"\\n📋 Classification Report (Patient-Level):")
print(classification_report(y_true_patient, y_pred_patient, target_names=le.classes_))

# %%
# Visualisasi Metrik Akhir
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle(f'Final Test Results — {best_model_name} (20% DWT Data)', fontsize=18, fontweight='bold')

# Confusion Matrix
ax = axes[0]
cm = confusion_matrix(y_true_patient, y_pred_patient)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_,
            annot_kws={'size': 18}, cbar=False)
ax.set_title('Confusion Matrix (Patient-Level)', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)

# ROC Curve
ax = axes[1]
fpr, tpr, thresholds = roc_curve(y_true_patient, y_prob_patient)
ax.plot(fpr, tpr, color='#2563EB', linewidth=2.5, label=f'AUC = {patient_metrics["auc"]:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax.fill_between(fpr, tpr, alpha=0.1, color='#2563EB')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('1 - Specificity (False Positive Rate)')
ax.set_ylabel('Sensitivity (True Positive Rate)')
ax.legend(fontsize=12, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)

# Bar chart metrik
ax = axes[2]
m_names = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
m_keys = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
m_vals = [patient_metrics[k] for k in m_keys]
colors_m = ['#2563EB', '#059669', '#D97706', '#DC2626', '#7C3AED']

bars = ax.bar(m_names, m_vals, color=colors_m, edgecolor='white', linewidth=2)
for bar, val in zip(bars, m_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.spines[['top', 'right']].set_visible(False)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'final_test_results.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 11. SHAP Explainability (Model Utama)
# 
# Menggunakan SHAP GradientExplainer untuk analisis interpretability pada model Final.

# %%
import shap

print("=" * 70)
print(f"🔍 SHAP EXPLAINABILITY ANALYSIS ({best_model_name})")
print("=" * 70)

np.random.seed(SEED)
bg_idx = np.random.choice(len(X_train_scaled), size=min(100, len(X_train_scaled)), replace=False)
background = torch.FloatTensor(X_train_scaled[bg_idx]).to(DEVICE)

n_exp = min(50, len(X_test_scaled))
exp_idx = np.random.choice(len(X_test_scaled), size=n_exp, replace=False)
exp_data = torch.FloatTensor(X_test_scaled[exp_idx]).to(DEVICE)
exp_labels = y_test_all[exp_idx]

print("\\n⏳ Computing SHAP values (GradientExplainer)...")
final_model.eval()

try:
    explainer = shap.GradientExplainer(final_model, background)
    shap_values = explainer.shap_values(exp_data)
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    print("✅ SHAP computed successfully!")
except Exception as e:
    print(f"⚠️ SHAP GradientExplainer failed: {e}")
    print("   Falling back to manual gradient-based importance...")
    exp_data_grad = exp_data.clone().requires_grad_(True)
    out = final_model(exp_data_grad)
    out[:, 1].sum().backward()
    shap_vals = exp_data_grad.grad.cpu().numpy()
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
# ## 12. Ringkasan & Simpan Hasil

# %%
print("\\n" + "=" * 70)
print("💾 MENYIMPAN HASIL")
print("=" * 70)

df_comparison.to_csv(OUTPUT_DIR / 'kfold_comparison.csv', index=False)
print(f"   ✅ K-Fold comparison: {OUTPUT_DIR / 'kfold_comparison.csv'}")

test_res_df = pd.DataFrame([
    {'level': 'segment', **test_metrics},
    {'level': 'patient', **patient_metrics}
])
test_res_df.to_csv(OUTPUT_DIR / 'final_test_results.csv', index=False)
print(f"   ✅ Final test results: {OUTPUT_DIR / 'final_test_results.csv'}")

lead_imp_df.to_csv(OUTPUT_DIR / 'lead_importance.csv', index=False)
print(f"   ✅ Lead importance: {OUTPUT_DIR / 'lead_importance.csv'}")

with open(OUTPUT_DIR / 'config.json', 'w') as f:
    json.dump({k: str(v) if isinstance(v, list) else v for k, v in CONFIG.items()}, f, indent=2)

final_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
torch.save(final_state, OUTPUT_DIR / f'final_model_{best_model_name.replace(" ", "_")}.pth')
print(f"   ✅ Final model saved.")

# %%
print("\\n" + "=" * 70)
print("📋 RINGKASAN NOTEBOOK 02: BASELINE CNN-TRANSFORMER")
print("=" * 70)

print(f\"\"\"
🔬 PIPELINE:
   DATA DWT → 80/20 Split → 5-Fold CV (Penguat) → 100% Training Model Final → Test (20% DWT) → SHAP

🏆 PERBANDINGAN MODEL K-FOLD (F1-Score):
   1. {MODEL_NAMES[0]:<25}: {df_comparison.loc[0, 'f1_mean']:.3f} ± {df_comparison.loc[0, 'f1_std']:.3f}
   2. {MODEL_NAMES[1]:<25}: {df_comparison.loc[1, 'f1_mean']:.3f} ± {df_comparison.loc[1, 'f1_std']:.3f}
   3. {MODEL_NAMES[2]:<25}: {df_comparison.loc[2, 'f1_mean']:.3f} ± {df_comparison.loc[2, 'f1_std']:.3f}

🏅 FINAL MODEL: {best_model_name} (Dilatih ulang selama {best_avg_epoch} epochs)

🧪 FINAL TEST (20% DWT Data) — Patient Level:
   • Accuracy   : {patient_metrics['accuracy']:.3f}
   • Sensitivity: {patient_metrics['sensitivity']:.3f}
   • Specificity: {patient_metrics['specificity']:.3f}
   • F1-Score   : {patient_metrics['f1']:.3f}
   • AUC-ROC    : {patient_metrics['auc']:.3f}

🔍 SHAP TOP-3 LEADS: {', '.join(lead_imp_df.head(3)['Lead'].values)}

🔜 LANGKAH SELANJUTNYA:
   → Notebook 3: Fine-Tuning (GAN augmentation → Grid Search → CNN-Transformer)
\"\"\")

print("✅ Notebook 02 Baseline CNN-Transformer (2-Stage Training) SELESAI!")
"""

with open('02_Baseline_CNN_Transformer.py', 'w') as f:
    f.write(content_before + new_content)

print("Patch applied successfully.")
