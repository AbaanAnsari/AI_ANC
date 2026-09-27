import json

with open("experiments/phase2_baseline/history.json", "r") as f:
    history = json.load(f)

print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train Enh':<10} | {'Val Enh':<10} | {'Train Cls':<10} | {'Val Cls':<10} | {'LR':<8} | {'Val Acc':<8} | {'Val SNR Imp':<12} | {'Enh/Clean RMS':<14}")
print("-" * 125)
for e in history:
    print(f"{e['epoch']:<5d} | {e['train_total_loss']:<10.6f} | {e['val_total_loss']:<10.6f} | {e['train_enhancement_loss']:<10.6f} | {e['val_enhancement_loss']:<10.6f} | {e['train_classification_loss']:<10.6f} | {e['val_classification_loss']:<10.6f} | {e['learning_rate']:<8.1e} | {e['val_classification_accuracy']*100:<7.2f}% | {e['val_snr_improvement_db']:<10.2f} dB | {e['sample_enhanced_clean_rms_ratio']:<14.6f}")
