import json

with open('experiments/phase2_step6_targeted_crm_full/history.json') as f:
    history = json.load(f)

print(f"{'Epoch':<5} | {'TrainLoss':<9} | {'ValLoss':<9} | {'ValEnh':<8} | {'ValMask':<8} | {'ValRec':<8} | {'ValEn':<8} | {'ValCls':<8} | {'Acc%':<6} | {'C0%':<5} | {'C1%':<5} | {'C2%':<5} | {'SNR_imp':<8} | {'Ratio':<7} | {'LR':<8} | {'Dur':<5}")
print("-" * 140)

for r in history:
    ep = r["epoch"]
    tl = r["train_total_loss"]
    vl = r["val_total_loss"]
    ve = r["val_enhancement_loss"]
    vm = r["val_mask_loss"]
    vr = r["val_reconstruction_loss"]
    vn = r["val_energy_loss"]
    vc = r["val_classification_loss"]
    acc = r["val_classification_accuracy"] * 100
    c0 = r["val_per_class_accuracy"]["0"] * 100
    c1 = r["val_per_class_accuracy"]["1"] * 100
    c2 = r["val_per_class_accuracy"]["2"] * 100
    snr = r["val_snr_improvement_db"]
    ratio = r["sample_enhanced_clean_rms_ratio"]
    lr = r["learning_rate"]
    dur = r["epoch_duration_seconds"]
    print(f"{ep:<5d} | {tl:<9.6f} | {vl:<9.6f} | {ve:<8.6f} | {vm:<8.6f} | {vr:<8.6f} | {vn:<8.6f} | {vc:<8.6f} | {acc:<6.2f} | {c0:<5.1f} | {c1:<5.1f} | {c2:<5.1f} | {snr:<8.2f} | {ratio:<7.4f} | {lr:<8.2e} | {dur:<5.1f}")
