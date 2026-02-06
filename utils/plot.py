import os
import matplotlib.pyplot as plt

# ===================== Visualization =====================

def plot_training_curves(train_losses, val_losses, val_metrics, save_dir):
    """
    Plot training curves
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # S-measure
    s_measures = [m["S"] for m in val_metrics]
    axes[0, 1].plot(epochs, s_measures, label='S-measure', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('S-measure')
    axes[0, 1].set_title('S-measure (Structure Measure)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # E-measure and F-measure
    e_measures = [m["E"] for m in val_metrics]
    fw_measures = [m["Fw"] for m in val_metrics]
    axes[1, 0].plot(epochs, e_measures, label='E-measure', marker='o', color='blue')
    axes[1, 0].plot(epochs, fw_measures, label='Fw-measure', marker='s', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('E-measure and Fw-measure')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # MAE
    mae_scores = [m["MAE"] for m in val_metrics]
    axes[1, 1].plot(epochs, mae_scores, label='MAE', marker='o', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Mean Absolute Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()