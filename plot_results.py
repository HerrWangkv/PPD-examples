import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. 数据准备 (基于 Results.md) ---
data = {
    "Method": ["Original", "PPD10", "PPD20", "PPD30", "PPD40", "PPD44", "WPD10_10", "WPD20_20", "WPD30_30", "WPD40_40", "WPD44_44", "WPD10_30_1", "WPD10_30_2", "WPD10_30_5", "WPD10_30_10", "WPD20_40_0.5", "WPD20_40_1", "WPD20_40_1.5", "WPD20_40_2", "WPD20_40_5", "WPD20_40_10", "SDEdit_0.2", "SDEdit_0.4", "SDEdit_0.6", "SDEdit_0.7", "SDEdit_0.8", "ControlNet_0.4", "ControlNet_0.5", "ControlNet_0.6", "ControlNet_0.8"],
    "mIoU": [32.61, 10.62, 17.47, 23.44, 27.01, 29.10, 18.50, 28.46, 29.18, 32.27, 32.21, 28.48, 26.35, 23.05, 21.07, 31.33, 31.08, 30.86, 30.92, 30.16, 29.72, 32.34, 27.36, 18.35, 12.67, 9.04, 9.90, 29.51, 32.31, 31.00],
    "AS": [0.9791, 1.0641, 1.0472, 1.0319, 1.0202, 1.0105, 1.0560, 1.0387, 1.0225, 0.9964, 0.9965, 1.0165, 1.0198, 1.0278, 1.0373, 1.0014, 1.0036, 1.0047, 1.0057, 1.0106, 1.0155, 0.9808, 0.9820, 1.0065, 1.0492, 1.0757, 1.0607, 0.9964, 0.9861, 0.9842],
    "SSIM": [0.9143, 0.7937, 0.8314, 0.8635, 0.8841, 0.8953, 0.8323, 0.8913, 0.8947, 0.9107, 0.9106, 0.8847, 0.8750, 0.8570, 0.8458, 0.9022, 0.9007, 0.8998, 0.8999, 0.8976, 0.8949, 0.9076, 0.8888, 0.8456, 0.8074, 0.7859, 0.8001, 0.8950, 0.9121, 0.9145]
}
for i in range(len(data["mIoU"])):
    data["mIoU"][i] = data["mIoU"][i] * 19 / 16 # Ignore 3 classes with 0 IoU (terrain, truck, train)

df = pd.DataFrame(data)

# --- 2. 论文专用样式配置 ---
plt.rcParams.update({
    "font.family": "serif", 
    "axes.labelsize": 12, 
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10, 
    "legend.fontsize": 10, 
    "axes.titlesize": 14,
    "savefig.dpi": 300
})

# --- Figure 1: Performance Trade-off Frontier (仅标注端点) ---
def plot_fig1():
    plt.figure(figsize=(6, 6))
    ppd = df[df['Method'].str.startswith('PPD')].sort_values('AS', ascending=False)
    wpd = df[df['Method'].str.match(r'^WPD\d+_\d+$')].sort_values('AS', ascending=False)
    sde = df[df['Method'].str.startswith('SDEdit') & ~df['Method'].str.endswith('_0.8')].sort_values('AS', ascending=False)
    controlnet = df[df['Method'].str.startswith('ControlNet') & ~df['Method'].str.endswith('_0.8')].sort_values('AS', ascending=False)
    
    plt.plot(wpd['AS'], wpd['SSIM'], 'o-', color='#27ae60', label="Ours", linewidth=2)
    plt.plot(ppd['AS'], ppd['SSIM'], 's--', color='#3498db', label="NeuralRemaster", linewidth=2)
    plt.plot(controlnet['AS'], controlnet['SSIM'], 'd:', color="#f39c12", label="ControlNet-Tile", linewidth=2)
    plt.plot(sde['AS'], sde['SSIM'], '^-.', color="#e73cd9", label="SDEdit", linewidth=2)
    plt.scatter(df.loc[0, 'AS'], df.loc[0, 'SSIM'], c='red', marker='*', s=250, label="Input", zorder=5)

    # 仅标注端点半径，移除箭头
    plt.text(1.066, 0.795, '$r=10$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.0, 0.895, '$r=44$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.058, 0.83, '$r=10$', color='#27ae60', fontweight='bold', fontsize=10)
    plt.text(0.994, 0.913, '$r=44$', color='#27ae60', fontweight='bold', fontsize=10)
    plt.text(0.970, 0.902, '$t_0=0.2$', color="#e73cd9", fontweight='bold', fontsize=10)
    plt.text(1.035, 0.805, '$t_0=0.7$', color='#e73cd9', fontweight='bold', fontsize=10)
    plt.text(0.982, 0.915, '$w=0.6$', color='#f39c12', fontweight='bold', fontsize=10)
    plt.text(1.048, 0.795, '$w=0.4$', color='#f39c12', fontweight='bold', fontsize=10)

    plt.xlabel('AS (Realism)'); plt.ylabel('SSIM (Structure Alignment)')
    plt.title('Performance Trade-off Frontier')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='lower left')
    plt.xlim(0.97, 1.08); plt.ylim(0.77, 0.92)
    plt.tight_layout(); plt.savefig('ablation1a_tradeoff.pdf'); plt.savefig('ablation1a_tradeoff.jpg')

    plt.figure(figsize=(6, 6))
    plt.plot(wpd['AS'], wpd['mIoU'], 'o-', color='#27ae60', label="Ours", linewidth=2)
    plt.plot(ppd['AS'], ppd['mIoU'], 's--', color='#3498db', label="NeuralRemaster", linewidth=2)
    plt.plot(controlnet['AS'], controlnet['mIoU'], 'd:', color="#f39c12", label="ControlNet-Tile", linewidth=2)
    plt.plot(sde['AS'], sde['mIoU'], '^-.', color='#e73cd9', label="SDEdit", linewidth=2)
    plt.scatter(df.loc[0, 'AS'], df.loc[0, 'mIoU'], c='red', marker='*', s=250, label="Input", zorder=5)

    # 仅标注端点半径，移除箭头
    plt.text(1.066, 12.4, '$r=10$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.0, 34.5, '$r=44$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.058, 22, '$r=10$', color='#27ae60', fontweight='bold', fontsize=10)
    plt.text(0.994, 38.6, '$r=44$', color='#27ae60', fontweight='bold', fontsize=10)
    plt.text(0.970, 37.5, '$t_0=0.2$', color='#e73cd9', fontweight='bold', fontsize=10)
    plt.text(1.035, 14.2, '$t_0=0.7$', color='#e73cd9', fontweight='bold', fontsize=10)
    plt.text(0.9815, 38.8, '$w=0.6$', color='#f39c12', fontweight='bold', fontsize=10)
    plt.text(1.048, 11.5, '$w=0.4$', color='#f39c12', fontweight='bold', fontsize=10)
    plt.xlabel('AS (Realism)'); plt.ylabel('mIoU (Semantic Consistency)')
    plt.title('Performance Trade-off Frontier')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='lower left')
    plt.xlim(0.97, 1.08); plt.ylim(10, 40)
    plt.tight_layout(); plt.savefig('ablation1b_tradeoff.pdf'); plt.savefig('ablation1b_tradeoff.jpg')

# --- Figure 2: Gamma Sensitivity Analysis ---
def plot_fig2():
    gamma_data = df[(df['Method'].str.startswith('WPD20_40_')) & (df['Method'] != 'WPD20_40_1.5')].copy()
    gamma_data['gamma'] = gamma_data['Method'].str.extract('(\d+\.?\d*)$').astype(float)
    gamma_data = gamma_data.sort_values('gamma')
    fig, ax1 = plt.subplots(figsize=(7, 5)); ax2 = ax1.twinx()
    ax1.plot(gamma_data['gamma'], gamma_data['SSIM'], 'g-o', label='SSIM')
    ax2.plot(gamma_data['gamma'], gamma_data['AS'], 'b-s', label='AS')
    ax1.set_xlabel(r'Hyperparameter $\gamma$'); ax1.set_ylabel('SSIM', color='g'); ax2.set_ylabel('AS', color='b')
    plt.title(r'Impact of $\gamma$ on DAFS Mapping'); plt.grid(True, axis='x', linestyle=':')
    plt.tight_layout(); plt.savefig('ablation2a_gamma.pdf');  plt.savefig('ablation2a_gamma.jpg')

    gamma_data = df[(df['Method'].str.startswith('WPD20_40_')) & (df['Method'] != 'WPD20_40_1.5')].copy()
    gamma_data['gamma'] = gamma_data['Method'].str.extract('(\d+\.?\d*)$').astype(float)
    gamma_data = gamma_data.sort_values('gamma')
    fig, ax1 = plt.subplots(figsize=(7, 5)); ax2 = ax1.twinx()
    ax1.plot(gamma_data['gamma'], gamma_data['mIoU'], 'g-o', label='mIoU')
    ax2.plot(gamma_data['gamma'], gamma_data['AS'], 'b-s', label='AS')
    ax1.set_xlabel(r'Hyperparameter $\gamma$'); ax1.set_ylabel('mIoU', color='g'); ax2.set_ylabel('AS', color='b')
    plt.title(r'Impact of $\gamma$ on DAFS Mapping'); plt.grid(True, axis='x', linestyle=':')
    plt.tight_layout(); plt.savefig('ablation2b_gamma.pdf');  plt.savefig('ablation2b_gamma.jpg')

# 运行绘图
plot_fig1(); plot_fig2()