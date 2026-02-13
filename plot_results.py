import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. 数据准备 (基于 Results.md) ---
data = {
    "Method": ["Original", "PPD10", "PPD20", "PPD30", "PPD40", "PPD44", "WPD10_10", "WPD20_20", "WPD30_30", "WPD40_40", "WPD44_44", "WPD10_30_1", "WPD10_30_2", "WPD10_30_5", "WPD10_30_10", "WPD20_40_0.5", "WPD20_40_1", "WPD20_40_1.5", "WPD20_40_2", "WPD20_40_5", "WPD20_40_10"],
    "mIoU": [32.61, 10.62, 17.47, 23.44, 27.01, 29.10, 18.50, 28.46, 29.18, 32.27, 32.21, 28.48, 26.35, 23.05, 21.07, 31.33, 31.08, 30.86, 30.92, 30.16, 29.72],
    "AS": [0.9791, 1.0641, 1.0472, 1.0319, 1.0202, 1.0105, 1.0560, 1.0387, 1.0225, 0.9964, 0.9965, 1.0165, 1.0198, 1.0278, 1.0373, 1.0014, 1.0036, 1.0047, 1.0057, 1.0106, 1.0155],
    "SSIM": [0.9143, 0.7937, 0.8314, 0.8635, 0.8841, 0.8953, 0.8323, 0.8913, 0.8947, 0.9107, 0.9106, 0.8847, 0.8750, 0.8570, 0.8458, 0.9022, 0.9007, 0.8998, 0.8999, 0.8976, 0.8949]
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
    plt.plot(ppd['AS'], ppd['SSIM'], 'o-', color='#3498db', label='PPD (Global Fourier)', linewidth=2)
    plt.plot(wpd['AS'], wpd['SSIM'], 's--', color='#27ae60', label='WPD (Local Wavelet)', linewidth=2)
    plt.scatter(df.loc[0, 'AS'], df.loc[0, 'SSIM'], c='red', marker='*', s=250, label='Original (Benchmark)', zorder=5)

    # 仅标注端点半径，移除箭头
    plt.text(1.066, 0.79, '$r=10$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.0, 0.89, '$r=44$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.058, 0.83, '$r=10$', color='#27ae60', fontweight='bold', fontsize=10)
    plt.text(0.994, 0.902, '$r=44$', color='#27ae60', fontweight='bold', fontsize=10)

    plt.xlabel('AS (Realism)'); plt.ylabel('SSIM (Structure Alignment)')
    plt.title('Performance Trade-off Frontier')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='lower left')
    plt.xlim(0.97, 1.075); plt.ylim(0.77, 0.92)
    plt.tight_layout(); plt.savefig('ablation1a_tradeoff.pdf'); plt.savefig('ablation1a_tradeoff.jpg')

    plt.figure(figsize=(6, 6))
    ppd = df[df['Method'].str.startswith('PPD')].sort_values('AS', ascending=False)
    wpd = df[df['Method'].str.match(r'^WPD\d+_\d+$')].sort_values('AS', ascending=False)
    plt.plot(ppd['AS'], ppd['mIoU'], 'o-', color='#3498db', label='PPD (Global Fourier)', linewidth=2)
    plt.plot(wpd['AS'], wpd['mIoU'], 's--', color='#27ae60', label='WPD (Local Wavelet)', linewidth=2)
    plt.scatter(df.loc[0, 'AS'], df.loc[0, 'mIoU'], c='red', marker='*', s=250, label='Original (Benchmark)', zorder=5)

    # 仅标注端点半径，移除箭头
    plt.text(1.066, 12, '$r=10$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.0, 34, '$r=44$', color='#3498db', fontweight='bold', fontsize=10)
    plt.text(1.058, 22, '$r=10$', color='#27ae60', fontweight='bold', fontsize=10)
    plt.text(0.994, 36.5, '$r=44$', color='#27ae60', fontweight='bold', fontsize=10)

    plt.xlabel('AS (Realism)'); plt.ylabel('mIoU (Semantic Consistency)')
    plt.title('Performance Trade-off Frontier')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='lower left')
    plt.xlim(0.97, 1.075); plt.ylim(10, 40)
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