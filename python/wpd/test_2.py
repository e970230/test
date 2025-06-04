import pywt
import numpy as np
import matplotlib.pyplot as plt


# 模擬訊號
t = np.linspace(0, 1, 1024)
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*150*t)

# 小波包分解
wavelet = 'db4'
max_level = 5
wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)

# 儲存能量分布
layer_energy = []

for level in range(1, max_level+1):
    nodes = wp.get_level(level, order='freq')
    energies = [np.sum(np.square(node.data)) for node in nodes]
    total = sum(energies)
    norm_energies = [e / total for e in energies]  # 正規化後的能量比例
    layer_energy.append(norm_energies)

for i, layer in enumerate(layer_energy, start=1):
    plt.figure()
    plt.bar(range(len(layer)), layer)
    plt.title(f"Level {i} - Energy Distribution (Normalized)")
    plt.xlabel("Node index")
    plt.ylabel("Energy ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


energy_threshold = 0.01  # 若 node 能量占比小於 1%
min_high_energy_nodes = 2

for i, energies in enumerate(layer_energy):
    high_energy_nodes = sum(e > energy_threshold for e in energies)
    if high_energy_nodes < min_high_energy_nodes:
        print(f"Level {i+1} 已過細化，建議停止在 Level {i}")
        break

