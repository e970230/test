import pywt
import numpy as np

# 模擬一個訊號
t = np.linspace(0, 1, 1024)
signal = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

# 小波包分解：使用 db4, 分解3層
wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=3)

# 顯示所有節點的 key 與其對應的訊號
for node in wp.get_level(3, order='freq'):
    print(f"Node: {node.path}, Data length: {len(node.data)}")

# 取某個特定子頻帶訊號（例如節點 'aad'）
subband = wp['aad'].data


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, node in enumerate(wp.get_level(3, order='freq')):
    plt.subplot(8, 1, i + 1)
    plt.plot(node.data)
    plt.title(f"Node: {node.path}")
plt.tight_layout()
plt.show()
