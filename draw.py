import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 输入层
ax.text(0.5, 0.9, 'Input (Batch x Seq x Channels x H x W)', ha='center', va='center', fontsize=12)

# 添加卷积层 (CNN)
cnn_box = mpatches.FancyBboxPatch((0.2, 0.7), 0.6, 0.1, boxstyle="round,pad=0.05", ec="black", fc="lightblue", lw=2)
ax.add_patch(cnn_box)
ax.text(0.5, 0.75, 'Conv2D Layer\n(3x3 Filter)shape:(-1,channels,h,w)', ha='center', va='center', fontsize=12)

# 添加LSTM层
lstm_box = mpatches.FancyBboxPatch((0.2, 0.5), 0.6, 0.1, boxstyle="round,pad=0.05", ec="black", fc="lightgreen", lw=2)
ax.add_patch(lstm_box)
ax.text(0.5, 0.55, 'LSTM Layer shape:(bs,-1,voc)', ha='center', va='center', fontsize=12)

# 添加全连接层
fc_box = mpatches.FancyBboxPatch((0.2, 0.3), 0.6, 0.1, boxstyle="round,pad=0.05", ec="black", fc="lightcoral", lw=2)
ax.add_patch(fc_box)
ax.text(0.5, 0.35, 'Fully Connected Layer shape:(bs,1)', ha='center', va='center', fontsize=12)

# 添加箭头连接各层，从下到上
ax.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.7), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.5, 0.2), xytext=(0.5, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))

# 输出层，位置稍微往下移
ax.text(0.5, 0.18, 'Output (Prediction)', ha='center', va='center', fontsize=12)

# 设置显示
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.title('CNN-LSTM Network Architecture')
plt.show()

