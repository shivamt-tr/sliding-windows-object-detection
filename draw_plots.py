# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 01:01:51 2022

@author: tripa
"""

train_loss_history = [66.5478, 50.1597, 40.6667, 35.2459, 31.5966, 28.9122, 26.8037, 25.1436, 24.8513, 25.1599, 25.3105, 23.7781, 22.6887, 21.1382, 21.1654]
val_loss_history = [53.5208, 39.3793, 30.2433, 24.8014, 21.7041, 19.3895, 18.2105, 17.5491, 17.8844, 17.7719, 17.4523, 16.7929, 16.1575, 16.2280, 15.4730]
train_acc_history = [0.2381, 0.4954, 0.5960, 0.6405, 0.6751, 0.6966, 0.7124, 0.7411, 0.7342, 0.7327, 0.7251, 0.7475, 0.7558, 0.7777, 0.7792]
val_acc_history = [0.5432, 0.6786, 0.7436, 0.7807, 0.7995, 0.8153, 0.8244, 0.8239, 0.8219, 0.8229, 0.8272, 0.8293, 0.8321, 0.8318, 0.8384]

# Plot loss and accuracy
fig = plt.figure(figsize=(10, 4))
rows, cols = 1, 2

fig.add_subplot(rows, cols, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss_history, color='blue', linewidth=1, label='Train')
plt.plot(val_loss_history, color='red', linewidth=1, label='Val')
plt.legend()

fig.add_subplot(rows, cols, 2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc_history, color='blue', linewidth=1, label='Train')
plt.plot(val_acc_history, color='red', linewidth=1, label='Val')
plt.legend()

plt.show()