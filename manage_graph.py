import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

clear = True
join_table = True

if clear:
    x = pd.DataFrame({'train_scores': []})
    x.to_csv('train_main.csv', index=False)
    y = pd.DataFrame({'a_loss': [],
                      'c_loss': [],
                      'total_loss': []
                      })
    y.to_csv('loss_main.csv', index=False)


train_main = pd.read_csv('train_main.csv')
train_buf = pd.read_csv('train_buf.csv')

loss_main = pd.read_csv('loss_main.csv')
loss_buf = pd.read_csv('loss_buf.csv')

a_loss = pd.read_csv('loss_main.csv', usecols=[0])
c_loss = pd.read_csv('loss_main.csv', usecols=[1])

if join_table:
    frame = [train_main, train_buf]
    joined = pd.concat(frame)
    joined.to_csv('train_main.csv', index=False)
    train_main = pd.read_csv('train_main.csv')

    frame = [loss_main, loss_buf]
    joined = pd.concat(frame)
    joined.to_csv('loss_main.csv', index=False)
    a_loss = pd.read_csv('loss_main.csv', usecols=[0])
    c_loss = pd.read_csv('loss_main.csv', usecols=[1])


plt.figure()
plt.plot(np.arange(len(train_main)), train_main)
plt.xlabel('episodes')
plt.ylabel('Total moving reward')

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(len(a_loss)), a_loss)
plt.plot(np.arange(len(a_loss)), np.zeros(len(a_loss)))
plt.xlabel('step')
plt.ylabel('Actor loss')

plt.subplot(2,1,2)
plt.plot(np.arange(len(c_loss)), c_loss)
plt.plot(np.arange(len(c_loss)), np.zeros(len(a_loss)))
plt.xlabel('step')
plt.ylabel('Critic loss')
plt.show()
