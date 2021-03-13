import matplotlib.pyplot as plt

plt.figure(1, figsize=(3,3))
ax = plt.subplot(111)

ann = ax.annotate("Test",
                  xy=(0.2, 0.2), xycoords='data',
                  xytext=(0.8, 0.8), textcoords='data',
                  size=10, va="center", ha="center",
                  
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=0.3",
                                  fc="w"), 
                  )

plt.show()

