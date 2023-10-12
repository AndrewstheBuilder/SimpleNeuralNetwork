import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class NNVisualizer:
    def __init__(self, ncols=5):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))
        plt.tight_layout()
        plt.ion()

    def draw_toy_heatmap(self):
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))

        for i, ax in enumerate(axes):
            W1 = np.random.randn(9,1)
            # Creating a heatmap for W1
            sns.heatmap(W1, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            ax.set_title("Heatmap of W1", fontsize=16)
            ax.set_xlabel("Input Neuron")
            if i == 0:
                ax.set_ylabel("Hidden Layer Neuron")

        plt.show()

    def draw_heatmap(self, Ws, Bs):
        for i, (W, B) in enumerate(zip(Ws, Bs)):
            ax_W = self.axes[2*i]
            ax_B = self.axes[2*i + 1]

            # Clear the current content of the axes
            ax_W.clear()
            ax_B.clear()

            # Creating a heatmap for W
            sns.heatmap(W, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_W, cbar=False)
            ax_W.set_title(f"Heatmap of W{i+1}", fontsize=16)
            ax_W.set_xlabel("Input Neuron")
            if i == 0:
                ax_W.set_ylabel("Hidden Layer Neuron")

            # Creating a heatmap for B
            sns.heatmap(np.atleast_2d(B), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_B, cbar=False)
            ax_B.set_title(f"Heatmap of B{i+1}", fontsize=16)
            ax_B.set_xlabel("Bias")
            ax_B.yaxis.set_visible(False)  # Hide y-axis for bias heatmap


        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Add a short pause to allow the plot to update

