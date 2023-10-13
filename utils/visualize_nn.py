import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button
import numpy as np
import tkinter as tk


class NNVisualizer:
    def __init__(self, ncols=5):
        self.ncols = ncols
        plt.ion()
        self.figures = []
        self.index = 0
        self.cost_list = []

    def draw_toy_heatmap(self):
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))

        for i, ax in enumerate(axes):
            W1 = np.random.randn(9, 1)
            # Creating a heatmap for W1
            sns.heatmap(W1, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            ax.set_title("Heatmap of W1", fontsize=16)
            ax.set_xlabel("Input Neuron")
            if i == 0:
                ax.set_ylabel("Hidden Layer Neuron")

        plt.show()

    def create_heatmap(self, Ws, Bs, iteration_num):
        # Store the data instead of the figure
        self.figures.append((Ws, Bs, iteration_num))

    def draw_heatmaps(self, cost_list):
        self.cost_list = cost_list
        self.root = tk.Tk()
        # Create a single figure and canvas and keep them
        self.fig = Figure(figsize=(15, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = tk.Button(button_frame, text='Previous', command=self.prev)
        btn_prev.pack(side=tk.LEFT)

        btn_next = tk.Button(button_frame, text='Next', command=self.next)
        btn_next.pack(side=tk.LEFT)

        self.update_plot()  # Update the plot when the window is first created
        self.root.mainloop()

    def update_plot(self):
        # Clear the current content of the figure
        self.fig.clf()
        # Get the data for the current plot
        Ws, Bs, iteration_num = self.figures[self.index]
        if(self.index == 0):
            current_cost = []
        else:
            current_cost = self.cost_list[self.index-1]
        # Add new subplots to the figure
        axes = self.fig.subplots(nrows=1, ncols=self.ncols)

        for i, (W, B) in enumerate(zip(Ws, Bs)):
            ax_W = axes[2*i]
            ax_B = axes[2*i + 1]

            # Creating a heatmap for W
            sns.heatmap(W, annot=True, cmap="coolwarm",
                        linewidths=0.5, ax=ax_W, cbar=False)
            ax_W.set_title(f"Heatmap of W{i+1}", fontsize=16)
            ax_W.set_xlabel("Input Neuron")
            if i == 0:
                ax_W.set_ylabel("Hidden Layer Neuron")

            # Creating a heatmap for B
            sns.heatmap(np.atleast_2d(B), annot=True, cmap="coolwarm",
                        linewidths=0.5, ax=ax_B, cbar=False)
            ax_B.set_title(f"Heatmap of B{i+1}", fontsize=16)
            ax_B.set_xlabel("Bias")
            ax_B.yaxis.set_visible(False)  # Hide y-axis for bias heatmap

        # Plot the cost last
        ax_cost = axes[self.ncols-1]
        ax_cost.set_xlabel("Iteration")
        ax_cost.yaxis.set_visible(True)
        ax_cost.set_ylabel("Cost")

        # Extract x and y values
        x = [list(d.keys())[0] for d in current_cost]
        y = [list(d.values())[0] for d in current_cost]

        # Plot the data using the ax object
        ax_cost.plot(x, y, marker='x')

        # Title of the plot using the ax object
        ax_cost.set_title('Cost over iterations', fontsize=16)

        # Add a title below the plots
        if(iteration_num == -1):
            self.fig.text(0.5, 0.01, f'Init', ha='center',
                          va='center', fontsize=16)
        else:
            self.fig.text(
                0.5, 0.01, f'Iteration: {iteration_num}', ha='center', va='center', fontsize=16)

        # Adjust layout
        self.fig.tight_layout()

        self.canvas.draw()  # Draw the updated figure on the canvas

    def next(self, *args):
        self.index = (self.index + 1) % len(self.figures)
        self.update_plot()

    def prev(self, *args):
        self.index = (self.index - 1) % len(self.figures)
        self.update_plot()
