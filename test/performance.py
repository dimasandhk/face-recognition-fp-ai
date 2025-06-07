import matplotlib.pyplot as plt
import numpy as np

# --- Data collated from the paper and public benchmarks ---
models = ["VGG-Face", "Facenet", "ArcFace", "SFace", "GhostFaceNet"]
accuracy = [98.87, 99.20, 99.83, 91.90, 99.70]
# Computational Cost in Million Floating Point Operations (MFLOPs)
mflops = [15500, 1600, 4100, 500, 150]

# --- Create the plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for Accuracy (left y-axis)
color_accuracy = 'tab:blue'
ax1.set_xlabel('Face Recognition Model', fontsize=12, labelpad=15)
ax1.set_ylabel('LFW Accuracy (%)', color=color_accuracy, fontsize=12)
bars = ax1.bar(models, accuracy, color=color_accuracy, label='LFW Accuracy (%)')
ax1.tick_params(axis='y', labelcolor=color_accuracy)
ax1.set_ylim(90, 100.5) # Set Y-axis limit to highlight differences

# Add accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval}%', ha='center', va='bottom')

# Create a second y-axis for Computational Cost (MFLOPs)
ax2 = ax1.twinx()
color_cost = 'tab:red'
ax2.set_ylabel('Computational Cost (MFLOPs - log scale)', color=color_cost, fontsize=12)
# Plot MFLOPs as a line plot on a logarithmic scale for better visualization
ax2.plot(models, mflops, color=color_cost, marker='o', linestyle='--', label='Cost (MFLOPs)')
ax2.set_yscale('log') # Use a log scale due to the wide range of MFLOPs values
ax2.tick_params(axis='y', labelcolor=color_cost)

# --- Final Touches ---
plt.title('Model Accuracy vs. Computational Efficiency Trade-off', fontsize=16, pad=20)
fig.tight_layout() # Adjust layout to make room for labels

# Add a legend
# To create a single legend for both axes, we combine their handles and labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left')

# Save the figure to a file
output_filename = 'figure4_model_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Chart saved as {output_filename}")

# Optionally, show the plot
plt.show()