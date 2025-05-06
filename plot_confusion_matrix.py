import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix values from our evaluation
cm = np.array([[4, 1],
               [1, 4]])

# Create the plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Crisis', 'Crisis'],
            yticklabels=['No Crisis', 'Crisis'])

plt.title('Confusion Matrix for Crisis Prediction Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add accuracy, precision, recall values as text
plt.text(-0.4, -0.2, f'Accuracy: 80%\nPrecision: 80%\nRecall: 80%\nSpecificity: 80%', 
         bbox=dict(facecolor='white', alpha=0.8))

# Save the plot
plt.tight_layout()
plt.savefig('confusion_matrix_viz.png')
plt.close() 