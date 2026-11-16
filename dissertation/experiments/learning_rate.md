## Learning Rate Experiments

### Objective

The purpose of these experiments was to identify a suitable learning rate for the neural network when training the Higgs boson classifier.

The learning rate controls how much the model's weights are adjusted during each optimisation step. A learning rate that is too high can cause the model to overshoot optimal weight values, while one that is too low can result in slower training and may prevent the model from reaching a good solution within the available epochs.

### Experimental Setup

To make the comparisons fair, the following settings were kept consistent across experiments:

- Model architecture: 41 → 64 → 32 → 1
- Optimiser: Adam
- Batch size: 1024
- Number of epochs: 20
- Random seed: 42
- Loss function: BCEWithLogitsLoss
- Model selection metric: Validation AMS

Only the learning rate was changed between experiments. Using the same random seed also helped ensure that each experiment started from the same initial model weights and followed the same training data shuffle sequence.

### Results

| Learning rate | Best validation AMS | Best epoch |
|---|---:|---:|
| 0.001 | 3.4168 | 7 |
| 0.00075 | 3.4161 | 7 |
| 0.0006 | 3.4460 | 5 and 8 |
| 0.0005 | **3.4620** | **7** |
| 0.0004 | 3.4460 | 8 |
| 0.0003 | 3.4358 | 18 |

### Analysis and Selection

The learning rate of 0.0005 achieved the highest validation AMS of 3.4620, making it the best-performing configuration based on the primary evaluation metric.

Although the learning rate of 0.001 achieved a slightly higher ROC-AUC of 0.9085, its validation AMS was lower at 3.4168. Since AMS is the primary metric for evaluating the classifier's ability to distinguish Higgs signal events from background events, the learning rate of 0.0005 was selected.

The results also showed that lower learning rates did not necessarily produce better performance. For example, the learning rate of 0.0003 reached its best AMS later, at epoch 18, but still did not outperform 0.0005.

### Additional Epoch Experiment

After selecting 0.0005, an additional experiment was conducted by increasing the number of epochs from 20 to 30 while keeping all other settings unchanged.

The best validation AMS remained 3.4620 at epoch 7. Although ROC-AUC increased slightly from 0.9072 to 0.9081, the additional training did not improve the primary evaluation metric.

Therefore, 20 epochs was retained as the baseline training configuration.

### Conclusion

A learning rate of **0.0005** was selected for the baseline model, with 20 training epochs and a batch size of 1024. This configuration achieved a best validation AMS of **3.4620 at epoch 7**, using a classification threshold of 0.84.

Future experiments will retain this baseline configuration while investigating other hyperparameters, allowing their individual effects on model performance to be assessed.