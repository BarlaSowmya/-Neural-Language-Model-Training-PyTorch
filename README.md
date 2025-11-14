# Neural Language Model Training PyTorch
# Assignment Overview
This project builds a neural language model from scratch for next-character prediction, using a GRU sequence model on Jane Austen's "Pride and Prejudice". The core objectives match the assignment:

* Custom code for PyTorch data preprocessing, batching, and training.
* Explore underfitting, best-fit, and overfitting regimes.
* Evaluate all setups with loss and character-level perplexity.
* Show clear, well-justified results with code and explanation.

# 1.Data and Tokenization 
## Dataset

The dataset used for these experiments is publicly available on Google Drive:

[Pride_and_Prejudice-Jane_Austen.txt](https://drive.google.com/file/d/1FNuJSYRdN2BomupmNkkKROBSkQzSPS9N/view?usp=sharing)

Download automatically in Colab with:
```
!pip install -q gdown
!gdown --id 1FNuJSYRdN2BomupmNkkKROBSkQzSPS9N -O Pride_and_Prejudice-Jane_Austen.txt
```

* Dataset: "Pride_and_Prejudice-Jane_Austen.txt" (public domain).
* Preprocessing: Lowercased, punctuation retained, no stopword removal.
* Tokenization: Character-level (every unique letter, digit, space, symbol is a token).
* Why character-level?:
  * Simpler and robust with classic literary text.
  * Guarantees a bounded vocabulary (~60-70 tokens).
  * Straightforward batching and sequence generation.

# 2. Model Architecture
* Model: Multi-layer Gated Recurrent Unit (GRU).
   *Embedding layer for vocabulary (maps character indices to vectors).
   *One or more stacked GRU layers.
   *Linear projection to vocabulary dimension.

* Why GRU?

   * Handles sequence dependencies efficiently without vanishing/exploding gradient issues of basic RNNs.
   * Simpler than LSTM, yet effective.
* Hyperparameters Tuned: hidden_dim, embed_dim, num_layers, dropout, epochs, batch size.

# 3. Training and Evaluation
* Loss Function: CrossEntropy (per character position in the sequence).
* Perplexity: P=exp(cross-entropy loss) (lower = better; reflects model's average "confusion" per character).
* Splitting: 90% train, 10% validation. Batched by sequence length (SEQ_LEN).
* Plot: Both training and validation loss per epoch for all experiments.
  
# 4. Experiments
Three types of model training runs were performed:

* Underfit: Used minimal model capacity or training duration, resulting in high loss and high perplexity on both train and validation sets.
* Best-Fit: Balanced model size, epochs, and regularization to achieve low and similar loss/perplexity on both train and validation splits.
* Overfit: Used maximum model capacity, no regularization, and many epochs, causing the training loss to drop very low while validation loss stagnates or rises (large gap).
Plots and results for all three regimes are provided in the repository.

# 5.Results Table
You can extract my "Assignment2.zip" file to access and evaluate my codes.
| Scenario | Train Pplx | Val Pplx |
| -------- | ---------- | -------- |
| Underfit | 8.9250     | 8.9431   |
| Best-Fit | 2.5493     | 2.5680   |
| Overfit  | 1.33       | 1.36     |

Exact values are provided in the individual notebook outputs.

# 6. Interpretation
Key outcomes:
* Underfit: model can't learn patterns.
* Best-fit: model generalizes well.
* Overfit: model memorizes training data but fails to generalize.
  
# 7. How Hyperparameters Control Fit
* Increase model size, epochs, remove dropout ⇒ overfits.
* Reduce model size, epochs, raise dropout ⇒ underfits.
* Tune for balance, use early stopping, reasonable dropout ⇒ best-fit.

  
# 8. References
PyTorch Documentation.
[PyTorch documentation — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/stable/index.html)

* NLP research and benchmarks for language modeling.​
* See assignment prompt and classic NLP resources for more details.
Assignment will also be available in a folder on repository

# 10. Conclusion
This project shows exactly how neural sequence modeling works, and how key decisions—model size, regularization, and training duration—directly lead to underfitting, best-fit, or overfitting. All code is from scratch and fully explained.

# Reproducibility
This codebase is designed to ensure full reproducibility of results for all language model experiments. The following best practices have been implemented:

Random seeds are set for PyTorch, NumPy, and Python’s random module to make results consistent:

python
```import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
 ```

Data download is automated: The dataset is fetched via public Google Drive link if not present locally, so no manual data upload is needed.

Environment requirements are listed, including Python and major library versions, to avoid setup discrepancies.

Model weights saving/loading: Best models are saved during training 
```
(torch.save(model.state_dict(), ...))
```
, and can be reloaded for evaluation or text generation using 
```
model.load_state_dict(torch.load('best_gru_model.pt')).
```
Hyperparameters are fixed in each config and reported in the README and notebook, allowing exact re-execution of each experiment.

Training/test splits are deterministic due to fixed random seed and split order.

Anyone with access to the code and public data link can exactly reproduce the reported results and plots by running the notebook or script in a standard Colab or local environment.
