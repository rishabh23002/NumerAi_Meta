
# Meta Learning for Numerai

## Project Overview
This project explores a meta-learning approach to the Numerai prediction problem. By treating each era as a unique task, this method allows our models to adapt dynamically to changes in data distribution across different eras. 

## Modeling Approaches

### Deep Learning Models
- **LSTM Model**
  - Number of layers: 4
- **Transformer Model**
  - Encoder only
  - Number of layers: 4

### Meta-Learning Techniques
1. **Model-Agnostic Meta-Learning (MAML)**
   - Models trained: MLP, Transformer, LSTM using MAML technique.

2. **MAML + Features from Other Models**
   - Combined features from LSTM and Transformer models, totaling 44 features to train an LSTM model using MAML.

3. **MAML with Test-Time Adaptation**
   - Employed test-time adaptation techniques to fine-tune models using cosine similarity measures between training and test distributions.
   - Ensemble model was used, integrating outputs from multiple models (Transformer, LSTM, AutoEncoder, MLP, Temporal Block) through a weighted sum computed via MAML.

### Ensemble and Adaptation Strategy
- **Ensemble Model**: Predictions were generated from five different models and combined using a weighted sum.
- **Weights Generation**: A separate model was used to predict weights for combining model outputs based on the training/test feature set.

