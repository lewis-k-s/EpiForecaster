# TODO

This is the high level todo for necessary steps in the codebase

## Model

**SIR regularization**

Constrain model forecasting to physics-based limits using SIR methods

- [ ] transformer output is S_t, I_t, R_t
- [ ] non-trainable layer computes derivatives of each
- [ ] Loss regularisation: $$L = w_1 L_{\text{Data}} + w_2 L_{\text{SIR}}$$
  - $L_{\text{Data}}$ (Standard): Difference between predicted and observed wastewater/hospitalizations.
  - $L_{\text{SIR}}$ (The Regularizer): This term penalizes the model if the change in $I$ doesn't match the SIR formula ($-\beta \frac{SI}{N} + \gamma I$).
- [ ] observation heads from latent SIR
  - $I_t \to \text{Predicted Hospitalizations}$
  - $I_t \to \text{Predicted Cases}$.
  - $I_t \to \text{Predicted Deaths}$.
  - $I_t \to \text{Predicted Wastewater}$.

## Training

**Synthetic data curriculum**

@docs/TRAINING_CURRICULUM.md

## Interpretation

Enhancing our tools for understanding model performance and underlying epidemic dynamics using the model

**Analyzing GNN gradient dynamics across the lockdown periods**

- Do we observe higher GNN gradients at the boundaries of lockdowns as the mobility regime shifts significantly?
- Do we observe _lower_ GNN gradients within the lockdowns while mobility is limited? Does this scale with the strength/severity of the restrictions?

- [ ] script to plot GNN gradnorm evolution through the time window, with lockdowns marked
- [ ] statistical analysis of lockdown strength vs. GNN gradnorm effect
