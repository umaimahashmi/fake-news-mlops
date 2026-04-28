# Fake News Detection — Paper Reproduction Results
Paper: Rout et al., J. Cybersecur. Priv. 2025, 5, 43

## Optimal Model Results (Table 13)
Dataset  Test Acc  Test F1  Test Prec  Test Rec    Params
   ISOT    0.9973   0.9973     0.9974    0.9973 7,281,346
   LIAR    0.5526   0.5386     0.5717    0.5526 7,281,346

Average Accuracy: 0.7750 (paper: 0.7985)
Average F1:       0.7679 (paper: 0.7984)

## Component Importance (Table 12)
      Component Removed  Avg_Impact  Std_Dev  Experiments
           no_attention     0.32020 0.283408            8
         no_feedforward     0.02260 0.031961            8
             no_dropout     0.02130 0.030123            8
   no_relative_position     0.01395 0.017890            2
 no_positional_encoding     0.01395 0.017890            6
no_residual_connections     0.01265 0.017890            8
          no_layer_norm     0.00995 0.014071            2
 no_layer_normalization     0.00995 0.014071            6
           no_cls_token     0.00665 0.009405            2