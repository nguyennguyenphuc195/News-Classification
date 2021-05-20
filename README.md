# News-Classification
Run train.py to retrain 1D cnn model from scratch.
Run test.py to evaluate pretrained model.

Classify Vietnameses news into categories with raw text

validation split = 0.2
|             | Accuracy on Train + Validation | Accuracy on Validation (at 20 epochs)|
| ----------------------- | -------------------|--------------------------|
| Simple Dense network    | _                  | 0.5882                   |
| LSTM                    | _                  | 0.5785                   |
| 1D convolutional network| 0.86               |  0.6282                  |

