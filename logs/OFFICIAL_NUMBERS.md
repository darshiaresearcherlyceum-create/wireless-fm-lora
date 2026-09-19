| Model | Params (Trainable) | CE@-5dB | CE@25dB | PREC@25dB | LOC RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Proposed | 35.09M (35,087,450) | −8.08 | −23.95 | 4.07 | 15.54 |
| Static LoRA | 8.62M (8,620,180) | −0.55 | −16.79 | 4.06 | 15.30 |
| A1 | 34.89M (34,890,586) | −1.33 | −23.83 | 3.98 | 13.62 |
| A2 | 35.04M (35,037,525) | −7.65 | −24.24 | 4.04 | 13.69 |
| A3 | 8.64M (8,640,789) | −1.14 | −11.87 | 4.91 | 12.27 |
| RZF | 0 | n/a | n/a | 103.44 | n/a |

*(124M frozen backbone parameters; 35.09M trainable Proposed parameters).*

Detection BER = 0 at 20–25 dB is from MMSE initialization, not from Dynamic LoRA.
