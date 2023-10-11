This is the temporary repo for UNO-DSt submitting to EMNLP 2023.

We give a brief introduction in T5.py for joint training process
For self-training, please refer to self-training folder

This code is built on the baseline of T5DST from Facebook Research.

| **MultiWoz Version** | **Model**| **attraction** | **hotel** | **restaurant** | **taxi**  | **train** | **Average** | **Margin (2.1->2.4)** |
|------------------|------------|------------|-------|------------|-------|-------|---------|--------|
| MultiWoz 2.1     | T5DST      | 30.45      | 19.38 | 20.42      | 66.32 | 25.60 | 32.44   |        |
| MultiWoz 2.1     | UNO-DST-JT | 32.86      | 22.91 | 29.47      | 66.00 | 31.68 | 36.58   |        |
| MultiWoz 2.1     | UNO-DST-ST | 33.09      | 25.66 | 30.99      | 65.48 | 48.90 | 40.82   |        |
| MultiWoz 2.4     | T5DST      | 31.38      | 16.51 | 15.78      | 66.13 | 23.80 | 30.72   |    -1.72     |
| MultiWoz 2.4     | UNO-DST-JT | 32.83      | 22.90 | 29.32      | 65.94 | 32.47 | 36.69   |    0.11      |
| MultiWoz 2.4     | UNO-DST-ST | **35.02**  | **25.72** | **31.50** | **66.00** | **52.55** | **42.16**   | **1.34** |

Table 1. Results of T5DST, UNO-DST for MultiWoz 2.1 and 2.4. JT/ST stands for UNO-DST joint and self-training periods. The best results are shown in bold format. 
# UNO
