| run | strategy | size | trainable_params_M | median_epoch_time_min | best_val_miou | test_miou | test_mdice | test_miou_ignore_boundary | latency_ms_bs1 |
|---|---|---|---|---|---|---|---|---|---|
| strategy_scratch_128 | scratch | 128 | 4.5346 | 1.9197 | 0.7409 | 0.7459 | 0.8437 | 0.7584 | 20.5050 |
| strategy_two_stage_128 | two_stage | 128 | 4.5346 | 1.8946 | 0.8651 | 0.8764 | 0.9315 | 0.9141 | 20.9367 |
| strategy_discrim_128 | discrim | 128 | 4.5346 | 1.9400 | 0.8620 | 0.8788 | 0.9318 | 0.9060 | 18.1768 |
| strategy_finetune_128 | finetune | 128 | 4.5346 | 1.9238 | 0.8739 | 0.8875 | 0.9376 | 0.9091 | 19.3631 |
| strategy_frozen_128 | frozen | 128 | 1.5626 | 1.1459 | 0.8786 | 0.8908 | 0.9398 | 0.9148 | 18.8409 |
| res_frozen_160 | frozen | 160 | 1.5626 | 1.5290 | 0.8895 | 0.9000 | 0.9452 | 0.9249 | 24.5616 |
| long_frozen_160 | frozen | 160 | 1.5626 | 1.4884 | 0.8935 | 0.9030 | 0.9469 | 0.9341 | 24.0792 |
| long_finetune_160 | finetune | 160 | 4.5346 | 2.7315 | 0.8923 | 0.9036 | 0.9471 | 0.9332 | 23.5144 |
| res_frozen_224 | frozen | 224 | 1.5626 | 2.9858 | 0.8979 | 0.9102 | 0.9511 | 0.9318 | 33.5653 |
