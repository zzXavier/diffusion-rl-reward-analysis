# Handoff

Completed on CPU: two behavior tests, four smoke policies, two real stochastic rollouts per epoch, two epochs per trained arm, two passes over each epoch's stored old policy transitions, adapter checkpoint/reload checks and independent pixel diagnostics. Saved smoke traces and per-image results are in `examples/smoke/`. The run records 32 per-transition old/new log probabilities and both learned LoRA updates have nonzero L2 norm (0.03894183). Tiny-policy smoke rewards use image total variation.

Formal evaluation requires a completed P02 head and CPU PickScore checkpoint, plus the pinned SD1.5 base. Run `python run.py reproduce --base LOCAL_SD15 --pickscore LOCAL_PICKSCORE --head P02_HEAD --device cuda`. The checked-in smoke result tests the mechanics, not human preference alignment. After formal validation, regenerate tables with `python run.py evaluate` from a clean clone.
