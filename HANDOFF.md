# Handoff

Formal training completed on an NVIDIA GeForce RTX 4060 Laptop 8 GB with torch 2.4.1+cu121. Both the SD1.5 policy and frozen FP32 P02 scorer ran on CUDA. The P02 encoder and head SHA values were verified. The four-parameter head is bundled with its MIT license; the trained SD1.5 adapters have the base model license in licenses/.

Real DDPO evidence includes 32 sampled trajectories and 16 optimizer updates per trained policy, 640 finite transition log-probability records in total, and LoRA delta L2 norms 0.46959622 and 0.47107000 after checkpoint reload. No logged ratio reached the 15% clipping boundary. Three untouched prompts tested all four methods, with four candidates per prompt. All 48 images are included; the two frozen-policy arms deliberately share a candidate bank. The run produced genuine parameter updates but no robust evidence of a semantic quality gain. Original summary SHA-256: dc8a0b4794ab1778cd6959d5ce8acd4203f610dd97ca3b7631fd79f81ff4a7f2.

Verified smoke path: two tests, tiny policy training and a second clean-output execution with matching summary. Formal command: python run.py reproduce --device cuda --reward-device cuda --base LOCAL_SD15 --cache CACHE --pickscore CACHE/PickScore. The default head path uses the exact checked-in P02 head.

Next independent check: clone the remote commit, run demo and evaluate, then reproduce both policies into a new output directory. Compare reward, independent color coverage and diversity across all four arms. The CUDA stochastic training can exhibit floating-point drift.