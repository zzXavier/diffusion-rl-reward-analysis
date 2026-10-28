# Source register

[DDPO-PyTorch](https://github.com/kvablack/ddpo-pytorch/tree/1958463f020112c9a7bc85768d296daacc2e1b4b), MIT. `src/ddpo/ddim_with_logprob.py` comes from its `ddpo_pytorch/diffusers_patch` directory. The original license is retained alongside it. The sole compatibility edit imports `randn_tensor` from `diffusers.utils.torch_utils`. That upstream file credits the original Hugging Face Diffusers DDIM implementation. The runner, prompt split, per-trajectory logging and independent color evaluator are local additions.

Formal base [SD1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14), smoke base [tiny SD](https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe/tree/3ee6c9f225f088ad5d35b624b6514b091e6a4849). Model weights are loaded from the local cache. The scoring routine and feature order from [Text-to-Image Preference Reranking](https://github.com/zzXavier/text-to-image-reranking) are reproduced in `src/pickscore.py`; the formal run records the SHA-256 of the supplied head. The scorer uses the fixed [PickScore checkpoint](https://huggingface.co/yuvalkirstain/PickScore_v1/tree/a4e4367c6dfa7288a00c550414478f865b875800).

The formal SD1.5 LoRA adapters build on the SD1.5 base and are distributed under its CreativeML Open RAIL-M terms. The [complete license](licenses/LICENSE-SD1.5.txt) accompanies the adapters. The model and use restrictions apply alongside the source-code notices.

This repository includes the original four-coefficient preference head under [its MIT license](examples/preference-head.LICENSE). Its canonical release SHA-256 is b9cd5ee6aa4ca6cb749a291ddfbf3c3da56e39559b38f54fe9d0418f2fd6a597. It was fit on 24 disjoint Pick-a-Pic training prompts using the fixed PickScore v1 encoder. The original images and encoder are obtained separately.

Preference-reranking source revision [915804b4962dd30a0f21d9c8001691ce29b1bf30](https://github.com/zzXavier/text-to-image-reranking/tree/915804b4962dd30a0f21d9c8001691ce29b1bf30). The bundled four coefficients are identical to results/head.json in that commit.
