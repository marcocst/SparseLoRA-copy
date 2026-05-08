# SparseLoRA: Improving the Rank-Efficiency of LoRA

## Quickstart

 1. Installing `SparseLoralib` is simply

 pip install SparseLoralib


 2. You can choose to adapt some layers by replacing them with counterparts implemented in `SparseLoralib`. We only support `nn.Linear`, `nn.Embedding`, and `nn.Conv2d` for now. We also support a `MergedLinear` for cases where a single `nn.Linear` represents more than one layers, such as in some implementations of the attention `qkv` projection (see Additional Notes for more).


 3. Before the training loop begins, mark only LoRA parameters as trainable.
 ```python
 import SparseLoralib as sparselora
 model = BigModel()
 # This sets requires_grad to False for all parameters without the string "sparselora_" in their names
 sparselora.mark_only_lora_as_trainable(model)
 # Training loop
 for batch in dataloader:
    ...
 ```
 4. When saving a checkpoint, generate a `state_dict` that only contains LoRA parameters.
 ```python
 # ===== Before =====
 # torch.save(model.state_dict(), checkpoint_path)
 # ===== After =====
 torch.save(sparselora.lora_state_dict(model), checkpoint_path)
 ```
 5. When loading a checkpoint using `load_state_dict`, be sure to set `strict=False`.
 ```python
 # Load the pretrained checkpoint first
 model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
 # Then load the LoRA checkpoint
 model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
 ```




