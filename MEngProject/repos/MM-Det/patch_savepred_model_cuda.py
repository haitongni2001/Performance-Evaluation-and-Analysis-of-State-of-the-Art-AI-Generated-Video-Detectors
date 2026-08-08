from pathlib import Path

p = Path("test_customized_savepred.py")
s = p.read_text(encoding="utf-8")

old = """    model.load_state_dict(state_dict, strict=config["cache_mm"])

    all_pred_dfs = []
"""

new = """    model.load_state_dict(state_dict, strict=config["cache_mm"])

    # Important for our custom save-prediction loop:
    # ensure model weights and input tensors are on the same device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_pred_dfs = []
"""

if old not in s:
    raise RuntimeError("Target block not found in test_customized_savepred.py")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched test_customized_savepred.py: model moved to CUDA")