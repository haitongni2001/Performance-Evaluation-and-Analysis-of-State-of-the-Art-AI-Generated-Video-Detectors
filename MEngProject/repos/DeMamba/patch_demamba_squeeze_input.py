from pathlib import Path

p = Path("util.py")
s = p.read_text(encoding="utf-8")

old = """        varInput = torch.autograd.Variable(input.contiguous().cuda())
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

        logit = model(varInput)
"""

new = """        # DeMamba dataloader returns [B, 1, T, C, H, W].
        # Model expects [B, T, C, H, W].
        if input.dim() == 6 and input.shape[1] == 1:
            input = input.squeeze(1)

        varInput = torch.autograd.Variable(input.contiguous().cuda())
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

        logit = model(varInput)
"""

if old not in s:
    raise RuntimeError("Target block not found in util.py")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched train_one_epoch input squeeze")