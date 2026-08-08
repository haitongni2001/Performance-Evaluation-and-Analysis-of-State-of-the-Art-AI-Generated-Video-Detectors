from pathlib import Path

p = Path("utils/trainer.py")
s = p.read_text(encoding="utf-8")

debug_block = """        print("DEBUG fns sample:", fns[:3] if isinstance(fns, (list, tuple)) else fns)
        print("DEBUG cached index top keys:", list(self.cached_mm_representations_index.keys())[:5])
        for _k in list(self.cached_mm_representations_index.keys())[:3]:
            try:
                print("DEBUG cached index subkeys", _k, list(self.cached_mm_representations_index[_k].keys())[:5])
            except Exception as _e:
                print("DEBUG subkey error", _e)

"""

s = s.replace(debug_block, "")
p.write_text(s, encoding="utf-8")
print("removed debug prints")
