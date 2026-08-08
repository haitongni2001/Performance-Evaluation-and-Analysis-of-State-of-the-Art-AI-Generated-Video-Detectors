from pathlib import Path

p = Path("utils/trainer.py")
s = p.read_text(encoding="utf-8")

needle = """        for i in range(0, L - self.config['window_size'] + 1, self.config['window_size']):
"""

insert = """        print("DEBUG fns sample:", fns[:3] if isinstance(fns, (list, tuple)) else fns)
        print("DEBUG cached index top keys:", list(self.cached_mm_representations_index.keys())[:5])
        for _k in list(self.cached_mm_representations_index.keys())[:3]:
            try:
                print("DEBUG cached index subkeys", _k, list(self.cached_mm_representations_index[_k].keys())[:5])
            except Exception as _e:
                print("DEBUG subkey error", _e)
"""

if insert not in s:
    s = s.replace(needle, insert + "\n" + needle)

p.write_text(s, encoding="utf-8")
print("added debug prints")