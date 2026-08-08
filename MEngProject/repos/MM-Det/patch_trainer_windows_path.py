from pathlib import Path

p = Path("utils/trainer.py")
s = p.read_text(encoding="utf-8")

old = """                for fn in fns:
                    *_, dname, label, frame_id = fn.rsplit('/')
                    prefix, index = frame_id.rsplit('__', maxsplit=1)
                    cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], int(index) + i)
                    visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                    textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
"""

new = """                for fn in fns:
                    parts = fn.replace('\\\\', '/').split('/')
                    if len(parts) >= 3:
                        dname, label, frame_id = parts[-3], parts[-2], parts[-1]
                    elif len(parts) == 2:
                        dname = 'customized'
                        label, frame_id = parts[-2], parts[-1]
                    else:
                        dname = 'customized'
                        label = 'unknown'
                        frame_id = parts[-1]

                    prefix, index = frame_id.rsplit('__', maxsplit=1)
                    cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], int(index) + i)
                    visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                    textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
"""

if old not in s:
    raise RuntimeError("Target block not found. The file may not be restored to the original version.")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched trainer.py successfully")