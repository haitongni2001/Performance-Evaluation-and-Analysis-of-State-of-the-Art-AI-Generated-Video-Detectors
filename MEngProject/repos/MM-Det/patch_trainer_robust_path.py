from pathlib import Path

p = Path("utils/trainer.py")
s = p.read_text(encoding="utf-8")

old = """                for fn in fns:
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

new = """                for fn in fns:
                    parts = fn.replace('\\\\', '/').split('/')

                    # Robust path parsing for Windows/customized datasets.
                    # Expected cache hierarchy:
                    #   cached_mm_representations[dataset_name][0_real or 1_fake][frame_name]
                    label_positions = [
                        idx for idx, part in enumerate(parts)
                        if part in ('0_real', '1_fake')
                    ]

                    if label_positions:
                        label_pos = label_positions[-1]
                        label = parts[label_pos]
                        dname = parts[label_pos - 1] if label_pos > 0 else self.config['classes'][0]
                        frame_id = parts[-1]
                    else:
                        dname = self.config['classes'][0]
                        if '0_real' in fn:
                            label = '0_real'
                        elif '1_fake' in fn:
                            label = '1_fake'
                        else:
                            label = 'unknown'
                        frame_id = parts[-1]

                    prefix, index = frame_id.rsplit('__', maxsplit=1)
                    cached_index = get_nearest_mm_index(
                        self.cached_mm_representations_index[dname][label][prefix],
                        int(index) + i
                    )
                    visual_representations.append(
                        self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual']
                    )
                    textual_representations.append(
                        self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1']
                    )
"""

if old not in s:
    print("Exact previous patch block not found. Trying to patch original repo block...")

    old_original = """                for fn in fns:
                    *_, dname, label, frame_id = fn.rsplit('/')
                    prefix, index = frame_id.rsplit('__', maxsplit=1)
                    cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], int(index) + i)
                    visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                    textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
"""

    if old_original not in s:
        raise RuntimeError(
            "Target block not found. Run this to inspect lines 88-110:\n"
            "python -c \"from pathlib import Path; lines=Path('utils/trainer.py').read_text(encoding='utf-8').splitlines(); "
            "[print(f'{i+1}: {lines[i]}') for i in range(88,110)]\""
        )

    s = s.replace(old_original, new)
else:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("patched utils/trainer.py successfully")