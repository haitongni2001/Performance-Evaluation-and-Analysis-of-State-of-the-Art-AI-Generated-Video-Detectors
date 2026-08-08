from pathlib import Path

p = Path(r"D:\HaitongNi\MEngProject_SecondBench\01_WaveRep\scripts\eval_genvidbench_waverep.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    '"video_id": row["video_id"],',
    '"video_id": row["rel_path"] if "rel_path" in row.index else Path(video_path).stem,'
)

# Also make source_type/task_type safe if copied from AIGVDBench script.
s = s.replace(
    '"source_type": row["source_type"],',
    '"source_type": row["pair"] if "pair" in row.index else "",'
)

s = s.replace(
    '"task_type": row["task_type"],',
    '"task_type": row["pair"] if "pair" in row.index else "",'
)

p.write_text(s, encoding="utf-8")
print("patched GenVidBench column compatibility")