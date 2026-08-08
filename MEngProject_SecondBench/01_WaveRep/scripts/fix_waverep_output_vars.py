from pathlib import Path

p = Path(r"D:\HaitongNi\MEngProject_SecondBench\01_WaveRep\scripts\eval_genvidbench_waverep.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    'OUT_CSV = r"D:\\HaitongNi\\MEngProject_SecondBench\\01_WaveRep\\results\\waverep_genvidbench_common1000_predictions.csv"',
    'OUTPUT_CSV = Path(r"D:\\HaitongNi\\MEngProject_SecondBench\\01_WaveRep\\results\\waverep_genvidbench_common1000_predictions.csv")'
)

s = s.replace(
    'OUT_TXT = r"D:\\HaitongNi\\MEngProject_SecondBench\\01_WaveRep\\results\\waverep_genvidbench_common1000_results.txt"',
    'OUTPUT_TXT = Path(r"D:\\HaitongNi\\MEngProject_SecondBench\\01_WaveRep\\results\\waverep_genvidbench_common1000_results.txt")'
)

s = s.replace(
    'OUT_GENERATOR_CSV = r"D:\\HaitongNi\\MEngProject_SecondBench\\01_WaveRep\\results\\waverep_genvidbench_common1000_generator_breakdown.csv"',
    'OUTPUT_GENERATOR_CSV = Path(r"D:\\HaitongNi\\MEngProject_SecondBench\\01_WaveRep\\results\\waverep_genvidbench_common1000_generator_breakdown.csv")'
)

# Safety: if script later still uses old names, redirect them.
s = s.replace("OUT_CSV", "OUTPUT_CSV")
s = s.replace("OUT_TXT", "OUTPUT_TXT")
s = s.replace("OUT_GENERATOR_CSV", "OUTPUT_GENERATOR_CSV")

p.write_text(s, encoding="utf-8")
print("fixed WaveRep output variable names")