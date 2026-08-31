"""Simple test runner — writes pytest output to a file."""
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_ingestion.py", "--tb=short", "-v"],
    capture_output=True,
    text=True,
    cwd=r"c:\Users\prami\scikit_learn_data",
)

output = result.stdout + result.stderr
print(output)

with open(r"c:\Users\prami\scikit_learn_data\test_result.txt", "w", encoding="utf-8") as f:
    f.write(output)

sys.exit(result.returncode)
