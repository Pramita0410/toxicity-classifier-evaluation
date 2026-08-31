C:\ProgramData\Anaconda3\python.exe -m pytest tests/test_ingestion.py -v --tb=short 2>&1 | Out-File -FilePath test_output.txt -Encoding utf8
Get-Content test_output.txt
