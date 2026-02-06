# Set PYTHONPATH to the current root directory to allow imports from 'app'
$env:PYTHONPATH = "$((Get-Item .).FullName);$env:PYTHONPATH"

Write-Host "Starting LLM Instructions Utility..." -ForegroundColor Cyan
streamlit run llm_instructions_utility/util_app.py
