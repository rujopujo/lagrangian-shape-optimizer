# Start the Shape Optimizer app locally (run after code changes).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$port = if ($args[0]) { [int]$args[0] } else { 8501 }
Write-Host "Starting Streamlit on http://127.0.0.1:$port" -ForegroundColor Cyan
streamlit run app.py --server.port $port --server.address 127.0.0.1
