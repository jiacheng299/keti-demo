$ErrorActionPreference = 'Stop'
$demoRoot = Split-Path -Parent $PSScriptRoot
$demoShare = Join-Path $demoRoot 'runs/share'
$demoPidFile = Join-Path $demoShare 'tunnel.pid'
if (-not (Test-Path -LiteralPath $demoPidFile)) { Write-Output 'No saved tunnel process.'; exit 0 }
$demoTunnelPid = [int](Get-Content -LiteralPath $demoPidFile -Raw).Trim()
$demoTunnel = Get-CimInstance Win32_Process -Filter "ProcessId=$demoTunnelPid"
if (-not $demoTunnel) { Write-Output 'Tunnel is already stopped.'; exit 0 }
$demoExpected = Join-Path $demoShare 'bin/cloudflared.exe'
if ($demoTunnel.ExecutablePath -ne $demoExpected) { throw 'Saved PID belongs to another program; nothing was stopped.' }
Stop-Process -Id $demoTunnelPid -ErrorAction Stop
Write-Output 'Public tunnel stopped. The local demo remains running.'
