$ErrorActionPreference = 'Stop'
$demoRoot = Split-Path -Parent $PSScriptRoot
$demoShare = Join-Path $demoRoot 'runs/share'
$demoExe = Join-Path $demoShare 'bin/cloudflared.exe'
if (-not (Test-Path -LiteralPath $demoExe)) { throw 'cloudflared.exe is missing; install the official Cloudflare release first.' }
$demoListener = Get-NetTCPConnection -State Listen -LocalPort 8502 -ErrorAction SilentlyContinue
if ($demoListener) {
    $demoServer = Get-CimInstance Win32_Process -Filter "ProcessId=$($demoListener.OwningProcess)"
    if (-not $demoServer.CommandLine.Contains($demoRoot) -or $demoServer.CommandLine -notlike '*serve_shared.py*') {
        throw 'Port 8502 is used by another application.'
    }
} else {
    $demoStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Start-Process -FilePath (Join-Path $demoRoot '.venv/Scripts/python.exe') -ArgumentList @('scripts/serve_shared.py','--host','127.0.0.1','--port','8502','--public') -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoShare "server-$demoStamp.stdout.log") -RedirectStandardError (Join-Path $demoShare "server-$demoStamp.stderr.log") | Out-Null
}
$demoPidFile = Join-Path $demoShare 'tunnel.pid'
$demoUrlFile = Join-Path $demoShare 'public-url.txt'
if (Test-Path -LiteralPath $demoPidFile) {
    $demoSavedPid = [int](Get-Content -LiteralPath $demoPidFile -Raw).Trim()
    $demoRunning = Get-CimInstance Win32_Process -Filter "ProcessId=$demoSavedPid"
    if ($demoRunning -and $demoRunning.ExecutablePath -eq $demoExe) {
        if (Test-Path -LiteralPath $demoUrlFile) { Get-Content -LiteralPath $demoUrlFile }
        Write-Output 'Tunnel is already running.'
        exit 0
    }
}
$demoStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$demoLog = Join-Path $demoShare "tunnel-$demoStamp.stderr.log"
$demoTunnel = Start-Process -FilePath $demoExe -ArgumentList @('tunnel','--url','http://127.0.0.1:8502','--protocol','http2','--no-autoupdate') -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoShare "tunnel-$demoStamp.stdout.log") -RedirectStandardError $demoLog -PassThru
$demoTunnel.Id | Set-Content -LiteralPath $demoPidFile
for ($demoAttempt=0; $demoAttempt -lt 25; $demoAttempt++) {
    Start-Sleep -Seconds 1
    if (Test-Path -LiteralPath $demoLog) {
        $demoMatch = [regex]::Match((Get-Content -LiteralPath $demoLog -Raw),'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com')
        if ($demoMatch.Success) {
            $demoMatch.Value | Set-Content -LiteralPath $demoUrlFile
            Write-Output $demoMatch.Value
            Write-Output 'No access code is required. Wait for the tunnel connection before sharing.'
            exit 0
        }
    }
}
Write-Output "Tunnel is still starting. Check $demoLog"
