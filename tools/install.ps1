<#
.SYNOPSIS
    Install the SAM 3 GIMP plug-in on Windows, and optionally pre-build the
    sam3gimpd daemon environment.

.DESCRIPTION
    The plug-in half is pure Python and installs by copying one directory into
    GIMP's user plug-ins folder -- GIMP requires <plug-ins>\sam3_gimp\sam3_gimp.py,
    the directory and the entry file sharing a name.

    The daemon half (torch, transformers, ~3 GB) lives in its own virtualenv
    under %LOCALAPPDATA%\sam3-gimp\venv and is normally built by the plug-in's
    own Setup dialog on first run. -SetupVenv does the same thing here, for
    people who would rather watch a console than a progress bar, and for
    unattended installs.

    Nothing in this script needs administrator rights: every path it writes to
    is user-scoped.

.PARAMETER PluginDir
    GIMP's user plug-ins directory. Default: $env:GIMP3_PLUGIN_DIR if set,
    otherwise $env:APPDATA\GIMP\<GimpVersion>\plug-ins.

.PARAMETER GimpVersion
    GIMP major.minor whose configuration directory to target. Default: the
    newest 3.x directory under %APPDATA%\GIMP that exists, else 3.0.

.PARAMETER Source
    The repository's plugin\sam3_gimp directory. Default: resolved relative to
    this script, so a checkout just works.

.PARAMETER SetupVenv
    Also create %LOCALAPPDATA%\sam3-gimp\venv and install the sam3gimpd package into
    it. Uses uv when available (much faster, and it can fetch its own Python),
    otherwise falls back to the py launcher / python on PATH. A uv this script
    downloads is checked against a pinned SHA-256 before it is run.

.PARAMETER Torch
    With -SetupVenv, also install torch and transformers into the venv, the
    same two steps the Setup dialog takes: the pinned torch and torchvision
    from the PyTorch index alone, then the daemon's [runtime] extra
    (transformers and friends, no torch) from PyPI. Without it the venv can
    still serve --stub mode, which is enough to prove the whole pipeline works
    before committing to a 3 GB download.

.PARAMETER Accelerator
    Which torch flavour -Torch installs: 'cuda' (default; NVIDIA, CUDA 12.8
    wheels) or 'cpu'. Setup's GUI detects this itself; here you say.

.PARAMETER CudaIndex
    PyTorch wheel index used by -Torch -Accelerator cuda. Default is the cu128
    channel. The cpu flavour always uses the /whl/cpu channel.

.PARAMETER PythonVersion
    Interpreter version uv provisions for the venv. Default 3.11, the same pin
    the Setup dialog uses (bootstrap.PINNED_PYTHON).

.PARAMETER Clean
    Delete the target sam3_gimp directory before copying, rather than merging
    into it. Use after renaming or removing plug-in modules.

.PARAMETER DryRun
    Report what would happen and change nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\install.ps1 -SetupVenv -Torch

.NOTES
    Written on Linux against the documented Windows layout; the GIMP paths and
    the uv bootstrap have not been executed on a Windows machine yet.
#>

[CmdletBinding()]
param(
    [string] $PluginDir,
    [string] $GimpVersion = '',
    [string] $Source,
    [switch] $SetupVenv,
    [switch] $Torch,
    [ValidateSet('cuda', 'cpu')]
    [string] $Accelerator = 'cuda',
    [string] $CudaIndex = 'https://download.pytorch.org/whl/cu128',
    [string] $PythonVersion = '3.11',
    [switch] $Clean,
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PluginName = 'sam3_gimp'
function Get-GimpVersion {
    <#  Newest GIMP 3.x config directory that actually exists.
        Hardcoding 3.0 wrote into the wrong folder on a 3.2 machine and the
        plug-in simply never appeared, with nothing to say why. #>
    param([string] $Requested)

    if ($Requested) { return $Requested }

    $root = Join-Path $env:APPDATA 'GIMP'
    if (Test-Path -LiteralPath $root) {
        $versions = @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^3\.\d+$' } |
            Sort-Object { [version] $_.Name })
        if ($versions.Count -gt 0) { return $versions[-1].Name }
    }
    return '3.0'
}

$GimpVersion = Get-GimpVersion $GimpVersion
$AppName    = 'sam3-gimp'
$UvVersion  = '0.9.7'   # mirrors bootstrap.UV_VERSION; tests/plugin/test_entry.py checks
# SHA-256 of the uv release assets this script can download, the same values
# bootstrap.UV_SHA256 pins.  The archive is checked before uv.exe is ever run;
# no entry, or a mismatch, stops the install.
$UvSha256   = @{
    'uv-x86_64-pc-windows-msvc.zip'  = '5d250c32d3604e28dbe18dc65c668ff628c53e00dde2c642576e831e4a60da64'
    'uv-aarch64-pc-windows-msvc.zip' = '4482ad2544544e1966b6d933d38e27368ce0307739d293322c10459f698a1629'
}
# torch and torchvision as every accelerator extra in _daemon\pyproject.toml
# pins them; mirrors bootstrap.TORCH_REQUIREMENTS.
$TorchRequirements = @('torch==2.9.0', 'torchvision==0.24.0')
$SkipDirs   = @('__pycache__', '.git', '.mypy_cache', '.pytest_cache', '.ruff_cache')
$SkipExt    = @('.pyc', '.pyo', '.orig', '.rej', '.swp')

# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
function Write-Step  { param([string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok    { param([string] $Message) Write-Host "    $Message" -ForegroundColor Green }
function Write-Info  { param([string] $Message) Write-Host "    $Message" }
function Write-Warn2 { param([string] $Message) Write-Host "    $Message" -ForegroundColor Yellow }

function Fail {
    param([string] $Message, [string] $Hint)
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    if ($Hint) { Write-Host "       $Hint" -ForegroundColor Yellow }
    exit 1
}

# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
function Get-SourceDir {
    param([string] $Explicit, [string] $ScriptDir)

    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit)) {
            Fail "The -Source directory does not exist: $Explicit"
        }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }

    # tools\install.ps1  ->  ..\plugin\sam3_gimp
    $candidate = Join-Path (Split-Path -Parent $ScriptDir) "plugin\$PluginName"
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    # A flattened release layout: install.ps1 next to the plug-in directory.
    $candidate = Join-Path $ScriptDir $PluginName
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    Fail "Could not find the plug-in source directory." `
         "Pass -Source <path to plugin\$PluginName>."
}

function Get-GimpPluginDir {
    param([string] $Explicit, [string] $Version)

    if ($Explicit)                 { return [System.Environment]::ExpandEnvironmentVariables($Explicit) }
    if ($env:GIMP3_PLUGIN_DIR)     { return $env:GIMP3_PLUGIN_DIR }
    if ($env:GIMP_PLUGIN_DIR)      { return $env:GIMP_PLUGIN_DIR }

    $roaming = $env:APPDATA
    if (-not $roaming) { $roaming = Join-Path $env:USERPROFILE 'AppData\Roaming' }
    return Join-Path $roaming "GIMP\$Version\plug-ins"
}

function Get-AppBaseDir {
    $local = $env:LOCALAPPDATA
    if (-not $local) { $local = Join-Path $env:USERPROFILE 'AppData\Local' }
    return Join-Path $local $AppName
}

# --------------------------------------------------------------------------- #
# copying
# --------------------------------------------------------------------------- #
function Get-PluginFiles {
    param([string] $Root)

    Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object {
        $relative = $_.FullName.Substring($Root.Length).TrimStart('\')
        $parts    = $relative.Split('\')
        $inSkipped = $false
        foreach ($part in $parts[0..([Math]::Max(0, $parts.Length - 2))]) {
            if ($SkipDirs -contains $part) { $inSkipped = $true }
        }
        (-not $inSkipped) -and ($SkipExt -notcontains $_.Extension) -and (-not $_.Name.EndsWith('~'))
    }
}


function Install-Plugin {
    param([string] $SourceDir, [string] $TargetRoot, [switch] $CleanFirst, [switch] $Simulate)

    $entry = Join-Path $SourceDir "$PluginName.py"
    if (-not (Test-Path -LiteralPath $entry)) {
        Fail "$SourceDir does not contain $PluginName.py." `
             "GIMP requires the entry file to be named after its directory."
    }

    $target = Join-Path $TargetRoot $PluginName

    if ($CleanFirst -and (Test-Path -LiteralPath $target)) {
        Write-Info "removing existing $target"
        if (-not $Simulate) { Remove-Item -LiteralPath $target -Recurse -Force }
    }

    if (-not (Test-Path -LiteralPath $TargetRoot)) {
        Write-Warn2 "$TargetRoot does not exist yet - creating it."
        Write-Warn2 "If GIMP $GimpVersion is installed but has never been run, that is expected."
        if (-not $Simulate) { New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null }
    }

    # The daemon ships inside the plug-in directory (sam3_gimp\_daemon), so the
    # ordinary file walk already includes it -- there is nothing to overlay.
    $items = New-Object System.Collections.ArrayList
    foreach ($f in @(Get-PluginFiles -Root $SourceDir)) {
        [void] $items.Add([pscustomobject]@{
            Full = $f.FullName
            Rel  = $f.FullName.Substring($SourceDir.Length).TrimStart('\')
        })
    }

    if (-not (Test-Path -LiteralPath (Join-Path $SourceDir '_daemon\pyproject.toml'))) {
        Write-Warn2 "_daemon\ is missing from the plug-in source."
        Write-Warn2 "Setup will not be able to install the daemon without it."
    }

    $copied  = 0
    $skipped = 0

    foreach ($file in $items) {
        $relative = $file.Rel
        $dest     = Join-Path $target $relative

        $same = $false
        if (Test-Path -LiteralPath $dest) {
            $sourceHash = (Get-FileHash -LiteralPath $file.Full -Algorithm SHA256).Hash
            $destHash   = (Get-FileHash -LiteralPath $dest        -Algorithm SHA256).Hash
            $same = ($sourceHash -eq $destHash)
        }

        if ($same) {
            $skipped++
            continue
        }

        Write-Info "+ $relative"
        if (-not $Simulate) {
            $parent = Split-Path -Parent $dest
            if (-not (Test-Path -LiteralPath $parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            Copy-Item -LiteralPath $file.Full -Destination $dest -Force
        }
        $copied++
    }

    Write-Ok "$copied file(s) written, $skipped unchanged"
    return $target
}

# --------------------------------------------------------------------------- #
# the daemon environment
# --------------------------------------------------------------------------- #
function Find-Uv {
    param([string] $BaseDir)

    $bundled = Join-Path $BaseDir 'tools\uv.exe'
    if (Test-Path -LiteralPath $bundled) { return $bundled }

    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    return $null
}

function Install-Uv {
    <#  The pinned release asset, the same one the Setup dialog downloads
        (bootstrap.UV_VERSION), so the two install routes cannot drift.  The
        astral.sh installer script fetches whatever is newest.  The archive is
        checked against $UvSha256 -- pinned here, not fetched from the release
        it is meant to vouch for -- before anything in it runs. #>
    param([string] $BaseDir, [switch] $Simulate)

    $toolsDir = Join-Path $BaseDir 'tools'
    $target   = Join-Path $toolsDir 'uv.exe'
    $arch     = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
    $asset    = "uv-$arch-pc-windows-msvc.zip"
    $url      = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$asset"
    $expected = $UvSha256[$asset]
    if (-not $expected) {
        Fail "No pinned checksum for $asset, so it will not be installed." `
             "Install uv yourself (it will be found on PATH), or let the plug-in's Setup dialog do it."
    }
    Write-Info "downloading uv $UvVersion ($asset) to $target"
    if ($Simulate) { return $target }

    if (-not (Test-Path -LiteralPath $toolsDir)) {
        New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
    }
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) "sam3-gimp-uv-$UvVersion"
    $zip     = Join-Path $staging $asset
    try {
        if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
        New-Item -ItemType Directory -Path $staging -Force | Out-Null
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            Fail "The uv download does not match its pinned checksum (expected $expected, got $actual)." `
                 "The file was corrupted or replaced in transit; nothing was installed. Try again, or check your proxy."
        }
        Write-Info 'sha256 verified'
        Expand-Archive -LiteralPath $zip -DestinationPath $staging -Force
        $exe = Get-ChildItem -LiteralPath $staging -Recurse -File -Filter 'uv.exe' | Select-Object -First 1
        if (-not $exe) { Fail "The uv archive did not contain uv.exe: $asset" }
        # Copied under a temporary name and renamed into place, so an
        # interrupted copy never leaves a truncated uv.exe that the next run
        # would find and use.
        $partial = "$target.part"
        Copy-Item -LiteralPath $exe.FullName -Destination $partial -Force
        Move-Item -LiteralPath $partial -Destination $target -Force
    }
    catch {
        Fail "Could not download uv $UvVersion : $($_.Exception.Message)" `
             "Fetch $url yourself, put uv.exe in $toolsDir and re-run, or let the plug-in's Setup dialog do it."
    }
    finally {
        if (Test-Path -LiteralPath $staging) {
            Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not (Test-Path -LiteralPath $target)) {
        Fail "uv did not appear at $target after installation."
    }
    return $target
}

function Find-SystemPython {
    foreach ($candidate in @('py', 'python3', 'python')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
    }
    return $null
}

function Test-TorchMarker {
    <#  Does the venv already hold torch for this accelerator, at today's pins?
        The marker is the one the Setup dialog reads and writes
        (bootstrap.torch_marker_file), so the two routes agree about it. #>
    param([string] $Venv, [string] $Extra)

    $marker = Join-Path $Venv 'sam3-gimp-torch.json'
    if (-not (Test-Path -LiteralPath $marker)) { return $false }
    try {
        $data = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
        return ($data.extra -eq $Extra) -and ((@($data.requirements) -join ' ') -eq ($TorchRequirements -join ' '))
    }
    catch {
        return $false
    }
}

function Write-TorchMarker {
    param([string] $Venv, [string] $Extra, [string] $Index)

    $data = [ordered]@{
        extra        = $Extra
        index_url    = $Index
        requirements = $TorchRequirements
        installed_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    }
    # WriteAllText writes UTF-8 without a byte-order mark.
    [System.IO.File]::WriteAllText((Join-Path $Venv 'sam3-gimp-torch.json'), ($data | ConvertTo-Json))
}

function New-Sam3Venv {
    param([string] $BaseDir, [string] $DaemonDir, [switch] $WithTorch, [switch] $Simulate)

    $venv       = Join-Path $BaseDir 'venv'
    $venvPython = Join-Path $venv 'Scripts\python.exe'

    if (-not (Test-Path -LiteralPath (Join-Path $DaemonDir 'pyproject.toml'))) {
        Fail "The daemon package directory was not found: $DaemonDir" `
             "It ships inside the plug-in at sam3_gimp\_daemon. Run this script from a full checkout or release zip, or skip -SetupVenv and let the Setup dialog install it."
    }

    $uv = Find-Uv -BaseDir $BaseDir
    if (-not $uv) { $uv = Install-Uv -BaseDir $BaseDir -Simulate:$Simulate }
    $haveUv = [bool] ($uv -and (Test-Path -LiteralPath $uv))
    # The package cache the Setup dialog uses too, so neither route downloads
    # torch a second time.
    $env:UV_CACHE_DIR = Join-Path $BaseDir 'cache\uv'

    if (Test-Path -LiteralPath $venvPython) {
        Write-Info "reusing existing venv at $venv"
    }
    else {
        Write-Info "creating venv at $venv (Python $PythonVersion)"
        if (-not $Simulate) {
            if ($haveUv) {
                # --python-preference managed lets uv download the interpreter
                # when the machine has none -- the same call the Setup dialog
                # makes.  --clear replaces a half-created venv, which uv
                # otherwise refuses to create over.
                & $uv venv --clear --python $PythonVersion --python-preference managed $venv
                if ($LASTEXITCODE -ne 0) { Fail "uv venv failed (exit $LASTEXITCODE)." }
            }
            else {
                $python = Find-SystemPython
                if (-not $python) {
                    Fail "No Python found to create the venv." `
                         "Install Python 3.10+ from python.org, or let uv handle it."
                }
                & $python -m venv $venv
                if ($LASTEXITCODE -ne 0) { Fail "python -m venv failed (exit $LASTEXITCODE)." }
            }
        }
    }

    # Two steps, never one command.  torch comes from the PyTorch index alone,
    # which serves all of its dependencies; then the daemon with its [runtime]
    # extra -- transformers and the rest, no torch -- from PyPI.  One command
    # over both indexes (an extra index plus --index-strategy
    # unsafe-best-match) would let any package come from whichever index
    # offers the best version, which uv documents as open to dependency
    # confusion.
    if ($WithTorch) {
        $index = if ($Accelerator -eq 'cuda') { $CudaIndex } else { 'https://download.pytorch.org/whl/cpu' }
        if (Test-TorchMarker -Venv $venv -Extra $Accelerator) {
            Write-Info "torch for $Accelerator is already installed"
        }
        else {
            Write-Step "Installing torch ($Accelerator) from $index (multi-gigabyte download)"
            if (-not $Simulate) {
                if ($haveUv) {
                    # --reinstall-package: 2.9.0+cpu satisfies torch==2.9.0, so
                    # without it switching to CUDA would keep the CPU build.
                    & $uv pip install --python $venvPython --index-url $index `
                        --reinstall-package torch --reinstall-package torchvision @TorchRequirements
                }
                else {
                    & $venvPython -m pip install --index-url $index --force-reinstall @TorchRequirements
                }
                if ($LASTEXITCODE -ne 0) {
                    Fail "torch installation failed (exit $LASTEXITCODE)." `
                         "Check that $index has a build for Python $PythonVersion."
                }
                Write-TorchMarker -Venv $venv -Extra $Accelerator -Index $index
            }
        }
        $requirement = "$DaemonDir[runtime]"
        Write-Step 'Installing sam3gimpd[runtime]: transformers and the rest, from PyPI'
    }
    else {
        $requirement = $DaemonDir
        Write-Step "Installing sam3gimpd (stub-capable, no torch) from $DaemonDir"
    }

    if (-not $Simulate) {
        if ($haveUv) {
            # A venv made by `uv venv` has no pip in it, so `python -m pip` fails
            # with "No module named pip"; uv installs into it by --python.
            # --reinstall-package: uv would otherwise keep an installed daemon of
            # the same version and ignore the new source.
            & $uv pip install --python $venvPython --reinstall-package sam3-gimp-daemon $requirement
        }
        else {
            & $venvPython -m pip install --upgrade pip
            & $venvPython -m pip install --upgrade $requirement
        }
        if ($LASTEXITCODE -ne 0) {
            Fail "sam3gimpd installation failed (exit $LASTEXITCODE)." `
                 "Check the messages above; the daemon itself is pure Python."
        }
    }

    return $venv
}

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot  = Split-Path -Parent $scriptDir

    Write-Host ''
    Write-Host 'SAM 3 for GIMP - installer' -ForegroundColor White
    Write-Host '--------------------------' -ForegroundColor White
    if ($DryRun) { Write-Warn2 'DRY RUN - nothing will be written.' }

    $sourceDir = Get-SourceDir -Explicit $Source -ScriptDir $scriptDir
    $targetRoot = Get-GimpPluginDir -Explicit $PluginDir -Version $GimpVersion

    Write-Step 'Locating directories'
    Write-Info "source : $sourceDir"
    Write-Info "target : $targetRoot"

    Write-Step 'Installing the plug-in'
    $installed = Install-Plugin -SourceDir $sourceDir -TargetRoot $targetRoot `
                                -CleanFirst:$Clean -Simulate:$DryRun
    Write-Ok "plug-in installed at $installed"

    if ($SetupVenv) {
        $baseDir   = Get-AppBaseDir
        # The daemon ships inside the plug-in directory (sam3_gimp\_daemon); the
        # repository has no top-level daemon\ any more.
        $daemonDir = Join-Path $sourceDir '_daemon'
        Write-Step "Building the sam3gimpd environment under $baseDir"
        $venv = New-Sam3Venv -BaseDir $baseDir -DaemonDir $daemonDir `
                             -WithTorch:$Torch -Simulate:$DryRun
        Write-Ok "environment ready at $venv"
        if (-not $Torch) {
            Write-Warn2 'torch was not installed: the daemon will only run in --stub mode.'
            Write-Warn2 'Re-run with -SetupVenv -Torch, or use the plug-in Setup dialog, to add it.'
        }
    }

    Write-Host ''
    Write-Host 'Next steps:' -ForegroundColor White
    Write-Info '1. Restart GIMP (plug-ins are only scanned at startup).'
    Write-Info '2. Filters > AI Segmentation > SAM 3 Setup / Doctor... to check the environment'
    Write-Info '   and, if you have not already, download the gated SAM 3 weights.'
    Write-Info '3. Open an image, then Filters > AI Segmentation > Segment interactively (canvas)...'
    Write-Host ''
}
catch {
    Write-Host ''
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    exit 1
}
