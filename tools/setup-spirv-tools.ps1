# Build the three SPIR-V command-line tools the bridge tests need (spirv-as, spirv-val and
# spirv-cross) from pinned source revisions into out/tools/spirv/bin. The Windows CI jobs cache
# that directory, keyed by this file and the MSVC toolset. Locally, put the printed directory on
# PATH before configuring, or pass it to CMake with -DSPIRV_WINDOW_AS=... and so on.
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\out\tools\spirv'),
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'

# SPIRV-Tools v2025.5 and the SPIRV-Headers revision its DEPS file names; SPIRV-Cross as used by
# the bridge fixtures since #553.
$revisions = [ordered]@{
    'SPIRV-Tools'   = '262bdab48146c937467f826699a40da0fdfc0f1a'
    'SPIRV-Headers' = 'b824a462d4256d720bebb40e78b9eb8f78bbb305'
    'SPIRV-Cross'   = '57e4efcc84112abcd426dc5ec248b5339bee04f2'
}
$tools = 'spirv-as.exe', 'spirv-val.exe', 'spirv-cross.exe'
$crossPatch = Join-Path $PSScriptRoot 'patches/spirv-cross-minlod.patch'

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Destination = (Resolve-Path -LiteralPath $Destination).Path
$bin = Join-Path $Destination 'bin'
$stamp = Join-Path $bin 'revisions.txt'
$expected = ($revisions.GetEnumerator() | ForEach-Object { "$($_.Key) $($_.Value)" }) -join "`n"
$expected += "`nSPIRV-Cross MinLod $((Get-FileHash -LiteralPath $crossPatch).Hash)"
$built = (Test-Path -LiteralPath $stamp) -and
    ((Get-Content -Raw -LiteralPath $stamp).Trim() -eq $expected) -and
    -not ($tools | Where-Object { -not (Test-Path -LiteralPath (Join-Path $bin $_)) })
if ($built) {
    Write-Host "SPIR-V tools already built at the pinned revisions: $bin"
} else {
    if (-not $env:VCToolsInstallDir) { & "$PSScriptRoot\dev.ps1" }
    $src = Join-Path $Destination 'src'
    foreach ($entry in $revisions.GetEnumerator()) {
        $dir = Join-Path $src $entry.Key
        if (-not (Test-Path -LiteralPath (Join-Path $dir '.git'))) {
            git init --quiet $dir
            if ($LASTEXITCODE -ne 0) { throw "git init failed for $($entry.Key)." }
        }
        git -C $dir fetch --quiet --depth 1 "https://github.com/KhronosGroup/$($entry.Key).git" $entry.Value
        if ($LASTEXITCODE -ne 0) { throw "Fetching $($entry.Key) $($entry.Value) failed." }
        git -C $dir checkout --quiet --force --detach FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { throw "Checking out $($entry.Key) failed." }
        $head = git -C $dir rev-parse HEAD
        if ($head -ne $entry.Value) { throw "$($entry.Key) is at $head, expected $($entry.Value)." }
    }

    git -C (Join-Path $src 'SPIRV-Cross') apply -- $crossPatch
    if ($LASTEXITCODE -ne 0) { throw 'Applying the SPIRV-Cross MinLod patch failed.' }

    # Static CRT, so the cached binaries do not depend on the runner's redistributable.
    $common = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_POLICY_DEFAULT_CMP0091=NEW',
                '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded')
    $build = Join-Path $Destination 'build'
    # SPIRV-Tools expands these paths inside CMake macros, where backslashes become escapes.
    $headers = (Join-Path $src 'SPIRV-Headers').Replace('\', '/')
    $interpreter = (Get-Command $Python).Source.Replace('\', '/')
    cmake -S (Join-Path $src 'SPIRV-Tools') -B (Join-Path $build 'SPIRV-Tools') @common `
        "-DSPIRV-Headers_SOURCE_DIR=$headers" "-DPython3_EXECUTABLE=$interpreter" `
        -DSPIRV_SKIP_TESTS=ON -DSPIRV_WERROR=OFF -DSPIRV_TOOLS_BUILD_STATIC=ON
    if ($LASTEXITCODE -ne 0) { throw 'SPIRV-Tools configuration failed.' }
    cmake --build (Join-Path $build 'SPIRV-Tools') --target spirv-as spirv-val
    if ($LASTEXITCODE -ne 0) { throw 'SPIRV-Tools build failed.' }
    cmake -S (Join-Path $src 'SPIRV-Cross') -B (Join-Path $build 'SPIRV-Cross') @common `
        -DSPIRV_CROSS_CLI=ON -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_SHARED=OFF
    if ($LASTEXITCODE -ne 0) { throw 'SPIRV-Cross configuration failed.' }
    cmake --build (Join-Path $build 'SPIRV-Cross') --target spirv-cross
    if ($LASTEXITCODE -ne 0) { throw 'SPIRV-Cross build failed.' }

    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    foreach ($tool in $tools) {
        $found = Get-ChildItem -LiteralPath $build -Recurse -File -Filter $tool | Select-Object -First 1
        if (-not $found) { throw "Build did not produce $tool." }
        Copy-Item -LiteralPath $found.FullName -Destination (Join-Path $bin $tool) -Force
    }
    Set-Content -LiteralPath $stamp -Value $expected -Encoding ascii
}

# Let each native process finish before truncating its output for the log.
$asVersion = & (Join-Path $bin 'spirv-as.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'spirv-as does not run.' }
$asVersion | Select-Object -First 1
$crossRevision = & (Join-Path $bin 'spirv-cross.exe') --revision
if ($LASTEXITCODE -ne 0) { throw 'spirv-cross does not run.' }
$crossRevision | Select-Object -First 1
Write-Output $bin
