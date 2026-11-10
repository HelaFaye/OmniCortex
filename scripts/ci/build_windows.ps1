# CI: build one Windows target's binaries.
#
#   pwsh scripts/ci/build_windows.ps1 -Target windows-x64 -Arch x64 [-Ref <llama.cpp ref>]
#
# -Arch is the Visual Studio platform: x64, Win32 or ARM64. Writes dist\omnicortex-<target>\ (bin\ with the
# llama.cpp tools and their DLLs, omnicortex-remote.exe) and dist\omnicortex-<target>.zip.
param([Parameter(Mandatory)][string]$Target, [Parameter(Mandatory)][string]$Arch, [string]$Ref = "",
      [string]$CMakeExtra = "")
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\..\.."
$out = Join-Path $root "dist\omnicortex-$Target"
$llama = Join-Path $root "build\llama.cpp"
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$out\bin" | Out-Null

if (-not (Test-Path "$llama\.git")) { git clone --depth 1 https://github.com/ggml-org/llama.cpp $llama }
if ($Ref) { git -C $llama fetch --depth 1 origin $Ref; git -C $llama checkout FETCH_HEAD }
Remove-Item -Recurse -Force "$llama\tools\cett-dump" -ErrorAction SilentlyContinue
Copy-Item -Recurse "$root\llama-tools\cett-dump" "$llama\tools\"
$cm = Get-Content "$llama\tools\CMakeLists.txt" -Raw
if ($cm -notmatch 'add_subdirectory\(cett-dump\)') { Add-Content "$llama\tools\CMakeLists.txt" "`nadd_subdirectory(cett-dump)" }

$flags = @("-S", $llama, "-B", "$llama\build", "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_OPENSSL=OFF")
if ($Arch -eq "ARM64") {
  # MSVC can't target a portable ARM CPU in llama.cpp (it only probes the build machine, whose SVE a Snapdragon
  # laptop lacks). Build with clang through llama.cpp's own toolchain file, as llama.cpp's CI does.
  if (-not (Get-Command clang -ErrorAction SilentlyContinue)) {
    $llvm = "C:\Program Files\LLVM\bin"
    if (-not (Test-Path "$llvm\clang.exe")) { choco install llvm -y --no-progress; if ($LASTEXITCODE) { throw "LLVM install failed" } }
    $env:PATH = "$llvm;$env:PATH"
  }
  if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    python -m pip install --quiet ninja          # native ARM64 wheel; its ninja.exe lands in Python's Scripts
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
      $env:PATH = "$(Split-Path (Get-Command python).Source)\Scripts;$env:PATH"
    }
  }
  clang --version
  $flags += @("-G", "Ninja Multi-Config", "-DCMAKE_TOOLCHAIN_FILE=$llama\cmake\arm64-windows-llvm.cmake", "-DGGML_OPENMP=OFF")
} else {
  $flags += @("-A", $Arch)
}
if ($CMakeExtra) { $flags += $CMakeExtra.Split(" ") }
cmake @flags
if ($LASTEXITCODE) { throw "cmake configure failed" }
cmake --build "$llama\build" --config Release -j $env:NUMBER_OF_PROCESSORS --target llama-cett-dump llama-server llama-quantize llama-eval-callback
if ($LASTEXITCODE) { throw "build failed" }
$bin = "$llama\build\bin\Release"
foreach ($f in "llama-cett-dump", "llama-server", "llama-quantize", "llama-eval-callback") { Copy-Item "$bin\$f.exe" "$out\bin\" }
Get-ChildItem "$bin\*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination "$out\bin\"

python -m pip install --quiet pyinstaller
python -m PyInstaller --onefile --clean --name omnicortex-remote --paths "$root\remote" `
  --distpath $out --workpath "$root\build\pyi-$Target" --specpath "$root\build\pyi-$Target" "$root\scripts\ci\remote_entry.py"
if ($LASTEXITCODE) { throw "PyInstaller failed" }
& "$out\omnicortex-remote.exe" --help | Out-Null
if ($LASTEXITCODE) { throw "omnicortex-remote does not start" }

Copy-Item "$root\README.md" "$out\"
Copy-Item "$root\remote\README.md" "$out\REMOTE.md"
Compress-Archive -Force -Path $out -DestinationPath "$root\dist\omnicortex-$Target.zip"
Get-ChildItem $out, "$out\bin"
