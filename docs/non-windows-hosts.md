# Contributing from macOS and Linux

Xbox2Windows targets x64 Windows for its C++, HLSL, MASM, D3D12, WARP, and GPU test suite. Contributors on macOS and Linux can still edit documentation, run the platform-independent Python checks, and run the LLVM 15 `gfx1013` shader translation checks locally before handing off Windows verification.

## Platform split

| Work or check | macOS / Linux | x64 Windows | Notes |
|---|---|---|---|
| Documentation edits | Yes | Yes | Markdown, diff hygiene, and link checks (`tests/docs_links_check.py`). |
| Platform-independent Python checks | Yes | Yes | The commands below cover the [Linux CI workflow](../.github/workflows/windows.yml), the BC7 block reference and the GPU report formatter. |
| LLVM 15 `gfx1013` translation checks | Yes | No (via CI / separate LLVM) | `tests/pixel_aot_check.py binary`, `tests/ffx_dispatch_aot_check.py`, and `tests/rdna2_tools_check.py` (with `LLVM_LIB`, `LLVM_MC`, `LLVM_OBJCOPY`) verify assembling, disassembling, and lowering only. |
| `cmake --preset dev` build (MSVC `cl` + Ninja) | No | Yes | `CMakePresets.json` selects `cl`. |
| MASM fixtures (`tests/cpu/*.asm`) and `cpu_*` tests | No | Yes | Assembled and built by the Windows toolchain. |
| DXC compilation and rendering (`pixel_aot_warp`, `pixel_aot_hardware`) | No | Yes | Requires Windows SDK / DXC (`dxc.exe`, `dxil.dll`) and D3D12. |
| D3D12, WARP, and DXR tests (`*_warp`, `*_hardware`) | No | Yes | Covers software WARP and physical GPU checks. |
| System32 PE export inventory (`tools/pe_inventory.py`) | No | Yes | Uses `ctypes.windll` on Windows. |
| Standalone GPU check packaging (`tools/package-gpu-check.ps1`) | No | Yes | Builds and smoke-tests the Windows x64 ZIP package. |

`tests/pixel_aot_check.py binary`, `tests/ffx_dispatch_aot_check.py`, and `tests/rdna2_tools_check.py` do not invoke DXC or render pixels; desktop HLSL compilation and readback (`pixel_aot_warp`, `pixel_aot_hardware`) remain Windows-only.

## Quick start on macOS

Install Python 3.12 and LLVM 15 with Homebrew (both are needed as explicit paths because `llvm@15` is keg-only):

```bash
brew install python@3.12 llvm@15
```

From the repository root, create a virtual environment under `out/pe-env`, install `pefile` at the version pinned in `requirements-dev.txt`, and run the platform-independent checks below:

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv out/pe-env
./out/pe-env/bin/python -m pip install -c requirements-dev.txt pefile
for t in \
  tests/script_state_check.py \
  tests/resource_root_check.py \
  tests/bcpack_check.py \
  tests/bcpack_symbols_check.py \
  tests/bc7_blocks_check.py \
  tests/bc7_tiles_check.py \
  tests/resource_ranges_check.py \
  tests/ttpf_batch_export_check.py \
  tests/awc_metadata_check.py \
  tests/sgd5_effects_check.py \
  tests/dxcontainer_part_offsets_check.py \
  tests/layout_chunks_check.py \
  tests/dependency_paths_check.py \
  tests/intake/paths_check.py \
  tests/shader_probe_check.py \
  tests/shader_census_check.py \
  tests/dxil_retarget_check.py \
  tests/rdna2_tools_check.py \
  tests/pso_readiness_check.py \
  tests/docs_links_check.py \
  tests/bidi_controls_check.py \
  tests/hardware_reports_check.py \
  tests/gpu_report_format_check.py; do
  ./out/pe-env/bin/python -B "$t"
done
```

`tests/form_coverage_check.py` and `tests/xmem_attribute_scan_check.py` import `iced-x86`, which pip builds from source on Apple Silicon, so this list leaves them out. Run them on Linux or Windows.

Then run the three LLVM 15 `gfx1013` checks:

```bash
LLVM_PREFIX="$(brew --prefix llvm@15)"
./out/pe-env/bin/python -B tests/pixel_aot_check.py binary \
  "$LLVM_PREFIX/bin/llvm-mc" "$LLVM_PREFIX/bin/llvm-objcopy" "$LLVM_PREFIX/lib/libLLVM-15.dylib"
./out/pe-env/bin/python -B tests/ffx_dispatch_aot_check.py \
  "$LLVM_PREFIX/bin/llvm-mc" "$LLVM_PREFIX/bin/llvm-objcopy" "$LLVM_PREFIX/lib/libLLVM-15.dylib"
LLVM_LIB="$LLVM_PREFIX/lib/libLLVM-15.dylib" \
  LLVM_MC="$LLVM_PREFIX/bin/llvm-mc" \
  LLVM_OBJCOPY="$LLVM_PREFIX/bin/llvm-objcopy" \
  ./out/pe-env/bin/python -B tests/rdna2_tools_check.py
```

## Quick start on Linux

The [CI workflow](../.github/workflows/windows.yml) runs the platform-independent Python suite on Ubuntu 24.04 with Python 3.12 (`python-linux`) and the `gfx1013` machine-code checks on Ubuntu 22.04 with LLVM 15 (`pixel-binary`).

On Debian/Ubuntu, install `python3-venv` first. Without it, `python3 -m venv` fails with `ensurepip is not available`:

```bash
sudo apt-get update -q
sudo apt-get install -y python3-venv
```

From the repository root, create `out/pe-env` (Python 3.11 or newer; CI uses 3.12), install dependencies, and run the platform-independent checks below:

```bash
python3 -m venv out/pe-env
./out/pe-env/bin/python -m pip install -r requirements-dev.txt
for t in \
  tests/script_state_check.py \
  tests/resource_root_check.py \
  tests/bcpack_check.py \
  tests/bcpack_symbols_check.py \
  tests/bc7_blocks_check.py \
  tests/bc7_tiles_check.py \
  tests/form_coverage_check.py \
  tests/resource_ranges_check.py \
  tests/ttpf_batch_export_check.py \
  tests/awc_metadata_check.py \
  tests/sgd5_effects_check.py \
  tests/dxcontainer_part_offsets_check.py \
  tests/layout_chunks_check.py \
  tests/dependency_paths_check.py \
  tests/intake/paths_check.py \
  tests/shader_probe_check.py \
  tests/shader_census_check.py \
  tests/dxil_retarget_check.py \
  tests/rdna2_tools_check.py \
  tests/pso_readiness_check.py \
  tests/xmem_attribute_scan_check.py \
  tests/docs_links_check.py \
  tests/bidi_controls_check.py \
  tests/hardware_reports_check.py \
  tests/gpu_report_format_check.py; do
  ./out/pe-env/bin/python -B "$t"
done
```

On Debian/Ubuntu, install LLVM 15 and run the three `gfx1013` checks:

```bash
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends llvm-15
./out/pe-env/bin/python -B tests/pixel_aot_check.py binary \
  llvm-mc-15 llvm-objcopy-15 libLLVM-15.so.1
./out/pe-env/bin/python -B tests/ffx_dispatch_aot_check.py \
  llvm-mc-15 llvm-objcopy-15 libLLVM-15.so.1
LLVM_LIB=libLLVM-15.so.1 LLVM_MC=llvm-mc-15 LLVM_OBJCOPY=llvm-objcopy-15 \
  ./out/pe-env/bin/python -B tests/rdna2_tools_check.py
```

## Optional Vulkan arithmetic check

With the Linux build of DXC, a Vulkan device, and [VkRunner](https://gitlab.freedesktop.org/mesa/vkrunner), run the generated FP32 exponent helpers against a Python reference.

```bash
python3 -B tests/pixel_fp32_vulkan_check.py /path/to/dxc /path/to/vkrunner
```

The probe checks finite output bits, signed zero, infinities, and NaN class through a compute buffer. A second dispatch uses an intentionally incorrect underflow rounding rule and must fail the same comparison. Repeat `--device <index>` to test specific one-based VkRunner device indices. The fixture exercises the desktop helpers through SPIR-V. Pixel shader lowering and D3D12 readback have their own tests.

## Expected CMake failure and no-stub policy

`CMakePresets.json` sets `CMAKE_CXX_COMPILER` to `cl` for the `dev` preset. Running `cmake --preset dev` on macOS or Linux is expected to fail:

```text
-- The CXX compiler identification is unknown
CMake Error at CMakeLists.txt:2 (project):
  The CMAKE_CXX_COMPILER:

    cl

  is not a full path and was not found in the PATH.
```

Do not work around this by stubbing `cl`, MASM, D3D12, or WARP to make CMake configure succeed on non-Windows hosts. Full compilation and runtime tests require x64 Windows.

## WSL and Linux translation layers

- **WSL:** You can run the Linux Python and LLVM 15 checks inside WSL, while the CMake, MSVC, DXC, and D3D12 build still runs from Windows PowerShell as documented in [First build](../README.md#first-build).
- **Linux GPU Check via Proton and Wine:** Contributors have verified the standalone Windows GPU Check under Linux translation layers (Wine or Proton with `vkd3d-proton`). Mesa RADV reported native 64-lane waves and passed forced wave64 on an [AMD Ryzen 5 integrated GPU](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7067289597) and an [RX 9070 XT reported by @barbiscool](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7069870226). On the RX 9070 XT, RADV defaults to 32-lane waves (unlike Windows 64-lane defaults) and passes forced 64. NVIDIA runs on an [RTX 4070](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064884397), [RTX 5060 Ti](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7067186626) and [RTX 3060](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064204388) reported 32-lane maximums and completed all 408 BVH comparisons. Translation-layer caveats: `vkd3d-proton` is required (Wine's built-in D3D12 fails `bvh_waves` and skips differential checks), Wine DXGI reports placeholder driver versions (e.g. `35.0.65535.65535`), and `bvh_root_roundtrip_hardware` fails on `vkd3d-proton` due to its acceptance of undefined root flags (`0x03000000`) that Microsoft D3D12 rejects with `E_INVALIDARG`. If you report a translation-layer run of the standalone GPU Check in `#1`, include the commit or release tag, GPU model, Mesa or NVIDIA driver version, Proton/Wine and `vkd3d-proton` versions, and each subtest's output (including skips).
- A native Linux runtime is outside the scope of this guide and remains a separate request from the x64 Windows target.

### Compile check with msvc-wine

[msvc-wine](https://github.com/mstorsjo/msvc-wine) runs the real MSVC toolchain and Windows SDK under Wine. This is not a stub, so you can use it to confirm that a C++ or HLSL change compiles with `cl` and DXC before Windows CI runs. Review the Visual Studio Build Tools license before using the unattended acceptance option below.

```bash
python3 vsdownload.py --accept-license --dest ~/.local/msvc
./install.sh ~/.local/msvc
```

The SDK's `dxc.exe` is not wrapped. Save this script as `~/.local/msvc/bin/x64/dxc` and make it executable. It maps absolute Linux paths to Wine's `Z:` drive:

```bash
#!/bin/bash
DXC="$HOME/.local/msvc/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"
args=()
for a in "$@"; do
  if [[ -e $a && $a == /* || $a == /*/* && -d ${a%/*} ]]; then args+=("z:${a//\//\\}"); else args+=("$a"); fi
done
WINEDEBUG=-all exec wine "$DXC" "${args[@]}"
```

Then configure with `CMAKE_SYSTEM_NAME=Windows`. Without it, CMake's compiler check fails to link with `LNK1181`.

```bash
export PATH="$HOME/.local/msvc/bin/x64:$PATH"
cmake --preset release -DCMAKE_SYSTEM_NAME=Windows -DDXC_EXECUTABLE="$HOME/.local/msvc/bin/x64/dxc"
cmake --build --preset release
```

The resulting test binaries run under Wine and `vkd3d`, not Windows D3D12 or WARP. Expect `desktop_memory_query`, the `d3d12_xs_shim_*` export and root checks, and `ffx_provider_loop_*` and `bvh_root_roundtrip_*` to fail there, and the DXR and BVH tests to skip. Treat a Wine run as a compile check only. Do not report Wine test results as Windows verification or hardware coverage.

## Troubleshooting

- **Homebrew `llvm@15` is keg-only:** Homebrew does not symlink `llvm@15` binaries into `PATH` by default. Pass `"$(brew --prefix llvm@15)"` explicitly instead of calling bare `llvm-mc`.
- **Shared library filename (`libLLVM-15.dylib` vs `libLLVM-15.so.1`):** Pass `libLLVM-15.dylib` from `"$(brew --prefix llvm@15)/lib"` on macOS, and `libLLVM-15.so.1` on Ubuntu/Debian.
- **LLVM versions newer than 15:** CI and the test fixtures target LLVM 15. Newer LLVM versions may print a different `exp` instruction spelling during AMDGPU disassembly.
- **`tests/shader_probe_check.py` and `tests/dxil_retarget_check.py` with vs without arguments:** Running either script with no arguments runs its platform-independent controls (`shader_metadata_controls` and `dxil_retarget_controls`). Passing compiled `.cso` shaders, `root_probe`, and `dxil.dll` (`shader_metadata_probe`, `dxil_retarget_probe`) requires Windows and DXC.
- **`tests/rdna2_tools_check.py` skips one test without `LLVM_LIB`:** Without `LLVM_LIB`, the script runs its pure-Python disassembler, extractor, and scanner checks and skips `test_llvm_comparison`. Setting `LLVM_LIB`, `LLVM_MC`, and `LLVM_OBJCOPY` runs the LLVM 15 comparison check as well.
- **`tools/pe_inventory.py` fails to import `ctypes.windll`:** `pe_inventory.py` inspects System32 exports on x64 Windows and is not supported on macOS or Linux.
- **Keep virtual environments out of Git:** Create the Python environment at `out/pe-env`, which is already ignored via `/out/` in `.gitignore`.

## Evidence and Windows verification handoff

Follow [Share a reproducible finding](../CONTRIBUTING.md#share-a-reproducible-finding) and [AGENTS.md](../AGENTS.md) when preparing a pull request or issue from a non-Windows host:

- Record the exact commit SHA under test, host OS and architecture, Python and LLVM versions, exact commands run, and observed results.
- State clearly that Windows C++, HLSL, WARP, and hardware checks were not run locally when contributing from macOS or Linux, and rely on the `build-and-test` CI job or a Windows contributor to verify Windows behavior.
- Never include game files, dumps, `.exe`/`.dll` binaries, torrents, distribution links, personal paths, or credentials in issues, pull requests, or Git.
