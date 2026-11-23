"""Compile the source-built address-window fixture and run the D3D12 readback."""
from pathlib import Path
import subprocess
import sys
import tempfile

assembler, validator, cross, dxc, renderer, adapter = sys.argv[1:]
root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as folder:
    directory = Path(folder)
    module = directory / "window.spv"
    hlsl = directory / "window.hlsl"
    dxil = directory / "window.dxil"
    subprocess.run([sys.executable, "-B", root / "tools/spirv_address_window.py",
                    Path(__file__).with_name("subtract.spvasm"), module,
                    "--assembler", assembler, "--validator", validator], check=True)
    subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl], check=True)
    subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", dxil, hlsl], check=True)
    sys.exit(subprocess.run([renderer, adapter, "--spirv-address-window", dxil]).returncode)
