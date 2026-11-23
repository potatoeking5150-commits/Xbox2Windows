"""Translate MinLod samples and check which mip they read."""
from pathlib import Path
import subprocess
import sys
import tempfile


SAMPLES = (
    ("implicit", "OpImageSampleImplicitLod %v4 %sampled %uv MinLod %one"),
    ("bias", "OpImageSampleImplicitLod %v4 %sampled %uv Bias|ConstOffset|MinLod %bias %offset %one"),
    ("grad", "OpImageSampleExplicitLod %v4 %sampled %uv Grad|MinLod %dx %dy %one"),
    ("no-clamp", "OpImageSampleImplicitLod %v4 %sampled %uv"),
)


def main(assembler, validator, cross, dxc, renderer, adapter):
    source = Path(__file__).with_suffix(".spvasm").read_text(encoding="ascii")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        asm, module, hlsl, dxil = (path / f"shader.{ext}" for ext in ("spvasm", "spv", "hlsl", "dxil"))
        for name, instruction in SAMPLES:
            asm.write_text(source.replace("@sample@", instruction), encoding="ascii")
            subprocess.run([assembler, "--target-env", "vulkan1.2", asm, "-o", module], check=True)
            subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
            translated = subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl],
                                        capture_output=True, text=True)
            if translated.returncode:
                print(translated.stderr, file=sys.stderr, flush=True)
                if "MinLod texture operand not supported in HLSL." in translated.stderr:
                    print("skip: SPIRV-Cross lacks MinLod support; run tools/setup-spirv-tools.ps1")
                    return 77
                translated.check_returncode()
            subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", dxil, hlsl], check=True)
            draw = subprocess.run([renderer, adapter, "--spirv-minlod-quad-lod", dxil],
                                  capture_output=True, text=True)
            print(f"{name}\n{draw.stdout}{draw.stderr}", flush=True)
            if draw.returncode == 77:
                return 77
            wrong = name == "no-clamp"
            witness = "pixel_mismatch x=0 y=0" if wrong else "pixels=64 channels=256 max_absolute_error=0"
            if draw.returncode != int(wrong) or witness not in draw.stdout + draw.stderr:
                raise RuntimeError(f"Unexpected {name} mip readback")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
