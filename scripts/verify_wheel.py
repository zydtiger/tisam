"""Build and verify the base wheel in an isolated environment on Linux or Windows."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Verify installed public imports and CLI without optional workflow extras."""
    project = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="tisam-wheel-") as directory:
        smoke = Path(directory)
        subprocess.run(["uv", "build", "--out-dir", str(smoke / "dist")], cwd=project, check=True)
        constraints = smoke / "constraints.txt"
        with constraints.open("w", encoding="utf-8") as output:
            subprocess.run(
                ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes"],
                cwd=project,
                stdout=output,
                check=True,
            )
        environment = smoke / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)],
            cwd=smoke,
            check=True,
        )
        windows = os.name == "nt"
        binaries = environment / ("Scripts" if windows else "bin")
        python = binaries / ("python.exe" if windows else "python")
        (wheel,) = (smoke / "dist").glob("*.whl")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--constraint",
                str(constraints),
                str(wheel),
            ],
            cwd=smoke,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "from tisam import TiSAM, ModelConfig, load_model; "
                "import importlib.util; assert importlib.util.find_spec('mammoth') is None",
            ],
            cwd=smoke,
            check=True,
        )
        subprocess.run(
            [str(binaries / ("tisam.exe" if windows else "tisam")), "--help"],
            cwd=smoke,
            check=True,
        )


if __name__ == "__main__":
    main()
