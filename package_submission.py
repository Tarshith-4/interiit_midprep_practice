"""Create a clean reviewer ZIP containing source, bundled data, outputs, and docs."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "SmartWealth_Submission.zip"
INCLUDE = [
    "README.md", "REPORT.md", "REPORT.pdf", "requirements.txt", "requirements-dev.txt", "engine.py", "validation.py",
    "allocation.py", "projection.py", "explain.py", "planner.py", "app.py", "tests.py",
    "generate_outputs.py", "package_submission.py", "run.bat", "run.sh",
    "data/investor_profiles.csv", "data/investment_products.csv",
    "data/synthetic_monthly_returns.csv", "outputs/allocations_output.csv",
    "outputs/summary_output.csv", "outputs/SmartWealth_Demo.mp4", "docs/demo_script.md",
]
OPTIONAL = {"outputs/SmartWealth_Demo.mp4"}


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(DEST, "w", compression=ZIP_DEFLATED) as archive:
        for relative in INCLUDE:
            path = ROOT / relative
            if not path.is_file():
                if relative in OPTIONAL:
                    continue
                raise FileNotFoundError(path)
            archive.write(path, arcname=relative)
    print(f"Created {DEST}")


if __name__ == "__main__":
    main()

