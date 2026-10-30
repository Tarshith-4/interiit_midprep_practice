"""Generate batch CSV exports outside the Streamlit UI."""
from engine import export_outputs


if __name__ == "__main__":
    allocations_path, summary_path = export_outputs()
    print(f"Wrote {allocations_path}")
    print(f"Wrote {summary_path}")

