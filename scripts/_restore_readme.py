"""One-off: restore full README from agent transcript."""
import json
from pathlib import Path

TRANSCRIPT = Path(
    r"C:\Users\Lorenzo\.cursor\projects\d-github-segmentation-rl\agent-transcripts"
    r"\20bfa264-bd97-46a8-ba3c-5cb35dd14d07\20bfa264-bd97-46a8-ba3c-5cb35dd14d07.jsonl"
)
ROOT = Path(__file__).resolve().parents[1]

REPLACEMENTS = [
    ("# Pipeline 3 — Doppio RL", "# Brain Tumor Segmentation — Doppio RL"),
    ("run_full_pipeline3.py", "run_full_pipeline.py"),
    ("environment3.py", "environment_roi.py"),
    ("environment4.py", "environment_refine.py"),
    ("agent2.py", "agent_roi.py"),
    ("agent3.py", "agent_refine.py"),
    ("rl_trainer2.py", "rl_trainer.py"),
    ("final_results3.csv", "final_results.csv"),
    ("per_sample_results3.csv", "per_sample_results.csv"),
    ("report3.md", "report.md"),
    ("Il **Pipeline 3**", "Il **pipeline attuale**"),
    ("Il Pipeline 3 è", "Il pipeline attuale è"),
    ("pipeline 3", "pipeline"),
    ("`agent2/3`, `environment3/4`", "`agent_roi`, `agent_refine`, `environment_roi`, `environment_refine`"),
    (
        "## 2. Confronto con gli altri pipeline\n\n| Aspetto | `run_full_pipeline.py` | `run_full_pipeline2.py` | **`run_full_pipeline.py`** |",
        "## 2. Confronto con le versioni precedenti (`old/`)\n\n"
        "| Aspetto | `old/scripts/run_full_pipeline.py` | `old/scripts/run_full_pipeline2.py` | **Pipeline attuale** |",
    ),
    (
        "| Moduli dedicati | `agent.py`, `environment.py` | `environment2.py` | **`agent_roi`, `agent_refine`, `environment_roi`, `environment_refine`** |",
        "| Moduli RL | `agent.py`, `environment.py` | `environment2.py` | **`agent_roi`, `agent_refine`, `environment_roi`, `environment_refine`** |",
    ),
    (
        "mantenendo file separati per non sovrascrivere il codice esistente.",
        "Il codice legacy è in `old/`.",
    ),
    (
        "- Moduli indipendenti (`agent_roi`, `agent_refine`, `environment_roi`, `environment_refine`) non interferiscono con il pipeline U-Net esistente.",
        "- Moduli dedicati sostituiscono il pipeline U-Net legacy conservato in `old/`.",
    ),
    ("### File sorgente del pipeline 3 (non modificano il pipeline 1)", "### File sorgente del pipeline"),
    ("- Pipeline U-Net + RL: `scripts/run_full_pipeline.py`\n- Pipeline U-Net + ROI Finder: `scripts/run_full_pipeline2.py`", "- Pipeline legacy: `old/scripts/`"),
    (
        "*Documento generato per il progetto `segmentation_rl`. Per aggiornamenti al codice, verificare che questo README rifletta la versione corrente di `run_full_pipeline.py` e `rl_trainer.py`.*",
        "*Relazione tecnica del progetto `segmentation_rl` — allineata a `scripts/run_full_pipeline.py`.*",
    ),
]

QUICKSTART = """```bash
pip install -r requirements.txt
python scripts/run_full_pipeline.py
```

"""


def main():
    text = None
    with TRANSCRIPT.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("role") != "assistant":
                continue
            for part in obj.get("message", {}).get("content", []):
                if part.get("type") == "tool_use" and part.get("name") == "Write":
                    inp = part.get("input", {})
                    if inp.get("path", "").endswith("README_pipeline3.md"):
                        text = inp["contents"]
                        break
            if text:
                break
    if not text:
        raise SystemExit("Transcript content not found")

    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    if QUICKSTART.strip() not in text:
        text = text.replace(
            "**Dataset di riferimento:**",
            QUICKSTART + "**Dataset di riferimento:**",
            1,
        )

    out = ROOT / "README.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
