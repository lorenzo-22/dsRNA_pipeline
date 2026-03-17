import json
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO, AlignIO
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, run_vienna_accessibility, parse_gene_ids

app = typer.Typer()

@app.command()
def accessibility(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input-file"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Calculate reference-anchored windowed accessibility."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    gene_configs = parse_gene_ids(input_file)
    gene_to_windows = {g['description'].replace(' ', '_'): g['windows'] for g in gene_configs}

    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Accessibility"):
        gene = get_gene_name(f.stem)
        gene_safe = gene.replace(' ', '_')
        pw_dir = org_base / gene / "alignments" / "pairwise"
        acc_base_dir = org_base / gene # For window folders
        
        windows = gene_to_windows.get(gene_safe, [300])
        win_str = ",".join(map(str, windows))

        if slurm:
            script = pw_dir.parent / "slurm_acc.sh" 
            log_file = pw_dir.parent / "job_acc.out"
            content = f"#!/bin/bash\n#SBATCH --job-name=acc_{gene[:10]}\n#SBATCH --output={log_file.resolve()}\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\n{sys.prefix}/bin/dsrna-pipeline internal-accessibility-run \"{f.resolve()}\" \"{pw_dir.resolve()}\" \"{acc_base_dir.resolve()}\" \"{reference_organism}\" \"{gene}\" --window-sizes \"{win_str}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_accessibility_run(f, pw_dir, acc_base_dir, reference_organism, gene, win_str)

@app.command(hidden=True)
def internal_accessibility_run(
    fasta_file: Path, 
    pw_dir: Path, 
    acc_base_dir: Path, 
    reference_organism: str, 
    gene_name: str,
    window_sizes: str = typer.Option("300", "--window-sizes")
):
    recs = list(SeqIO.parse(fasta_file, "fasta"))
    ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
    if not ref_rec: return
    
    ref_acc = run_vienna_accessibility(str(ref_rec.seq))
    nto_accs = {}
    for r in recs:
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        nto_accs[org_name] = run_vienna_accessibility(str(r.seq))
        
    windows = [int(x) for x in window_sizes.split(",")]
    for ws in windows:
        acc_dir = acc_base_dir / str(ws) / "accessibility"
        for d in ["plots", "data"]: (acc_dir / d).mkdir(parents=True, exist_ok=True)
        run_windowed_accessibility(recs, ref_rec, ref_acc, nto_accs, pw_dir, acc_dir, reference_organism, gene_name, ws)

def run_windowed_accessibility(recs, ref_rec, ref_acc, nto_accs, pw_dir, acc_dir, reference_organism, gene_name, window_size):
    if len(ref_acc) < window_size:
        logger.warning(f"Gene '{gene_name}' (ref len {len(ref_acc)}) is shorter than window size {window_size}. Skipping accessibility windowed analysis.")
        return

    window_data = []
    for r in recs:
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        
        needle_files = list((pw_dir / "needle").glob(f"{org_name.replace(' ', '_')}_*_vs_ref.needle"))
        if not needle_files: continue
        
        aln = AlignIO.read(needle_files[0], "emboss")
        ref_aln, que_aln = str(aln[0].seq), str(aln[1].seq)
        que_acc = nto_accs[org_name]
        
        ref_to_col, que_to_col = [idx for idx, b in enumerate(ref_aln) if b != '-'], [idx for idx, b in enumerate(que_aln) if b != '-']
        
        use_len = min(len(que_to_col), len(que_acc))
        col_to_que_acc = {que_to_col[q_idx]: que_acc[q_idx] for q_idx in range(use_len)}
        
        results = []
        for start_ref in range(len(ref_to_col) - (window_size - 1)):
            start_col, end_col = ref_to_col[start_ref], ref_to_col[start_ref + (window_size - 1)]
            span = end_col - start_col + 1
            win_q_acc = sum(col_to_que_acc.get(c, 0.0) for c in range(start_col, end_col+1)) / span
            results.append({"RefPos": start_ref + 1, "NTO_Acc": win_q_acc, "Organism": org_name})
        if results: window_data.append(pd.DataFrame(results))
        
    if window_data:
        wdf = pd.concat(window_data)
        org_wdf = wdf.groupby(["Organism", "RefPos"]).mean().reset_index()
        avg_df = wdf.select_dtypes(include=[np.number]).groupby(wdf["RefPos"]).mean().reset_index()
        
        ref_win_acc = [np.mean(ref_acc[idx:idx+window_size]) for idx in range(len(ref_acc)-(window_size - 1))]
        ref_acc_series = pd.Series(ref_win_acc, index=range(1, len(ref_win_acc) + 1))
        avg_df["Ref_Acc"] = avg_df["RefPos"].map(ref_acc_series)
        
        org_wdf.to_csv(acc_dir / "data" / "organism_windowed_accessibility.csv", index=False)
        avg_df.to_csv(acc_dir / "data" / "windowed_analysis.csv", index=False)
        
        plt.figure(figsize=(12, 6))
        unique_orgs = org_wdf["Organism"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
        for org, color in zip(unique_orgs, colors):
            o_data = org_wdf[org_wdf["Organism"] == org]
            plt.plot(o_data["RefPos"], o_data["NTO_Acc"], color=color, alpha=0.4, lw=1, label=org)
            
        plt.plot(range(1, len(ref_win_acc)+1), ref_win_acc, label=f"Ref ({reference_organism})", color="black", lw=2.5)
        plt.plot(avg_df["RefPos"], avg_df["NTO_Acc"], label="Average NTOs", color="red", lw=2, linestyle="--")
        plt.title(f"Windowed Accessibility ({window_size}bp): {gene_name}"); plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(acc_dir / "plots" / "ref_vs_avg_accessibility.png", dpi=300, bbox_inches="tight"); plt.close()
