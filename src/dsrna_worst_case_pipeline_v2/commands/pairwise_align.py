import json
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pymsaviz import MsaViz
from typing import List, Dict, Optional
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, parse_needle_output, get_anchored_sequences, add_labels_to_barplot, calculate_column_ic, parse_gene_ids

app = typer.Typer()

@app.command()
def pairwise(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input-file"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Perform Pairwise Alignment and prepare for windowed analysis."""
    ref_safe = reference_organism.replace(" ", "_")
    gene_configs = parse_gene_ids(input_file)
    gene_to_windows = {g['description'].replace(' ', '_'): g['windows'] for g in gene_configs}

    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Pairwise"):
        gene = get_gene_name(f.stem)
        gene_safe = gene.replace(' ', '_')
        base_dir = output_base / "Organisms" / ref_safe / gene
        aln_dir = base_dir / "alignments" / "pairwise"
        for d in ["fasta", "plots", "needle", "slurm"]: (aln_dir / d).mkdir(parents=True, exist_ok=True)
        
        recs = list(SeqIO.parse(f, "fasta"))
        ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
        if not ref_rec: continue
        
        ref_tmp = aln_dir / "fasta" / "reference.fasta"
        SeqIO.write(ref_rec, ref_tmp, "fasta")

        windows = gene_to_windows.get(gene_safe, [300])
        win_str = ",".join(map(str, windows))

        if slurm:
            script = aln_dir / "slurm" / f"pairwise_{gene_safe}.sh"
            log_file = aln_dir / "slurm" / "job.out"
            content = f"#!/bin/bash\n#SBATCH --job-name=pair_{gene[:10]}\n#SBATCH --output={log_file.resolve()}\n#SBATCH --mem={mem}\n#SBATCH --time=01:00:00\n\nmodule load EMBOSS\n{sys.prefix}/bin/dsrna-pipeline internal-pairwise-run \"{f.resolve()}\" \"{ref_tmp.resolve()}\" \"{aln_dir.resolve()}\" \"{reference_organism}\" \"{gene}\" --window-sizes \"{win_str}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_pairwise_run(f, ref_tmp, aln_dir, reference_organism, gene, win_str)

@app.command(hidden=True)
def internal_pairwise_run(
    fasta_file: Path, 
    ref_tmp: Path, 
    aln_dir: Path, 
    reference_organism: str, 
    gene_name: str,
    window_sizes: str = typer.Option("300", "--window-sizes")
):
    recs = list(SeqIO.parse(fasta_file, "fasta"))
    metrics, anchored_recs = [], []
    ref_rec = list(SeqIO.parse(ref_tmp, "fasta"))[0]
    anchored_recs.append(SeqRecord(ref_rec.seq, id=f"REF_{reference_organism.replace(' ', '_')}", description=""))
    
    for i, r in enumerate(recs):
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        
        q_tmp = aln_dir / "fasta" / f"temp_{org_name.replace(' ', '_')}_{i}.fasta"
        SeqIO.write(r, q_tmp, "fasta")
        out_n = aln_dir / "needle" / f"{org_name.replace(' ', '_')}_{i}_vs_ref.needle"
        if not out_n.exists():
            subprocess.run(["needle", "-asequence", str(ref_tmp), "-bsequence", str(q_tmp), "-outfile", str(out_n), "-datafile", "EDNAFULL", "-gapopen", "10", "-gapextend", "0.5"], check=True, capture_output=True)
        
        metrics.append({**parse_needle_output(out_n), "Organism": org_name})
        ref_anch, que_anch = get_anchored_sequences(out_n)
        if que_anch: anchored_recs.append(SeqRecord(Seq(que_anch), id=f"{org_name.replace(' ', '_')}_{i}", description=""))
        q_tmp.unlink()
        
    if metrics:
        df = pd.DataFrame(metrics); df.to_csv(aln_dir / "pairwise_metrics.csv", index=False)
        plt.figure(figsize=(12, 7)); ax = sns.barplot(data=df.melt(id_vars="Organism", value_vars=["Similarity", "Gaps"]), x="Organism", y="value", hue="variable")
        avg_s = df["Similarity"].mean(); plt.axhline(avg_s, color='blue', ls='--', alpha=0.7, label=f'Avg Similarity ({avg_s:.1f}%)')
        plt.xticks(rotation=45, ha='right'); plt.title(f"Pairwise Metrics: {gene_name}"); plt.ylabel("Percentage (%)"); plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); add_labels_to_barplot(ax); plt.tight_layout(); plt.savefig(aln_dir / "plots" / "metrics_comparison.png", dpi=300); plt.close()

    if len(anchored_recs) > 1:
        a_fa = aln_dir / "fasta" / "anchored_alignment.fasta"
        SeqIO.write(anchored_recs, a_fa, "fasta")
        try:
            mv = MsaViz(a_fa, format="fasta", show_consensus=True)
            fig = mv.plotfig(); fig.savefig(aln_dir / "plots" / "anchored_alignment.png", dpi=300); plt.close(fig)
            ic_df = pd.DataFrame([{"RefPos": i + 1, "IC": calculate_column_ic([str(r.seq[i]) for r in anchored_recs])} for i in range(len(anchored_recs[0].seq))])
            ic_df.to_csv(aln_dir / "anchored_information_content.csv", index=False)
            plt.figure(figsize=(15, 5)); ic_df["IC_smoothed"] = ic_df["IC"].rolling(window=30, center=True, min_periods=1).mean()
            plt.fill_between(ic_df["RefPos"], ic_df["IC"], color="skyblue", alpha=0.2); plt.plot(ic_df["RefPos"], ic_df["IC_smoothed"], color="Slateblue", alpha=0.8, lw=2)
            plt.ylim(0, 2.1); plt.savefig(aln_dir / "plots" / "information_content.png", dpi=300); plt.close()
        except: pass

    # Run windowed analysis
    windows = [int(x) for x in window_sizes.split(",")]
    base_dir = aln_dir.parent.parent
    for ws in windows:
        win_dir = base_dir / str(ws) / "similarity" / "pairwise"
        win_dir.mkdir(parents=True, exist_ok=True)
        pairwise_window_analysis(aln_dir, win_dir, gene_name, ws, reference_organism)

def pairwise_window_analysis(aln_dir: Path, out_dir: Path, title: str, window_size: int, reference_organism: str):
    try:
        window_data = []
        needle_files = list((aln_dir / "needle").glob("*.needle"))
        if not needle_files:
            return

        # Check ref length from first needle file
        aln_test = AlignIO.read(needle_files[0], "emboss")
        ref_test = [idx for idx, b in enumerate(str(aln_test[0].seq)) if b != '-']
        if len(ref_test) < window_size:
            logger.warning(f"Gene '{title}' (ref len {len(ref_test)}) is shorter than window size {window_size}. Skipping pairwise windowed identity.")
            return

        for out_n in needle_files:
            org_name = out_n.stem.rsplit('_vs_ref', 1)[0].replace('_', ' ')
            aln = AlignIO.read(out_n, "emboss")
            ref_aln, que_aln = str(aln[0].seq), str(aln[1].seq)
            ref_to_col = [idx for idx, b in enumerate(ref_aln) if b != '-']
            
            results = []
            for start_ref in range(len(ref_to_col) - (window_size - 1)):
                start_col, end_col = ref_to_col[start_ref], ref_to_col[start_ref + (window_size - 1)]
                span = end_col - start_col + 1
                ident = sum(1 for a, b in zip(ref_aln[start_col:end_col+1], que_aln[start_col:end_col+1]) if a == b and a != '-') / span
                results.append({"RefPos": start_ref + 1, "Identity": ident, "Organism": org_name})
            
            if results:
                window_data.append(pd.DataFrame(results))
        
        if window_data:
            wdf = pd.concat(window_data)
            org_wdf = wdf.groupby(["Organism", "RefPos"]).mean().reset_index()
            avg_df = wdf.select_dtypes(include=[np.number]).groupby(wdf["RefPos"]).mean().reset_index()
            org_wdf.to_csv(out_dir / "organism_windowed_identity.csv", index=False)
            avg_df.to_csv(out_dir / "windowed_identity.csv", index=False)
            
            plt.figure(figsize=(12, 6))
            unique_orgs = org_wdf["Organism"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
            for org, color in zip(unique_orgs, colors):
                o_data = org_wdf[org_wdf["Organism"] == org]
                plt.plot(o_data["RefPos"], o_data["Identity"]*100, color=color, alpha=0.4, lw=1, label=org)
            plt.plot(avg_df["RefPos"], avg_df["Identity"]*100, label="Average Identity", color="red", lw=2, linestyle="--")
            plt.title(f"Windowed Identity ({window_size}bp): {title}")
            plt.xlabel("Window Start Position (Reference bp)"); plt.ylabel("% Identity"); plt.ylim(0, 100)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small'); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / "windowed_identity.png", dpi=300, bbox_inches="tight"); plt.close()
    except Exception as e: print(f"Error in pairwise_window_analysis: {e}")
