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
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import List, Dict, Optional
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, add_labels_to_barplot
from loguru import logger

app = typer.Typer()

@app.command()
def bowtie_match(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Find 21-mer matches with up to 2 mismatches using a single Bowtie run."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Bowtie Match"):
        gene = get_gene_name(f.stem)
        bt_dir = org_base / gene / "bowtie_matches"
        for d in ["fasta", "plots", "index", "results"]: (bt_dir / d).mkdir(parents=True, exist_ok=True)
        
        recs = list(SeqIO.parse(f, "fasta"))
        ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
        if not ref_rec: continue
        
        # 1. Create 21-mers from reference gene
        kmers_file = bt_dir / "fasta" / "ref_21mers.fasta"
        kmers = []
        ref_seq_str = str(ref_rec.seq)
        for i in range(len(ref_seq_str) - 20):
            kmer = ref_seq_str[i:i+21]
            kmers.append(SeqRecord(Seq(kmer), id=f"kmer_{i+1}_pos_{i+1}", description=""))
        SeqIO.write(kmers, kmers_file, "fasta")
        
        # 2. Prepare NTO database (all sequences except reference)
        nto_file = bt_dir / "fasta" / "nto_sequences.fasta"
        ntos = []
        for r in recs:
            org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
            if org_name.lower() == reference_organism.lower(): continue
            ntos.append(r)
        if not ntos: continue
        SeqIO.write(ntos, nto_file, "fasta")
        
        if slurm:
            script = bt_dir / "slurm_bowtie.sh"
            content = f"#!/bin/bash\n#SBATCH --job-name=bt_{gene[:10]}\n#SBATCH --output={bt_dir}/job_bowtie.out\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\nmodule load bowtie\ndsrna-pipeline internal-bowtie-run \"{kmers_file.resolve()}\" \"{nto_file.resolve()}\" \"{bt_dir.resolve()}\" \"{gene}\" \"{reference_organism}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_bowtie_run(kmers_file, nto_file, bt_dir, gene, reference_organism)

@app.command(hidden=True)
def internal_bowtie_run(kmers_file: Path, nto_file: Path, bt_dir: Path, gene_name: str, reference_organism: str):
    idx_base = bt_dir / "index" / "nto_idx"
    
    # 1. Build index
    subprocess.run(["bowtie-build", str(nto_file), str(idx_base)], check=True, capture_output=True)
    
    # 2. Run Bowtie once with -v 2 (up to 2 mismatches)
    out_file = bt_dir / "results" / "matches_raw.txt"
    # -f is needed for FASTA input
    cmd = ["bowtie", "-f", "-v", "2", "-a", "--best", "--strata", str(idx_base), str(kmers_file), str(out_file)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Bowtie failed with exit code {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        raise e
    
    if not out_file.exists() or out_file.stat().st_size == 0:
        logger.info(f"No bowtie matches found for {gene_name}")
        return

    # 3. Parse and analyze
    df = pd.read_csv(out_file, sep="\t", header=None, names=["kmer_id", "strand", "target", "offset", "seq", "qual", "others", "mismatches"])
    
    # Function to count mismatches from Bowtie's 8th column
    def count_mms(mm_str):
        if pd.isna(mm_str) or mm_str == "" or str(mm_str).strip() == "":
            return 0
        return str(mm_str).count(",") + 1

    df["mismatches_count"] = df["mismatches"].apply(count_mms)
    df["RefPos"] = df["kmer_id"].str.extract(r'pos_(\d+)').astype(int)
    
    # Map target ID to Organism name
    nto_recs = list(SeqIO.parse(nto_file, "fasta"))
    id_to_org = {}
    for r in nto_recs:
        try:
            meta = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1])
            id_to_org[r.id] = meta.get("organism_name", "Unknown")
        except:
            id_to_org[r.id] = "Unknown"
    df["Organism"] = df["target"].map(id_to_org)
    df.to_csv(bt_dir / "results" / "all_matches.csv", index=False)
    
    # 4. Plots
    max_pos = df["RefPos"].max()
    all_pos = np.arange(1, max_pos + 1)
    
    # Deduplicate for positional counts (per kmer location)
    df_dedup = df.drop_duplicates(subset=["kmer_id", "offset", "mismatches_count"])
    v0 = df_dedup[df_dedup["mismatches_count"] == 0].groupby("RefPos").size().reindex(all_pos, fill_value=0)
    v1 = df_dedup[df_dedup["mismatches_count"] == 1].groupby("RefPos").size().reindex(all_pos, fill_value=0)
    v2 = df_dedup[df_dedup["mismatches_count"] == 2].groupby("RefPos").size().reindex(all_pos, fill_value=0)
    
    # 4.1. Three-panel Positional Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].bar(all_pos, v0, width=1.0, color='green', alpha=0.8)
    axes[0].set_title(f"Perfect Matches - {gene_name}")
    axes[0].set_ylabel("Hit Count")
    axes[0].text(0.02, 0.95, f"Total: {v0.sum()}", transform=axes[0].transAxes, bbox=dict(facecolor='lightgreen', alpha=0.5))
    
    axes[1].bar(all_pos, v1, width=1.0, color='orange', alpha=0.8)
    axes[1].set_title(f"1 Mismatch Alignments - {gene_name}")
    axes[1].set_ylabel("Hit Count")
    axes[1].text(0.02, 0.95, f"Total: {v1.sum()}", transform=axes[1].transAxes, bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    axes[2].bar(all_pos, v2, width=1.0, color='red', alpha=0.8)
    axes[2].set_title(f"2 Mismatch Alignments - {gene_name}")
    axes[2].set_ylabel("Hit Count")
    axes[2].set_xlabel("Reference Position (bp)")
    axes[2].text(0.02, 0.95, f"Total: {v2.sum()}", transform=axes[2].transAxes, bbox=dict(facecolor='lightcoral', alpha=0.5))
    
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(bt_dir / "plots" / "conservation_plot.png", dpi=300)
    plt.close()

    # 4.2. Per-Organism Stacked Bar Plot
    # Aggregate matches per organism and mismatch level
    org_counts = df.groupby(["Organism", "mismatches_count"]).size().unstack(fill_value=0)
    # Ensure all levels 0, 1, 2 exist
    for col in [0, 1, 2]:
        if col not in org_counts.columns: org_counts[col] = 0
    org_counts = org_counts[[0, 1, 2]].sort_values(by=0, ascending=False)
    
    ax = org_counts.plot(kind='bar', stacked=True, figsize=(12, 7), color=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title(f"Total 21-mer Matches per Organism: {gene_name}")
    plt.ylabel("Match Count")
    plt.xlabel("Organism")
    plt.xticks(rotation=45, ha='right')
    plt.legend(["Perfect", "1 Mismatch", "2 Mismatches"])
    plt.tight_layout()
    plt.savefig(bt_dir / "plots" / "matches_per_organism.png", dpi=300)
    plt.close()

    # 4.3. Windowed Stacked Plot (300bp)
    # Sum counts in 300bp windows (Forward-looking: index is window start)
    window_size = 300
    # We use a trick to get a forward-looking rolling sum: reverse, roll, reverse
    win_v0 = v0[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]
    win_v1 = v1[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]
    win_v2 = v2[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]
    
    plt.figure(figsize=(15, 6))
    plt.stackplot(all_pos, win_v0, win_v1, win_v2, labels=["Perfect", "1 MM", "2 MM"], colors=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
    plt.title(f"300bp Windowed Matches (Cumulative): {gene_name}")
    plt.xlabel("Window Start Position (Reference bp)")
    plt.ylabel("Total Matches in Window")
    plt.xlim(1, max_pos)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(bt_dir / "plots" / "windowed_matches_stacked.png", dpi=300)
    plt.close()

    # 5. Save windowed metrics for aggregation
    window_summary = pd.DataFrame({
        "RefPos": all_pos,
        "Hits_0mm": win_v0.values,
        "Hits_1mm": win_v1.values,
        "Hits_2mm": win_v2.values
    })
    window_summary.to_csv(bt_dir / "results" / "windowed_bowtie_summary.csv", index=False)

if __name__ == "__main__":
    app()
