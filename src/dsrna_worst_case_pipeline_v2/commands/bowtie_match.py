import json
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, parse_gene_ids
from loguru import logger

app = typer.Typer()

@app.command()
def bowtie(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input-file"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Find 21-mer matches and prepare for windowed analysis."""
    ref_safe = reference_organism.replace(" ", "_")
    gene_configs = parse_gene_ids(input_file)
    gene_to_windows = {g['description'].replace(' ', '_'): g['windows'] for g in gene_configs}

    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Bowtie Match"):
        gene = get_gene_name(f.stem)
        gene_safe = gene.replace(' ', '_')
        base_dir = output_base / "Organisms" / ref_safe / gene
        aln_dir = base_dir / "alignments" / "bowtie_matches"
        for d in ["fasta", "plots", "index", "results", "slurm"]: (aln_dir / d).mkdir(parents=True, exist_ok=True)
        
        recs = list(SeqIO.parse(f, "fasta"))
        ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
        if not ref_rec: continue
        
        kmers_file = aln_dir / "fasta" / "ref_21mers.fasta"
        nto_file = aln_dir / "fasta" / "nto_sequences.fasta"
        
        if not kmers_file.exists():
            kmers = []
            ref_seq_str = str(ref_rec.seq)
            for i in range(len(ref_seq_str) - 20):
                kmer = ref_seq_str[i:i+21]
                kmers.append(SeqRecord(Seq(kmer), id=f"kmer_{i+1}_pos_{i+1}", description=""))
            SeqIO.write(kmers, kmers_file, "fasta")
        
        if not nto_file.exists():
            ntos = [r for r in recs if reference_organism.lower() not in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()]
            if ntos: SeqIO.write(ntos, nto_file, "fasta")
        
        windows = gene_to_windows.get(gene_safe, [300])
        win_str = ",".join(map(str, windows))

        if slurm:
            script = aln_dir / "slurm" / f"bowtie_{gene_safe}.sh"
            log_file = aln_dir / "slurm" / "job.out"
            content = f"#!/bin/bash\n#SBATCH --job-name=bt_{gene[:10]}\n#SBATCH --output={log_file.resolve()}\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\nmodule load bowtie\n{sys.prefix}/bin/dsrna-pipeline internal-bowtie-run \"{kmers_file.resolve()}\" \"{nto_file.resolve()}\" \"{aln_dir.resolve()}\" \"{gene}\" \"{reference_organism}\" --window-sizes \"{win_str}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_bowtie_run(kmers_file, nto_file, aln_dir, gene, reference_organism, win_str)

@app.command(hidden=True)
def internal_bowtie_run(
    kmers_file: Path, 
    nto_file: Path, 
    aln_dir: Path, 
    gene_name: str, 
    reference_organism: str,
    window_sizes: str = typer.Option("300", "--window-sizes")
):
    idx_base = aln_dir / "index" / "nto_idx"
    if not (aln_dir / "results" / "all_matches.csv").exists():
        subprocess.run(["bowtie-build", str(nto_file), str(idx_base)], check=True, capture_output=True)
        out_file = aln_dir / "results" / "matches_raw.txt"
        subprocess.run(["bowtie", "-f", "-v", "2", "-a", "--best", "--strata", str(idx_base), str(kmers_file), str(out_file)], check=True, capture_output=True)
        
        if not out_file.exists() or out_file.stat().st_size == 0: return

        df = pd.read_csv(out_file, sep="\t", header=None, names=["kmer_id", "strand", "target", "offset", "seq", "qual", "others", "mismatches"])
        df["mismatches_count"] = df["mismatches"].apply(lambda x: 0 if pd.isna(x) else str(x).count(",") + 1)
        df["RefPos"] = df["kmer_id"].str.extract(r'pos_(\d+)').astype(int)
        
        nto_recs = list(SeqIO.parse(nto_file, "fasta"))
        id_to_org = {}
        for r in nto_recs:
            try:
                meta = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1])
                id_to_org[r.id] = meta.get("organism_name", "Unknown")
            except: id_to_org[r.id] = "Unknown"
        df["Organism"] = df["target"].map(id_to_org)
        df.to_csv(aln_dir / "results" / "all_matches.csv", index=False)

    # Run windowed analysis
    windows = [int(x) for x in window_sizes.split(",")]
    base_dir = aln_dir.parent.parent
    for ws in windows:
        win_dir = base_dir / str(ws) / "bowtie_matches"
        win_dir.mkdir(parents=True, exist_ok=True)
        bowtie_window_analysis(aln_dir, win_dir, gene_name, ws)

def bowtie_window_analysis(aln_dir: Path, win_dir: Path, gene_name: str, window_size: int):
    raw_matches = aln_dir / "results" / "all_matches.csv"
    if not raw_matches.exists(): return
    df = pd.read_csv(raw_matches)
    
    max_pos = df["RefPos"].max()
    if max_pos < window_size:
        logger.warning(f"Gene '{gene_name}' (max pos {max_pos}) is shorter than window size {window_size}. Skipping bowtie windowed summary.")
        return

    all_pos = np.arange(1, max_pos + 1)
    df_dedup = df.drop_duplicates(subset=["kmer_id", "offset", "mismatches_count"])
    
    v0 = df_dedup[df_dedup["mismatches_count"] == 0].groupby("RefPos").size().reindex(all_pos, fill_value=0)
    v1 = df_dedup[df_dedup["mismatches_count"] == 1].groupby("RefPos").size().reindex(all_pos, fill_value=0)
    v2 = df_dedup[df_dedup["mismatches_count"] == 2].groupby("RefPos").size().reindex(all_pos, fill_value=0)
    
    win_v0 = v0[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]
    win_v1 = v1[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]
    win_v2 = v2[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]
    
    pd.DataFrame({"RefPos": all_pos, "Hits_0mm": win_v0.values, "Hits_1mm": win_v1.values, "Hits_2mm": win_v2.values}).to_csv(win_dir / "windowed_bowtie_summary.csv", index=False)
    
    plt.figure(figsize=(15, 6))
    plt.stackplot(all_pos, win_v0, win_v1, win_v2, labels=["Perfect", "1 MM", "2 MM"], colors=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
    plt.title(f"{window_size}bp Windowed Matches: {gene_name}"); plt.xlabel("Window Start Position (Reference bp)"); plt.ylabel("Total Matches in Window")
    plt.legend(loc='upper right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(win_dir / "windowed_matches_stacked.png", dpi=300); plt.close()
