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
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, parse_needle_output, get_anchored_sequences, add_labels_to_barplot, calculate_column_ic

def pairwise_align(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Perform Pairwise Alignment anchored to reference using EMBOSS Needle."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Pairwise"):
        gene = get_gene_name(f.stem)
        g_dir = org_base / gene / "pairwise_alignments"
        for d in ["fasta", "plots", "needle"]: (g_dir / d).mkdir(parents=True, exist_ok=True)
        recs = list(SeqIO.parse(f, "fasta"))
        ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
        if not ref_rec: continue
        ref_tmp = g_dir / "fasta" / "reference.fasta"
        SeqIO.write(ref_rec, ref_tmp, "fasta")
        if slurm:
            script = g_dir / "slurm_pairwise.sh"
            content = f"#!/bin/bash\n#SBATCH --job-name=pair_{gene[:10]}\n#SBATCH --output={g_dir}/job_pairwise.out\n#SBATCH --mem={mem}\n#SBATCH --time=01:00:00\n\nmodule load EMBOSS\n{sys.executable} {Path(__file__).resolve().parent.parent / 'main.py'} pairwise internal-pairwise-run \"{f.resolve()}\" \"{ref_tmp.resolve()}\" \"{g_dir.resolve()}\" \"{reference_organism}\" \"{gene}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_pairwise_run(f, ref_tmp, g_dir, reference_organism, gene)

def internal_pairwise_run(fasta_file: Path, ref_tmp: Path, g_dir: Path, reference_organism: str, gene_name: str):
    recs = list(SeqIO.parse(fasta_file, "fasta"))
    metrics, anchored_recs = [], []
    window_data = []
    ref_rec = list(SeqIO.parse(ref_tmp, "fasta"))[0]
    anchored_recs.append(SeqRecord(ref_rec.seq, id=f"REF_{reference_organism.replace(' ', '_')}", description=""))
    for i, r in enumerate(recs):
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        
        q_tmp = g_dir / "fasta" / f"temp_{org_name.replace(' ', '_')}_{i}.fasta"
        SeqIO.write(r, q_tmp, "fasta")
        out_n = g_dir / "needle" / f"{org_name.replace(' ', '_')}_{i}_vs_ref.needle"
        subprocess.run(["needle", "-asequence", str(ref_tmp), "-bsequence", str(q_tmp), "-outfile", str(out_n), "-datafile", "EDNAFULL", "-gapopen", "10", "-gapextend", "0.5"], check=True, capture_output=True)
        
        metrics.append({**parse_needle_output(out_n), "Organism": org_name})
        ref_anch, que_anch = get_anchored_sequences(out_n)
        if que_anch: anchored_recs.append(SeqRecord(Seq(que_anch), id=f"{org_name.replace(' ', '_')}_{i}", description=""))
        
        # Windowed identity calculation (300bp window)
        aln = AlignIO.read(out_n, "emboss")
        ref_aln, que_aln = str(aln[0].seq), str(aln[1].seq)
        ref_to_col = [idx for idx, b in enumerate(ref_aln) if b != '-']
        results = []
        for start_ref in range(len(ref_to_col) - 299):
            start_col, end_col = ref_to_col[start_ref], ref_to_col[start_ref + 299]
            span = end_col - start_col + 1
            ident = sum(1 for a, b in zip(ref_aln[start_col:end_col+1], que_aln[start_col:end_col+1]) if a == b and a != '-') / span
            results.append({"RefPos": start_ref + 1, "Identity": ident, "Organism": org_name})
        if results:
            window_data.append(pd.DataFrame(results))
            
        q_tmp.unlink()
        
    if metrics:
        df = pd.DataFrame(metrics); df.to_csv(g_dir / "pairwise_metrics.csv", index=False)
        plt.figure(figsize=(12, 7)); ax = sns.barplot(data=df.melt(id_vars="Organism", value_vars=["Similarity", "Gaps"]), x="Organism", y="value", hue="variable")
        avg_s = df["Similarity"].mean(); plt.axhline(avg_s, color='blue', ls='--', alpha=0.7, label=f'Avg Similarity ({avg_s:.1f}%)')
        plt.xticks(rotation=45, ha='right'); plt.title(f"Pairwise Metrics: {gene_name}"); plt.ylabel("Percentage (%)"); plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); add_labels_to_barplot(ax); plt.tight_layout(rect=[0, 0.03, 0.85, 0.90]); plt.savefig(g_dir / "plots" / "metrics_comparison.png", dpi=300, bbox_inches="tight"); plt.close()

    if window_data:
        wdf = pd.concat(window_data)
        # Average across multiple sequences of the same organism at each position
        org_wdf = wdf.groupby(["Organism", "RefPos"]).mean().reset_index()
        avg_df = wdf.select_dtypes(include=[np.number]).groupby(wdf["RefPos"]).mean().reset_index()
        
        # Save full windowed identity by organism to CSV
        org_wdf.to_csv(g_dir / "organism_windowed_identity.csv", index=False)
        # Save average windowed identity to CSV
        avg_df.to_csv(g_dir / "windowed_identity.csv", index=False)
        
        plt.figure(figsize=(12, 6))
        unique_orgs = org_wdf["Organism"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
        
        for org, color in zip(unique_orgs, colors):
            o_data = org_wdf[org_wdf["Organism"] == org]
            plt.plot(o_data["RefPos"], o_data["Identity"]*100, color=color, alpha=0.4, lw=1, label=org)
            
        plt.plot(avg_df["RefPos"], avg_df["Identity"]*100, label="Average Identity", color="red", lw=2, linestyle="--")
        plt.title(f"Windowed Identity (300bp): {gene_name}")
        plt.xlabel("Window Start Position (Reference bp)")
        plt.ylabel("% Identity"); plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small'); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(g_dir / "plots" / "windowed_identity.png", dpi=300, bbox_inches="tight"); plt.close()

    if len(anchored_recs) > 1:
        a_fa = g_dir / "fasta" / "anchored_alignment.fasta"
        SeqIO.write(anchored_recs, a_fa, "fasta")
        try:
            mv = MsaViz(a_fa, format="fasta", show_consensus=True)
            fig = mv.plotfig(); fig.suptitle(f"Reference-Anchored Alignment: {gene_name}\n(Gaps in reference removed)", fontsize=14)
            fig.subplots_adjust(top=0.85); fig.savefig(g_dir / "plots" / "anchored_alignment.png", dpi=300, bbox_inches="tight"); plt.close(fig)
            
            # Calculate IC for this anchored alignment
            aln_recs = list(SeqIO.parse(a_fa, "fasta"))
            aln_len = len(aln_recs[0].seq)
            ic_results = []
            for i in range(aln_len):
                col = [str(r.seq[i]) for r in aln_recs]
                ic_results.append({"RefPos": i + 1, "IC": calculate_column_ic(col)})
            
            ic_df = pd.DataFrame(ic_results)
            ic_df.to_csv(g_dir / "anchored_information_content.csv", index=False)
            
            # Plot IC
            plt.figure(figsize=(15, 5))
            ic_df["IC_smoothed"] = ic_df["IC"].rolling(window=30, center=True, min_periods=1).mean()
            plt.fill_between(ic_df["RefPos"], ic_df["IC"], color="skyblue", alpha=0.2, label="Raw IC")
            plt.plot(ic_df["RefPos"], ic_df["IC_smoothed"], color="Slateblue", alpha=0.8, lw=2, label="Smoothed IC (30bp)")
            plt.ylim(0, 2.1); plt.xlabel("Reference Position (bp)"); plt.ylabel("Information Content (bits)")
            plt.title(f"Pairwise-Anchored Information Content: {gene_name}"); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.legend(loc='upper right')
            plt.tight_layout(); plt.savefig(g_dir / "plots" / "information_content.png", dpi=300); plt.close()
            
        except Exception as e: print(f"Error plotting anchored: {e}")
