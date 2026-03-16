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
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, run_vienna_accessibility

def calculate_accessibility(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Calculate reference-anchored windowed accessibility and conservation."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Accessibility"):
        gene = get_gene_name(f.stem)
        pw_dir = org_base / gene / "pairwise_alignments"
        acc_dir = org_base / gene / "accessibility"
        for d in ["plots", "data"]: (acc_dir / d).mkdir(parents=True, exist_ok=True)
        ref_tmp = pw_dir / "fasta" / "reference.fasta"
        if not ref_tmp.exists(): continue
        if slurm:
            script = acc_dir / "slurm_acc.sh"
            content = f"#!/bin/bash\n#SBATCH --job-name=acc_{gene[:10]}\n#SBATCH --output={acc_dir}/job_acc.out\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\ndsrna-pipeline internal-accessibility-run \"{f.resolve()}\" \"{ref_tmp.resolve()}\" \"{pw_dir.resolve()}\" \"{acc_dir.resolve()}\" \"{reference_organism}\" \"{gene}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_accessibility_run(f, ref_tmp, pw_dir, acc_dir, reference_organism, gene)

def internal_accessibility_run(fasta_file: Path, ref_tmp: Path, pw_dir: Path, acc_dir: Path, reference_organism: str, gene_name: str):
    recs = list(SeqIO.parse(fasta_file, "fasta"))
    ref_rec = list(SeqIO.parse(ref_tmp, "fasta"))[0]
    ref_acc = run_vienna_accessibility(str(ref_rec.seq))
    window_data = []
    for i, r in enumerate(recs):
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        needle_file = pw_dir / "needle" / f"{org_name.replace(' ', '_')}_{i}_vs_ref.needle"
        if not needle_file.exists(): continue
        aln = AlignIO.read(needle_file, "emboss")
        ref_aln, que_aln = str(aln[0].seq), str(aln[1].seq)
        que_acc = run_vienna_accessibility(str(r.seq))
        ref_to_col, que_to_col = [idx for idx, b in enumerate(ref_aln) if b != '-'], [idx for idx, b in enumerate(que_aln) if b != '-']
        col_to_que_acc = {col: que_acc[q_idx] for q_idx, col in enumerate(que_to_col)}
        results = []
        for start_ref in range(len(ref_to_col) - 299):
            start_col, end_col = ref_to_col[start_ref], ref_to_col[start_ref + 299]
            span = end_col - start_col + 1
            ident = sum(1 for a, b in zip(ref_aln[start_col:end_col+1], que_aln[start_col:end_col+1]) if a == b and a != '-') / span
            win_q_acc = sum(col_to_que_acc.get(c, 0.0) for c in range(start_col, end_col+1)) / span
            results.append({"RefPos": start_ref + 1, "Identity": ident, "NTO_Acc": win_q_acc, "Organism": org_name})
        window_data.append(pd.DataFrame(results))
    if window_data:
        wdf = pd.concat(window_data)
        # Average across multiple sequences of the same organism at each position
        org_wdf = wdf.groupby(["Organism", "RefPos"]).mean().reset_index()
        avg_df = wdf.select_dtypes(include=[np.number]).groupby(wdf["RefPos"]).mean().reset_index()
        
        # Add Reference Accessibility to the average dataframe
        ref_win_acc = [np.mean(ref_acc[idx:idx+300]) for idx in range(len(ref_acc)-299)]
        # Ensure lengths match (RefPos is 1-based start)
        ref_acc_series = pd.Series(ref_win_acc, index=range(1, len(ref_win_acc) + 1))
        avg_df["Ref_Acc"] = avg_df["RefPos"].map(ref_acc_series)
        
        # Save full windowed accessibility by organism to CSV
        org_wdf.to_csv(acc_dir / "data" / "organism_windowed_accessibility.csv", index=False)
        avg_df.to_csv(acc_dir / "data" / "windowed_analysis.csv", index=False)
        plt.figure(figsize=(12, 6))
        unique_orgs = org_wdf["Organism"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
        
        for org, color in zip(unique_orgs, colors):
            o_data = org_wdf[org_wdf["Organism"] == org]
            plt.plot(o_data["RefPos"], o_data["NTO_Acc"], color=color, alpha=0.4, lw=1, label=org)
            
        ref_win_acc = [np.mean(ref_acc[idx:idx+300]) for idx in range(len(ref_acc)-299)]
        plt.plot(range(1, len(ref_win_acc)+1), ref_win_acc, label=f"Ref ({reference_organism})", color="black", lw=2.5)
        plt.plot(avg_df["RefPos"], avg_df["NTO_Acc"], label="Average NTOs", color="red", lw=2, linestyle="--")
        plt.title(f"Windowed Accessibility: {gene_name}"); plt.xlabel("Window Start Position (Reference bp)"); plt.ylabel("Prob Unpaired"); plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small'); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(acc_dir / "plots" / "ref_vs_avg_accessibility.png", dpi=300, bbox_inches="tight"); plt.close()
        
        plt.figure(figsize=(12, 6))
        for org, color in zip(unique_orgs, colors):
            o_data = org_wdf[org_wdf["Organism"] == org]
            plt.plot(o_data["RefPos"], o_data["Identity"]*100, color=color, alpha=0.4, lw=1, label=org)
            
        plt.plot(avg_df["RefPos"], avg_df["Identity"]*100, label="Average Identity", color="red", lw=2, linestyle="--")
        plt.title(f"Windowed Identity: {gene_name}"); plt.xlabel("Window Start Position (Reference bp)"); plt.ylabel("% Identity"); plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small'); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(acc_dir / "plots" / "windowed_identity.png", dpi=300, bbox_inches="tight"); plt.close()
