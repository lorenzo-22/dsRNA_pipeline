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
from Bio.SeqRecord import SeqRecord
from pymsaviz import MsaViz
from typing import Optional, List
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, add_labels_to_barplot, calculate_information_content, parse_gene_ids

app = typer.Typer()

@app.command()
def align(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input-file"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Perform MSA and prepare for windowed analysis."""
    ref_safe = reference_organism.replace(" ", "_")
    gene_configs = parse_gene_ids(input_file)
    gene_to_windows = {g['description'].replace(' ', '_'): g['windows'] for g in gene_configs}

    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="MSA"):
        gene = get_gene_name(f.stem)
        gene_safe = gene.replace(' ', '_')
        base_dir = output_base / "Organisms" / ref_safe / gene
        aln_dir = base_dir / "alignments" / "msa"
        for d in ["fasta", "plots", "slurm"]: (aln_dir / d).mkdir(parents=True, exist_ok=True)
        
        recs = list(SeqIO.parse(f, "fasta"))
        if len(recs) < 2: continue
        
        # Find reference
        ref_idx = -1
        for i, r in enumerate(recs):
            try:
                meta = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1])
                if reference_organism.lower() in meta.get("organism_name", "").lower():
                    ref_idx = i
                    break
            except: continue

        renamed, ref_id = [], None
        for i, r in enumerate(recs):
            try:
                name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown").replace(" ", "_")
            except Exception:
                name = r.id
            rid = f"{name}_{i}"
            if i == ref_idx:
                ref_id = rid
                renamed.insert(0, SeqRecord(r.seq, id=rid, description=""))
            else:
                renamed.append(SeqRecord(r.seq, id=rid, description=""))
        
        temp_in = aln_dir / "fasta" / "renamed_orthologs.fasta"
        SeqIO.write(renamed, temp_in, "fasta")
        aln_file = aln_dir / "fasta" / "aligned.fasta"
        msa_plot = aln_dir / "plots" / "alignment.png"
        ic_plot = aln_dir / "plots" / "information_content.png"
        ic_csv = aln_dir / "information_content.csv"
        
        windows = gene_to_windows.get(gene_safe, [300])
        win_str = ",".join(map(str, windows))

        if slurm:
            script = aln_dir / "slurm" / f"msa_{gene_safe}.sh"
            ref_arg = f'--reference-id "{ref_id}"' if ref_id else ""
            log_file = aln_dir / "slurm" / "job.out"
            content = f"#!/bin/bash\n#SBATCH --job-name=msa_{gene[:10]}\n#SBATCH --output={log_file.resolve()}\n#SBATCH --cpus-per-task=4\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\nmodule load clustal-omega\nif [ ! -f \"{aln_file.resolve()}\" ]; then\n  clustalo -i {temp_in.resolve()} -o {aln_file.resolve()} --force --outfmt=fasta --threads=4\nfi\n{sys.prefix}/bin/dsrna-pipeline internal-msa-plot {aln_file.resolve()} {msa_plot.resolve()} {ic_plot.resolve()} {ic_csv.resolve()} \"{gene}\" {ref_arg} --window-sizes \"{win_str}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            if not aln_file.exists():
                subprocess.run(["bash", "-c", f"module load clustal-omega && clustalo -i {temp_in} -o {aln_file} --force --outfmt=fasta"], check=True)
            internal_msa_plot(aln_file, msa_plot, ic_plot, ic_csv, gene, ref_id, windows)

@app.command(hidden=True)
def internal_msa_plot(
    aln_file: Path, 
    plot_path: Path, 
    ic_plot: Path, 
    ic_csv: Path, 
    title: str, 
    reference_id: Optional[str] = typer.Option(None, "--reference-id"),
    window_sizes: str = typer.Option("300", "--window-sizes")
):
    try:
        windows = [int(x) for x in window_sizes.split(",")]
        
        mv = MsaViz(aln_file, format="fasta", show_consensus=True)
        fig = mv.plotfig()
        fig.suptitle(f"MSA: {title}", fontsize=16)
        fig.subplots_adjust(top=0.85)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        recs = list(SeqIO.parse(aln_file, "fasta"))
        ref_rec = next((r for r in recs if reference_id == r.id), recs[0])
        ref_seq = str(ref_rec.seq)
        
        metrics = []
        for r in recs:
            if r.id == ref_rec.id: continue
            m = sum(1 for a, b in zip(ref_seq, str(r.seq)) if a == b and a != '-')
            t = sum(1 for a, b in zip(ref_seq, str(r.seq)) if a != '-' or b != '-')
            metrics.append({"Org": r.id.rsplit('_', 1)[0].replace('_', ' '), "Identity": (m/t)*100 if t > 0 else 0, "Gaps": (str(r.seq).count('-')/len(ref_seq))*100})
            
        if metrics:
            df = pd.DataFrame(metrics)
            plt.figure(figsize=(12, 7))
            ax = sns.barplot(data=df.melt(id_vars="Org", value_vars=["Identity", "Gaps"]), x="Org", y="value", hue="variable")
            avg_i = df["Identity"].mean()
            plt.axhline(avg_i, color='blue', ls='--', label=f'Avg Identity ({avg_i:.1f}%)')
            plt.xticks(rotation=45, ha='right'); plt.title(f"MSA Metrics: {title}"); plt.ylabel("%"); plt.ylim(0, 100)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); add_labels_to_barplot(ax); plt.tight_layout()
            plt.savefig(plot_path.parent / "msa_metrics_comparison.png", dpi=300, bbox_inches="tight"); plt.close()
            
        ic_df = calculate_information_content(aln_file, reference_id)
        if not ic_df.empty:
            ic_df.to_csv(ic_csv, index=False)
            plt.figure(figsize=(15, 5))
            ic_df["IC_smoothed"] = ic_df["IC"].rolling(window=30, center=True, min_periods=1).mean()
            plt.fill_between(ic_df["Position"], ic_df["IC"], color="skyblue", alpha=0.2); plt.plot(ic_df["Position"], ic_df["IC_smoothed"], color="Slateblue", alpha=0.8, lw=2)
            plt.ylim(0, 2.1); plt.savefig(ic_plot, dpi=300); plt.close()

        # Run windowed analysis
        base_dir = aln_file.parent.parent.parent
        for ws in windows:
            win_dir = base_dir / str(ws) / "similarity" / "msa"
            win_dir.mkdir(parents=True, exist_ok=True)
            msa_window_analysis(aln_file, win_dir, title, ws, reference_id)

    except Exception as e:
        print(f"Error in internal_msa_plot: {e}")

def msa_window_analysis(aln_file: Path, out_dir: Path, title: str, window_size: int, reference_id: Optional[str]):
    try:
        recs = list(SeqIO.parse(aln_file, "fasta"))
        ref_rec = next((r for r in recs if reference_id == r.id), recs[0])
        ref_seq = str(ref_rec.seq)
        ref_to_col = [i for i, b in enumerate(ref_seq) if b != '-']
        
        if len(ref_to_col) < window_size:
            logger.warning(f"Gene '{title}' (ref len {len(ref_to_col)}) is shorter than window size {window_size}. Skipping MSA windowed identity.")
            return

        win_results = []
        for r in recs:
            if r.id == ref_rec.id: continue
            q_seq, org = str(r.seq), r.id.rsplit('_', 1)[0].replace('_', ' ')
            for s in range(len(ref_to_col) - (window_size - 1)):
                sc, ec = ref_to_col[s], ref_to_col[s + (window_size - 1)]
                win_ref, win_que = ref_seq[sc:ec+1], q_seq[sc:ec+1]
                ident = sum(1 for a, b in zip(win_ref, win_que) if a == b and a != '-') / (ec - sc + 1)
                win_results.append({"RefPos": s + 1, "Identity": ident, "Org": org})
                
        if win_results:
            wdf = pd.DataFrame(win_results)
            org_wdf = wdf.groupby(["Org", "RefPos"]).mean().reset_index()
            avg_wdf = wdf.select_dtypes(include=[np.number]).groupby(wdf["RefPos"]).mean().reset_index()
            avg_wdf.rename(columns={"Identity": "MSA_Identity"}, inplace=True)
            avg_wdf.to_csv(out_dir / "windowed_msa_identity.csv", index=False)
            
            plt.figure(figsize=(12, 6))
            unique_orgs = org_wdf["Org"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
            for org, color in zip(unique_orgs, colors):
                o_data = org_wdf[org_wdf["Org"] == org]
                plt.plot(o_data["RefPos"], o_data["Identity"]*100, color=color, alpha=0.4, lw=1, label=org)
            plt.plot(avg_wdf["RefPos"], avg_wdf["MSA_Identity"]*100, label="Average Identity", color="red", lw=2, linestyle="--")
            plt.title(f"MSA Windowed Identity ({window_size}bp): {title}")
            plt.xlabel("Window Start Position (Reference bp)"); plt.ylabel("% Identity"); plt.ylim(0, 100)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small'); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / "msa_windowed_identity.png", dpi=300, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"Error in msa_window_analysis: {e}")
