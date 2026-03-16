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
from typing import Optional
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, add_labels_to_barplot, calculate_information_content

def align_sequences(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G"),
):
    """Perform MSA using Clustal Omega."""
    ref_safe = reference_organism.replace(" ", "_")
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="MSA"):
        gene = get_gene_name(f.stem)
        g_dir = output_base / "Organisms" / ref_safe / gene / "msa"
        for d in ["fasta", "plots", "slurm"]: (g_dir / d).mkdir(parents=True, exist_ok=True)
        recs = list(SeqIO.parse(f, "fasta"))
        if len(recs) < 2: continue
        ref_idx = next((i for i, r in enumerate(recs) if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), -1)
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
        
        temp_in = g_dir / "fasta" / "renamed_orthologs.fasta"
        SeqIO.write(renamed, temp_in, "fasta")
        aln, plot, ic_p, ic_c = g_dir / "fasta" / "aligned.fasta", g_dir / "plots" / "alignment.png", g_dir / "plots" / "information_content.png", g_dir / "information_content.csv"
        
        if slurm:
            script = g_dir / "slurm" / f"msa_{f.stem}.sh"
            ref_arg = f'--reference-id "{ref_id}"' if ref_id else ""
            content = f"#!/bin/bash\n#SBATCH --job-name=msa_{gene[:10]}\n#SBATCH --output={g_dir}/slurm/job.out\n#SBATCH --cpus-per-task=4\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\nmodule load clustal-omega\nclustalo -i {temp_in.resolve()} -o {aln.resolve()} --force --outfmt=fasta --threads=4\ndsrna-pipeline internal-msa-plot {aln.resolve()} {plot.resolve()} {ic_p.resolve()} {ic_c.resolve()} \"{gene}\" {ref_arg}\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            subprocess.run(["bash", "-c", f"module load clustal-omega && clustalo -i {temp_in} -o {aln} --force --outfmt=fasta"], check=True)
            internal_msa_plot(aln, plot, ic_p, ic_c, gene, ref_id)

def internal_msa_plot(aln_file: Path, plot_path: Path, ic_plot: Path, ic_csv: Path, title: str, reference_id: Optional[str] = typer.Option(None, "--reference-id")):
    try:
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
            plt.xticks(rotation=45, ha='right')
            plt.title(f"MSA Metrics: {title}")
            plt.ylabel("%")
            plt.ylim(0, 100)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            add_labels_to_barplot(ax)
            plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
            plt.savefig(plot_path.parent / "msa_metrics_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()
            
        ref_to_col = [i for i, b in enumerate(ref_seq) if b != '-']
        win_results = []
        for r in recs:
            if r.id == ref_rec.id: continue
            q_seq, org = str(r.seq), r.id.rsplit('_', 1)[0].replace('_', ' ')
            for s in range(len(ref_to_col) - 299):
                sc, ec = ref_to_col[s], ref_to_col[s + 299]
                win_ref, win_que = ref_seq[sc:ec+1], q_seq[sc:ec+1]
                ident = sum(1 for a, b in zip(win_ref, win_que) if a == b and a != '-') / (ec - sc + 1)
                win_results.append({"RefPos": s + 1, "Identity": ident, "Org": org, "SeqID": r.id})
                
        if win_results:
            wdf = pd.DataFrame(win_results)
            # Average across multiple sequences of the same organism at each position
            org_wdf = wdf.groupby(["Org", "RefPos"]).mean().reset_index()
            avg_wdf = wdf.select_dtypes(include=[np.number]).groupby(wdf["RefPos"]).mean().reset_index()
            
            # Save average windowed MSA identity to CSV
            avg_wdf.rename(columns={"Identity": "MSA_Identity"}, inplace=True)
            avg_wdf.to_csv(plot_path.parent.parent / "windowed_msa_identity.csv", index=False)
            
            plt.figure(figsize=(12, 6))
            unique_orgs = org_wdf["Org"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
            
            for org, color in zip(unique_orgs, colors):
                o_data = org_wdf[org_wdf["Org"] == org]
                plt.plot(o_data["RefPos"], o_data["Identity"]*100, color=color, alpha=0.4, lw=1, label=org)
            
            plt.plot(avg_wdf["RefPos"], avg_wdf["Identity"]*100, label="Average Identity", color="red", lw=2, linestyle="--")
            plt.title(f"MSA Windowed Identity: {title}")
            plt.xlabel("Window Start Position (Reference bp)")
            plt.ylabel("% Identity")
            plt.ylim(0, 100)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path.parent / "msa_windowed_identity.png", dpi=300, bbox_inches="tight")
            plt.close()
            
        ic_df = calculate_information_content(aln_file, reference_id)
        if not ic_df.empty:
            ic_df.to_csv(ic_csv, index=False)
            plt.figure(figsize=(15, 5))
            # Smooth with a rolling window (e.g., 30 bp)
            ic_df["IC_smoothed"] = ic_df["IC"].rolling(window=30, center=True, min_periods=1).mean()

            plt.fill_between(ic_df["Position"], ic_df["IC"], color="skyblue", alpha=0.2, label="Raw IC")
            plt.plot(ic_df["Position"], ic_df["IC"], color="skyblue", alpha=0.3, lw=0.5)
            plt.plot(ic_df["Position"], ic_df["IC_smoothed"], color="Slateblue", alpha=0.8, lw=2, label="Smoothed IC (30bp)")

            plt.ylim(0, 2.1)
            ref_name = reference_id.rsplit('_', 1)[0].replace('_', ' ') if reference_id else ""
            plt.xlabel(f"Alignment Position{' (Ref: ' + ref_name + ')' if ref_name else ''}")
            plt.ylabel("Information Content (bits)")
            plt.title(f"Information Content: {title}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            plt.savefig(ic_plot, dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
