import json
import pandas as pd
import numpy as np
import typer
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from Bio import SeqIO
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name, parse_gene_ids

app = typer.Typer()

@app.command()
def aggregate(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input-file"),
):
    """Aggregate windowed metrics for all specified window sizes and generate summary plots."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    gene_configs = parse_gene_ids(input_file)
    gene_to_windows = {g['description'].replace(' ', '_'): g['windows'] for g in gene_configs}

    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Aggregating Metrics"):
        gene = get_gene_name(f.stem)
        gene_safe = gene.replace(' ', '_')
        gene_dir = org_base / gene
        
        # Base alignment files
        aln_base = gene_dir / "alignments"
        ic_csv = aln_base / "pairwise" / "anchored_information_content.csv" 
        
        windows = gene_to_windows.get(gene_safe, [300])
        
        # Check ref length if fasta exists
        ref_len = None
        if f.exists():
            recs = list(SeqIO.parse(f, "fasta"))
            ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
            if ref_rec:
                ref_len = len(str(ref_rec.seq))

        for ws in windows:
            if ref_len and ref_len < ws:
                logger.warning(f"Gene '{gene}' (ref len {ref_len}) is shorter than window {ws}. Skipping aggregation.")
                continue

            ws_dir = gene_dir / str(ws)
            pw_csv = ws_dir / "similarity" / "pairwise" / "windowed_identity.csv"
            pw_org_csv = ws_dir / "similarity" / "pairwise" / "organism_windowed_identity.csv"
            msa_csv = ws_dir / "similarity" / "msa" / "windowed_msa_identity.csv"
            acc_csv = ws_dir / "accessibility" / "data" / "windowed_analysis.csv"
            acc_org_csv = ws_dir / "accessibility" / "data" / "organism_windowed_accessibility.csv"
            bt_win_csv = ws_dir / "bowtie_matches" / "windowed_bowtie_summary.csv"

            if not (pw_csv.exists() and acc_csv.exists() and bt_win_csv.exists()):
                logger.warning(f"Missing core metric files for {gene} window {ws}. Skipping.")
                continue
                
            df_pw = pd.read_csv(pw_csv).rename(columns={"Identity": "Avg_Pairwise_Identity"})
            df_acc = pd.read_csv(acc_csv)
            df_bt = pd.read_csv(bt_win_csv)
            
            # Merge
            final_df = pd.merge(df_pw, df_acc[["RefPos", "NTO_Acc", "Ref_Acc"]], on="RefPos", how="inner")
            final_df = pd.merge(final_df, df_bt, on="RefPos", how="left")
            
            if msa_csv.exists():
                df_msa = pd.read_csv(msa_csv).rename(columns={"MSA_Identity": "Avg_MSA_Identity"})
                final_df = pd.merge(final_df, df_msa[["RefPos", "Avg_MSA_Identity"]], on="RefPos", how="left")
            
            if ic_csv.exists():
                df_ic = pd.read_csv(ic_csv)
                # Calculate windowed mean IC
                ic_vals = df_ic["IC"].values
                win_ic = []
                for start_ref in range(len(ic_vals) - (ws - 1)):
                    win_ic.append(np.mean(ic_vals[start_ref : start_ref + ws]))
                df_win_ic = pd.DataFrame({"RefPos": range(1, len(win_ic) + 1), "Pairwise_IC": win_ic})
                final_df = pd.merge(final_df, df_win_ic, on="RefPos", how="left")

            final_df.fillna(0, inplace=True)
            final_df.rename(columns={"RefPos": "Window_Start_Pos", "NTO_Acc": "Avg_NTO_Accessibility", "Ref_Acc": "Ref_Accessibility", "Hits_0mm": "Bowtie_Hits_0mm", "Hits_1mm": "Bowtie_Hits_1mm", "Hits_2mm": "Bowtie_Hits_2mm"}, inplace=True)
            
            agg_dir = ws_dir / "summary"
            agg_dir.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(agg_dir / "window_metrics_summary.csv", index=False)
            
            # 5-Panel Plot
            fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
            x = final_df["Window_Start_Pos"]
            
            axes[0].plot(x, final_df["Ref_Accessibility"], color="black", lw=2)
            axes[0].set_title(f"Ref Accessibility ({ws}bp): {gene}"); axes[0].set_ylabel("Prob Unpaired"); axes[0].set_ylim(0, 1)
            
            if acc_org_csv.exists():
                df_acc_org = pd.read_csv(acc_org_csv)
                for org, o_df in df_acc_org.groupby("Organism"):
                    axes[1].plot(o_df["RefPos"], o_df["NTO_Acc"], color="gray", alpha=0.3, lw=1)
            axes[1].plot(x, final_df["Avg_NTO_Accessibility"], color="red", lw=2, label="Avg NTO")
            axes[1].set_title(f"NTO Accessibility ({ws}bp): {gene}"); axes[1].set_ylim(0, 1); axes[1].legend()
            
            if pw_org_csv.exists():
                df_pw_org = pd.read_csv(pw_org_csv)
                for org, o_df in df_pw_org.groupby("Organism"):
                    axes[2].plot(o_df["RefPos"], o_df["Identity"]*100, color="gray", alpha=0.3, lw=1)
            axes[2].plot(x, final_df["Avg_Pairwise_Identity"]*100, color="blue", lw=2, label="Avg Identity")
            axes[2].set_title(f"Pairwise Identity ({ws}bp): {gene}"); axes[2].set_ylabel("% Identity"); axes[2].set_ylim(0, 100); axes[2].legend()

            if "Pairwise_IC" in final_df.columns:
                ic_smooth = final_df["Pairwise_IC"].rolling(window=30, center=True, min_periods=1).mean()
                axes[3].fill_between(x, final_df["Pairwise_IC"], color="skyblue", alpha=0.2)
                axes[3].plot(x, ic_smooth, color="Slateblue", lw=2)
                axes[3].set_title(f"Pairwise IC (Mean in Window): {gene}"); axes[3].set_ylim(0, 2.1)
            
            axes[4].stackplot(x, final_df["Bowtie_Hits_0mm"], final_df["Bowtie_Hits_1mm"], final_df["Bowtie_Hits_2mm"], labels=["0 MM", "1 MM", "2 MM"], colors=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
            axes[4].set_title(f"Bowtie Hits ({ws}bp): {gene}"); axes[4].set_xlabel("Ref Position"); axes[4].legend()
            
            plt.tight_layout(); plt.savefig(agg_dir / "pipeline_summary_metrics.png", dpi=300); plt.close()
            
            worst_case = final_df.sort_values(by=["Avg_Pairwise_Identity", "Avg_NTO_Accessibility", "Bowtie_Hits_0mm", "Bowtie_Hits_1mm"], ascending=[False, False, False, False]).head(10)
            worst_case.to_csv(agg_dir / "top_10_worst_case_windows.csv", index=False)
            
    logger.info("Aggregation complete.")

if __name__ == "__main__":
    app()
