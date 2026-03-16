import pandas as pd
import numpy as np
import typer
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name

app = typer.Typer()

@app.command()
def aggregate(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
):
    """Aggregate windowed metrics and generate a summary 5-panel figure with individual sequences."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Aggregating Metrics"):
        gene = get_gene_name(f.stem)
        gene_dir = org_base / gene
        
        pw_csv = gene_dir / "pairwise_alignments" / "windowed_identity.csv"
        pw_org_csv = gene_dir / "pairwise_alignments" / "organism_windowed_identity.csv"
        acc_csv = gene_dir / "accessibility" / "data" / "windowed_analysis.csv"
        acc_org_csv = gene_dir / "accessibility" / "data" / "organism_windowed_accessibility.csv"
        ic_csv = gene_dir / "pairwise_alignments" / "anchored_information_content.csv"
        bt_raw_csv = gene_dir / "bowtie_matches" / "results" / "all_matches.csv"
        bt_win_csv = gene_dir / "bowtie_matches" / "results" / "windowed_bowtie_summary.csv"
        
        if not (pw_csv.exists() and acc_csv.exists()):
            logger.warning(f"Missing core metric files for {gene}. Skipping.")
            continue
            
        df_pw = pd.read_csv(pw_csv)
        df_acc = pd.read_csv(acc_csv)
        
        # Merge basic metrics
        acc_cols = ["RefPos", "NTO_Acc"]
        if "Ref_Acc" in df_acc.columns: acc_cols.append("Ref_Acc")
        final_df = pd.merge(df_pw, df_acc[acc_cols], on="RefPos", how="inner")
        if "Ref_Acc" not in final_df.columns: final_df["Ref_Acc"] = np.nan
        
        # Add IC if exists
        if ic_csv.exists():
            df_ic = pd.read_csv(ic_csv)
            final_df = pd.merge(final_df, df_ic.rename(columns={"RefPos": "RefPos", "IC": "Pairwise_IC"}), on="RefPos", how="left")
        else: final_df["Pairwise_IC"] = np.nan
        
        # Bowtie hits logic
        if bt_win_csv.exists():
            df_bt = pd.read_csv(bt_win_csv)
            final_df = pd.merge(final_df, df_bt, on="RefPos", how="left")
        elif bt_raw_csv.exists():
            df_raw = pd.read_csv(bt_raw_csv)
            df_dedup = df_raw.drop_duplicates(subset=["kmer_id", "offset", "mismatches_count"])
            max_pos = final_df["RefPos"].max()
            all_pos = np.arange(1, max_pos + 1)
            
            v0 = df_dedup[df_dedup["mismatches_count"] == 0].groupby("RefPos").size().reindex(all_pos, fill_value=0)
            v1 = df_dedup[df_dedup["mismatches_count"] == 1].groupby("RefPos").size().reindex(all_pos, fill_value=0)
            v2 = df_dedup[df_dedup["mismatches_count"] == 2].groupby("RefPos").size().reindex(all_pos, fill_value=0)
            
            # Forward-looking rolling sum (300bp)
            win_v0 = v0[::-1].rolling(window=300, min_periods=1).sum()[::-1]
            win_v1 = v1[::-1].rolling(window=300, min_periods=1).sum()[::-1]
            win_v2 = v2[::-1].rolling(window=300, min_periods=1).sum()[::-1]
            
            df_bt = pd.DataFrame({
                "RefPos": all_pos,
                "Hits_0mm": win_v0.values,
                "Hits_1mm": win_v1.values,
                "Hits_2mm": win_v2.values
            })
            final_df = pd.merge(final_df, df_bt, on="RefPos", how="left")
        else:
            for c in ["Hits_0mm", "Hits_1mm", "Hits_2mm"]: final_df[c] = 0

        final_df.fillna(0, inplace=True)
        final_df.rename(columns={"RefPos": "Window_Start_Pos", "Identity": "Avg_Pairwise_Identity", "NTO_Acc": "Avg_NTO_Accessibility", "Ref_Acc": "Ref_Accessibility", "Hits_0mm": "Bowtie_Hits_0mm", "Hits_1mm": "Bowtie_Hits_1mm", "Hits_2mm": "Bowtie_Hits_2mm"}, inplace=True)
        
        agg_dir = gene_dir / "summary"
        agg_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(agg_dir / "window_metrics_summary.csv", index=False)
        
        # 5-Panel Plot
        fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
        x = final_df["Window_Start_Pos"]
        
        # 1. Ref Acc
        axes[0].plot(x, final_df["Ref_Accessibility"], color="black", lw=2)
        axes[0].set_title(f"Reference Sequence Accessibility (300bp Window): {gene}"); axes[0].set_ylabel("Prob Unpaired"); axes[0].set_ylim(0, 1)
        
        # 2. NTO Acc (Indiv + Avg)
        if acc_org_csv.exists():
            df_acc_org = pd.read_csv(acc_org_csv)
            for org, o_df in df_acc_org.groupby("Organism"):
                axes[1].plot(o_df["RefPos"], o_df["NTO_Acc"], color="gray", alpha=0.4, lw=1.2)
        axes[1].plot(x, final_df["Avg_NTO_Accessibility"], color="red", lw=3, label="Average NTO")
        axes[1].set_title(f"NTO Accessibility (300bp Window): {gene}"); axes[1].set_ylabel("Prob Unpaired"); axes[1].set_ylim(0, 1); axes[1].legend(loc='upper right')
        
        # 3. Pairwise Identity (Indiv + Avg)
        if pw_org_csv.exists():
            df_pw_org = pd.read_csv(pw_org_csv)
            for org, o_df in df_pw_org.groupby("Organism"):
                axes[2].plot(o_df["RefPos"], o_df["Identity"]*100, color="gray", alpha=0.4, lw=1.2)
        axes[2].plot(x, final_df["Avg_Pairwise_Identity"]*100, color="blue", lw=3, label="Average Identity")
        axes[2].set_title(f"Pairwise Identity to NTOs (300bp Window): {gene}"); axes[2].set_ylabel("% Identity"); axes[2].set_ylim(0, 100); axes[2].legend(loc='upper right')

        # 4. Pairwise-Anchored IC
        if "Pairwise_IC" in final_df.columns:
            ic_smooth = final_df["Pairwise_IC"].rolling(window=30, center=True, min_periods=1).mean()
            axes[3].fill_between(x, final_df["Pairwise_IC"], color="skyblue", alpha=0.2)
            axes[3].plot(x, ic_smooth, color="Slateblue", lw=2)
            axes[3].set_title(f"Pairwise-Anchored Information Content: {gene}"); axes[3].set_ylabel("IC (bits)"); axes[3].set_ylim(0, 2.1)
        
        # 5. Bowtie Matches
        axes[4].stackplot(x, final_df["Bowtie_Hits_0mm"], final_df["Bowtie_Hits_1mm"], final_df["Bowtie_Hits_2mm"], labels=["0 MM", "1 MM", "2 MM"], colors=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
        axes[4].set_title(f"Total 21-mer Bowtie Hits across NTOs (300bp Window): {gene}"); axes[4].set_ylabel("Match Count"); axes[4].set_xlabel("Window Start Position (Reference bp)"); axes[4].legend(loc='upper right')
        
        for ax in axes: ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(agg_dir / "pipeline_summary_metrics.png", dpi=300); plt.close()
        
        # Top 10 - Worst Case
        worst_case_windows = final_df.sort_values(by=["Avg_Pairwise_Identity", "Avg_NTO_Accessibility", "Bowtie_Hits_0mm", "Bowtie_Hits_1mm"], ascending=[False, False, False, False]).head(10)
        worst_case_windows.to_csv(agg_dir / "top_10_worst_case_windows.csv", index=False)
        
    logger.info("Metrics aggregation and plotting complete.")

if __name__ == "__main__":
    app()
