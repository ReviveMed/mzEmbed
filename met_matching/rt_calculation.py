import pandas as pd
import numpy as np
# import sys
# import pubchempy as pcp
# import time
# import pickle
# import glob
import os
import json
# import networkx as nx
# import copy
# import re

import metabolite_name_matching_main as mnm

from google.cloud import storage

client = storage.Client()

adduct_path = "MCP_Adducts_RT/adducts_current.txt"
bucket = client.get_bucket('mzlearn-webapp.appspot.com')
all_targets_flat_path = "MCP_Adducts_RT/Multi-Project Targets and Metadata/all_targets_flat.csv"

# download adducts file to local
blob = bucket.blob(adduct_path)
print("downloading adduct to local")
blob.download_to_filename("adducts.txt")

# get existing all flat targets
blob = bucket.blob(all_targets_flat_path)
print("downloading all_targets_flat to local")
blob.download_to_filename("all_targets_flat.csv")

def get_mcp_hits(matched_df, max_ambig=np.Inf):
    """
    Link input queries to MCP entries.
    Helper function for processing targeted metabolite inputs.
    Args:
        matched_df (pd.DataFrame): DataFrame of matched metabolites-- output from met_match()
        max_ambig: maximum number of matches allowed for each input query (default Inf)
    Returns:
        all_mcp_hits (pd.DataFrame): DataFrame of all MCP hits for each input query
    """
    all_met_db_hits = pd.DataFrame()
    for targ in matched_df.loc[(matched_df.n_match>0) & (matched_df.n_match<max_ambig)].itertuples():
        mcp_hits = targ.MCP_match.split(",")
        met_db_hits = mnm.met_db.loc[mnm.met_db.MCP_met_ID.isin(mcp_hits)].copy()
        met_db_hits["query"] = targ.input
        met_db_hits["n_match"] = targ.n_match
        all_met_db_hits = pd.concat([all_met_db_hits, met_db_hits])

    return all_met_db_hits



def compile_target_list():
    """
    Script for processing, name matching, and compiling RT ranges for targeted metabolites.
    Run inside of the MCP_Adducts_RT/Multi-Project Targets and Metadata folder.
    Hardcoded to consider only datasets from human/mouse plasma/serum.
    Returns:
        all_targets_flat (pd.DataFrame): DataFrame of all targets with RT ranges
    Effects:
        Saves all_targets_flat.csv to MCP_Adducts_RT/Multi-Project Targets and Metadata folder
    """
    # tabulate summary stats for each dataset (ion, species, chrom)
    target_stats = {}
    all_targets = {}
    unmatched_targets = {}
    for root, dirs, files in os.walk(TARGET_DB_PATH):
        for dir in dirs:
            # if "rcc" in dir:  # filter for rcc data only
            folder_to_search = os.path.join(root, dir)
            stats_exists = os.path.exists(os.path.join(folder_to_search, "stats.json"))
            targets_exists = os.path.exists(os.path.join(folder_to_search, "targets.csv"))
            target_list, refmet_matched, mcp_all_std_matched, mcp_all_input_matched, mcp_std_matched, mcp_input_matched = [
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
            if stats_exists and targets_exists:
                with open(os.path.join(folder_to_search, "stats.json")) as f:
                    summary_stats = json.load(f)
                initial_targets = pd.read_csv(os.path.join(folder_to_search, "targets.csv"))
                target_list = initial_targets.copy()
                target_stats[dir] = {
                    "Ion": summary_stats["Ion"],
                    "Species": summary_stats["Species"],
                    "Sample Type": summary_stats["Sample Type"],
                    "Chromatography": summary_stats["Chromatography"],
                    "Initial Targets": None,
                    "RefMet/CID matched": None,
                    "Standardized met_match() matched": None,
                    "Input met_match() matched": None,
                    "Unique name matches": None,
                    "Ambig matches": None,
                    "Total name matches": None,
                    # "Match failure": None,
                    "Remaining unmatched": None,
                    "RT length": None
                }

                # filter for human and mouse data only, unlabeled targets only
                if (summary_stats["Species"] in ["human", "mouse"]) & (summary_stats["target_type"] == "unlabeled") & (
                        summary_stats["Sample Type"] in ["serum", "plasma"]):
                    if "Input name" in target_list.columns:
                        max_rt = all_max_rt.loc[dir, "rt_max"]
                        target_list.rtmed = target_list.rtmed / max_rt  # scale all RTs to max observed
                        # Match via RefMet and pubchem IDs
                        refmet_matched = mnm.refmet_db.merge(
                            target_list[["Input name", "Standardized name", "rtmed", "mzmed"]], left_on="refmet_name",
                            right_on="Standardized name").set_index("pubchem_cid")
                        refmet_matched = refmet_matched.merge(
                            mnm.met_db.loc[~mnm.met_db.pcp_pubchem_cid.isna()].set_index("pcp_pubchem_cid"),
                            left_index=True, right_index=True)[
                            ["Input name", "MCP_met_ID", "refmet_name", "metabolite__name", "HMDB_ID", "rtmed",
                             "mzmed"]]
                        refmet_matched["n_match"] = 1
                        target_list = target_list.loc[
                            ~target_list["Standardized name"].isin(refmet_matched["refmet_name"])]
                        # print("After matching by refmet:", len(target_list))

                        # Match via MCP using met_match() on standardized names
                        mcp_std_matched = pd.DataFrame()
                        if len(target_list) > 0:
                            std_targets = target_list.loc[target_list["Standardized name"] != "-"]
                            mcp_all_std_matched = mnm.met_match(df=std_targets, name_col="Standardized name",
                                                                mass_col="Exact mass", formula_col="Formula")
                            # mcp_std_matched = mcp_all_std_matched.loc[mcp_all_std_matched.n_match==1]
                            mcp_std_matched = get_mcp_hits(mcp_all_std_matched)
                            if len(mcp_std_matched) > 0:
                                mcp_std_matched = target_list.merge(mcp_std_matched, left_on="Standardized name",
                                                                    right_on="query")
                                # mcp_std_matched["metabolite__name"] = [i for i in mnm.met_db.set_index("MCP_met_ID").loc[mcp_std_matched.MCP_match].metabolite__name]
                                target_list = target_list.loc[
                                    ~target_list["Standardized name"].isin(mcp_std_matched["Standardized name"])]
                                # print("After matching by standardized name:", len(target_list))

                        # Match via MCP using met_match() on input names
                        mcp_input_matched = pd.DataFrame()
                        if len(target_list) > 0:
                            mcp_all_input_matched = mnm.met_match(df=target_list, name_col="Input name",
                                                                  mass_col="Exact mass", formula_col="Formula")
                            # mcp_input_matched = mcp_all_input_matched.loc[mcp_all_input_matched.n_match==1]
                            mcp_input_matched = get_mcp_hits(mcp_all_input_matched)
                            if len(mcp_input_matched) > 0:
                                mcp_input_matched = target_list.merge(mcp_input_matched, left_on="Input name",
                                                                      right_on="query")
                                # mcp_input_matched["metabolite__name"] = [i for i in mnm.met_db.set_index("MCP_met_ID").loc[mcp_input_matched.MCP_match].metabolite__name]
                                # remove matched targets from target_list
                                target_list = target_list.loc[
                                    ~target_list["Input name"].isin(mcp_input_matched["Input name"])]
                                # print("After matching by input name:", len(target_list))

                        # Combine match methods
                        all_mcp_matched = pd.concat([mcp_std_matched, mcp_input_matched])
                        all_mcp_matched.rename(columns={"MCP_match": "MCP_met_ID", "HMDB_match": "HMDB_ID"},
                                               inplace=True)
                        # df_to_merge = [df for df in [mcp_std_matched, mcp_input_matched, refmet_matched] if len(df)>0]
                        # if len(all_mcp_matched)>0 and len(refmet_matched)>0:
                        merged_match = pd.concat([all_mcp_matched[["MCP_met_ID", "Input name", "Standardized name",
                                                                   "metabolite__name", "HMDB_ID", "rtmed", "mzmed",
                                                                   "n_match"]].rename(
                            columns={"Standardized name": "refmet_name"}), refmet_matched])

                        all_targets[dir] = merged_match
                        unmatched_targets[dir] = target_list

                        if "n_match" in mcp_all_input_matched.columns:
                            match_fail = sum(mcp_all_input_matched.n_match == 0)
                        else:
                            match_fail = 0

                        target_stats[dir] = {
                            "Ion": summary_stats["Ion"],
                            "Species": summary_stats["Species"],
                            "Sample Type": summary_stats["Sample Type"],
                            "Chromatography": summary_stats["Chromatography"],
                            "Initial Targets": len(initial_targets),
                            "RefMet/CID matched": len(refmet_matched),
                            "Standardized met_match() matched": len(mcp_std_matched),
                            "Input met_match() matched": len(mcp_input_matched),
                            "Unique name matches": len(
                                merged_match.loc[merged_match["n_match"] == 1]["Input name"].unique()),
                            "Ambig matches": len(
                                merged_match.loc[(merged_match["n_match"] > 1)]["Input name"].unique()),
                            "Total name matches": len(merged_match["Input name"].unique()),
                            # "Match failure": match_fail,
                            # "Ambiguous_percent": round(sum(mcp_all_input_matched.n_match>10)/len(initial_targets)*100, 1),
                            "Remaining unmatched": len(target_list),
                            "RT length": summary_stats["Max RT (sec)"]
                        }


    dataset_list = pd.DataFrame.from_dict(target_stats, orient="index")
    dataset_list["match_percent"] = [round(i * 100, 1) for i in
                                     dataset_list["Total name matches"] / dataset_list["Initial Targets"]]

    for k, v in all_targets.items():
        v["Ion"] = dataset_list.loc[k, "Ion"]
        v["Species"] = dataset_list.loc[k, "Species"]
        v["Chromatography"] = dataset_list.loc[k, "Chromatography"]
        v["Sample Type"] = dataset_list.loc[k, "Sample Type"]
        v["dataset"] = k

    all_targets_flat = pd.concat(list(all_targets.values()))
    all_targets_flat.groupby(["Ion", "Species", "Chromatography", "Sample Type"]).agg(
        {"MCP_met_ID": lambda x: len(np.unique(x))})  # .to_clipboard()

    print("Unique matched metabolites:", len(set(all_targets_flat["Input name"])))
    print("Unique input metabolites:", len(set().union(
        *[set(v["Input name"]) for k, v in unmatched_targets.items() if k in dataset_list.index])) + len(
        set(all_targets_flat["Input name"])))
    print("Unambiguous matches:", len(all_targets_flat.loc[all_targets_flat.n_match == 1]["Input name"].unique()))


    # save all_targets_flat, then upload-- TO DO
    gcp_path = "MCP_Adducts_RT/Multi-Project Targets and Metadata/all_targets_flat.csv"
    atf_blob = bucket.blob(gcp_path)
    atf_blob.upload_from_filename("all_targets_flat.csv")
    print("Uploaded all_targets_flat.csv to", gcp_path)

    return dataset_list, all_targets_flat