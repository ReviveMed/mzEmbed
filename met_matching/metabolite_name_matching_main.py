import pandas as pd
import sys
import numpy as np
from ast import literal_eval
import re
from fuzzywuzzy import fuzz
from requests import get
import os
from google.cloud import storage

dir_path = os.path.dirname(os.path.realpath(__file__))

if (not os.path.exists(dir_path+"/MCP_Metabolites")) or (not os.path.exists(dir_path+"/adduct.txt")):
    client = storage.Client()

    MCP_path = "MCP_Adducts_RT/MCP_Metabolites"
    adduct_path = "MCP_Adducts_RT/adducts_current.txt"
    bucket = client.get_bucket('mzlearn-webapp.appspot.com')

    if not os.path.exists(dir_path+"/MCP_Metabolites"):
        # download to local
        blob = bucket.blob(MCP_path)
        # download MCP_path to local
        print("downloading MCP_path to local")
        blob.download_to_filename(dir_path+"/MCP_Metabolites")

    if not os.path.exists(dir_path+"/adduct.txt"):
        # download to local
        blob = bucket.blob(adduct_path)
        # download MCP_path to local
        print("downloading adduct to local")
        blob.download_to_filename(dir_path+"/adduct.txt")

met_db = pd.read_csv(dir_path+"/MCP_Metabolites", sep="\t")  # hardcoded internal MCP met DB
refmet_db = pd.read_csv(dir_path+"/refmet.csv")  # hardcoded refmet db of standardized names with PubChem and InChI keys

# preprocess met_db
met_db_processed = met_db.fillna("").copy()
met_db_processed.metabolite__monisotopic_molecular_weight = [np.nan if i=="" else i for i in met_db_processed.metabolite__monisotopic_molecular_weight]
syn_parsed = []
for syns, met_name in np.array(met_db_processed[["synonyms__synonym", "metabolite__name"]]):
    syn_list = literal_eval(syns.lower())
    if met_name != "":
        syn_list += [met_name]
    syn_parsed += [[re.sub(" ", "", j) for j in syn_list]]
met_db_processed["parsed_synonyms"] = syn_parsed
met_db_processed.reset_index(inplace=True)

def refmet_query(name):
    """
    Function for querying RefMet REST API for metabolite information.
    Args:
        name (str): metabolite name to query
    Returns:
        dict of metabolite information, or blank if unavailable
    """

    query_url = f'https://www.metabolomicsworkbench.org/rest/refmet/match/{name}/name'
    query_return = get(query_url).text
    refmet_out = {"name": "", "pubchem_cid": "", "inchi_key": "", "exactmass": "", "formula": "", "super_class": "", "main_class": "", "sub_class": ""}
    if query_return != "[]":
        refmet_out = eval(query_return)

    return refmet_out



def count_carbon_bonds(chem_name):
    """
    Function for counting the total number of carbons and double bonds in a lipid. Internal helper for use in met_match()
    Args:
        chem_name: name of lipid (i.e. "PC(22:1(13Z)/16:0)")
    Returns:
        List with two elements; first element is total number of C atoms and second is total number of double bonds.
    """
    query_carbons = re.findall("(\d+:\d+)", chem_name)
    if len(query_carbons)>0:
        query_carbons = [i.split(":") for i in query_carbons]
        query_carbons = np.array([[int(i) for i in j] for j in query_carbons])
        return list(np.sum(query_carbons, 0))
    else:
        return [0, 0]

def met_match(chem_list=None, mode="both", met_db=met_db_processed, df=None, name_col=None, mass_col=None,
              formula_col=None, mz_tol=0.01):
    """
    Function for querying met_db by metabolite synonyms. Requires MCP_met list.
    Args:
        chem_list: list of query IDs (chemical names) or None
        mode: "lipid", "polar", or "both". Filters met_db before performing search.
        met_db: preprocessed MCP_met list
        df: df with query IDs and exact monoisotopic masses or None
        name_col: column name for query IDs in df
        mass_col: column name for exact monoisotopic masses in df
        formula_col: column name for chemical formula in df
        mz_tol: mass tolerance for matching
    Returns:
        data frame of inputs mapped to candidate HMDB and MCP IDs, along with the total number of matches and a description of the method used ("synonym", "prefix", or "synonym prefix").
    """

    assert (bool(chem_list is None) ^ bool(df is None)), "Either input_chems or df must be None"
    name_mass_dict = None
    name_formula_dict = None
    if df is None:
        input_chems = chem_list
    if chem_list is None:
        input_chems = list(df[name_col])
        if mass_col is not None:
            name_mass_dict = dict(zip(list(df[name_col]), list(df[mass_col])))
        if formula_col is not None:
            name_formula_dict = dict(zip(list(df[name_col]), list(df[formula_col])))


    # preprocess met_db
    met_db_for_query = met_db.copy()
    lipid_met_db = met_db_for_query.loc[
        (met_db_for_query["taxonomy__super_class"] == "Lipids and lipid-like molecules") | (
                met_db_for_query["taxonomy__super_class"] == "")]
    polar_met_db = met_db_for_query.loc[
        met_db_for_query["taxonomy__super_class"] != "Lipids and lipid-like molecules"]

    matches = {chem: {"match_type": "None", "mode": mode} for chem in input_chems}

    # check for exact name matches
    for input_chem in input_chems:
        matches[input_chem]["input"] = input_chem
        query_chem = input_chem
        if query_chem[-1] == "*":  # drop trailing "*"
            query_chem = query_chem[0:-1]
        query_chem = re.sub(' \(.*\)$', "", query_chem)  # strip alternate identifiers in trailing parentheses
        query_chem = re.sub(" ", "", query_chem.lower())
        matches[input_chem]["cleaned"] = query_chem
        name_hits = met_db_for_query.loc[
            met_db_for_query.metabolite__name.apply(lambda x: query_chem == x.lower())].copy()
        if name_mass_dict is not None:
            # print("name hits", name_hits[["index", "metabolite__name", "metabolite__monisotopic_molecular_weight"]])
            name_hits["mass_diff"] = abs(
                name_hits.metabolite__monisotopic_molecular_weight - name_mass_dict[input_chem])
            # filter for mass matches within mz_tol or nan
            name_hits = name_hits.loc[(name_hits.mass_diff < mz_tol) | (name_hits.mass_diff == np.nan)]
        if name_formula_dict is not None:
            if name_formula_dict[input_chem] != "":
                name_hits = name_hits.loc[name_hits.metabolite__chemical_formula == name_formula_dict[input_chem]]
        matches[input_chem]["n_match"] = len(name_hits)
        if len(name_hits) > 0:
            matches[input_chem]["HMDB_match"] = ",".join(list(name_hits.HMDB_ID))
            matches[input_chem]["MCP_match"] = ",".join(list(name_hits.MCP_met_ID))
            matches[input_chem]["match_type"] = "exact name"
        else:
            matches[input_chem]["HMDB_match"] = "None"
            matches[input_chem]["MCP_match"] = "None"

    # pick out remaining unmatched chemicals
    unmatched = [k for k, v in matches.items() if v["n_match"] == 0]

    # check for exact synonym matches
    for input_chem in unmatched:
        synonym_hits_formula_filtered = pd.DataFrame()
        matches[input_chem]["input"] = input_chem
        query_chem = input_chem
        if query_chem[-1] == "*":  # drop trailing "*"
            query_chem = query_chem[0:-1]
        # query_chem = re.sub(' \((\w+)\)$', "", query_chem)  # strip alternate identifiers in trailing parentheses
        query_chem = re.sub(' \(.*\)$', "", query_chem)  # strip alternate identifiers in trailing parentheses
        query_chem = re.sub(" ", "", query_chem.lower())
        matches[input_chem]["cleaned"] = query_chem
        match_ind = [query_chem in met for met in met_db_for_query.parsed_synonyms]
        if sum(match_ind) > 0:
            match_ind = list(np.concatenate(np.argwhere(match_ind)))
            synonym_hits = met_db_for_query.loc[match_ind].copy()
            if name_mass_dict is not None:
                # print("synonym hits", synonym_hits[["index", "metabolite__name", "metabolite__monisotopic_molecular_weight"]])
                synonym_hits["mass_diff"] = abs(
                    synonym_hits.metabolite__monisotopic_molecular_weight - name_mass_dict[input_chem])
                # filter for mass matches within mz_tol or nan
                synonym_hits = synonym_hits.loc[(synonym_hits.mass_diff < mz_tol) | (synonym_hits.mass_diff == np.nan)]
            if name_formula_dict is not None:
                if name_formula_dict[input_chem] != "":
                    synonym_hits_formula_filtered = synonym_hits.loc[synonym_hits.metabolite__chemical_formula == name_formula_dict[input_chem]].copy()
                    synonym_hits = synonym_hits_formula_filtered.copy()
            if len(synonym_hits) > 0:
                matches[input_chem]["HMDB_match"] = ",".join(list(synonym_hits.HMDB_ID))
                matches[input_chem]["MCP_match"] = ",".join(list(synonym_hits.MCP_met_ID))
                matches[input_chem]["match_type"] = "exact synonym"
                matches[input_chem]["n_match"] = len(synonym_hits)
            else:
                if len(synonym_hits_formula_filtered>0):
                    matches[input_chem]["HMDB_match"] = ",".join(list(synonym_hits_formula_filtered.HMDB_ID))
                    matches[input_chem]["MCP_match"] = ",".join(list(synonym_hits_formula_filtered.MCP_met_ID))
                    matches[input_chem]["match_type"] = "exact synonym"
                    matches[input_chem]["n_match"] = len(synonym_hits_formula_filtered)
                else:
                    matches[input_chem]["HMDB_match"] = "None"
                    matches[input_chem]["MCP_match"] = "None"
                    matches[input_chem]["n_match"] = 0


    # pick out remaining unmatched chemicals
    unmatched = [k for k, v in matches.items() if v["n_match"] == 0]

    # check for name matches by matching class prefix and carbon number-- lipids & both
    # define lipid class list
    lipid_class_list = list(set([j for j in [re.sub("\(.*", "", i) if ("(" in i) & (":" in i) else "" for i in
                                             met_db.loc[
                                                 met_db.taxonomy__super_class == "Lipids and lipid-like molecules"].metabolite__name]
                                 if ((len(j) < 7) & (len(j) > 0) & (" " not in j) & ("-" not in j)) if j[0].isalpha()]))
    lipid_class_list.sort()

    # dict for converting lipid classes from other formats; values are classes as recorded in met_db
    lipid_class_dict = {'DAG': 'DG',
                        'LPA': 'LysoPA',
                        'LPC': 'LysoPC',
                        'LPE': 'LysoPE',
                        'LPI': 'LysoPI',
                        'LSM': 'LysoSM',
                        'MAG': 'MG',
                        'TAG': 'TG'}

    if (mode == "lipid") or (mode == "both"):
        for q_raw in unmatched:
            lipid_formula_filtered = pd.DataFrame()
            # check if query has ":" in it and starts with a lipid class
            if ":" in q_raw:
                lipid_class = ""
                if any([q_raw.startswith(lip) for lip in lipid_class_list]):
                    lipid_class = lipid_class_list[
                        np.argwhere([q_raw.startswith(lip) for lip in lipid_class_list])[0][0]]
                if any([q_raw.startswith(lip) for lip in lipid_class_dict.keys()]):
                    lipid_class = lipid_class_dict[list(lipid_class_dict.keys())[
                        np.argwhere([q_raw.startswith(lip) for lip in lipid_class_dict.keys()])[0][0]]]
                if lipid_class != "":
                    query = q_raw.lower()
                    query_carbons = count_carbon_bonds(query)
                    met_db_prefix_sub = lipid_met_db.loc[[str(i).lower().startswith(lipid_class.lower()) for i in
                                                          lipid_met_db.metabolite__name]].copy()
                    if len(met_db_prefix_sub) > 0:
                        met_db_prefix_sub["carbons"] = [count_carbon_bonds(met) for met in
                                                        met_db_prefix_sub.metabolite__name]
                        lipid_db_hits = met_db_prefix_sub.loc[
                            [i == query_carbons for i in met_db_prefix_sub["carbons"]]].copy()
                        if len(lipid_db_hits) > 0:
                            if name_mass_dict is not None:
                                # print("lipid hits", lipid_db_hits[["index", "metabolite__name", "metabolite__monisotopic_molecular_weight"]])
                                lipid_db_hits["mass_diff"] = abs(
                                    lipid_db_hits.metabolite__monisotopic_molecular_weight - name_mass_dict[input_chem])
                                # filter for mass matches within mz_tol or nan
                                lipid_db_hits = lipid_db_hits.loc[
                                    (lipid_db_hits.mass_diff < mz_tol) | (lipid_db_hits.mass_diff == np.nan)]
                            if name_formula_dict is not None:
                                if name_formula_dict[q_raw] != "":
                                    lipid_formula_filtered = lipid_db_hits.loc[lipid_db_hits.metabolite__chemical_formula == name_formula_dict[q_raw]].copy()
                                    lipid_db_hits = lipid_formula_filtered.copy()
                            if len(lipid_db_hits)>0:
                                hmdb_hits = [i for i in set(lipid_db_hits.HMDB_ID) if i]
                                mcp_hits = [i for i in set(lipid_db_hits.MCP_met_ID) if i]
                                matches[q_raw] = {"input": q_raw, "mode": mode, "cleaned": lipid_class,
                                                "n_match": len(lipid_db_hits), "HMDB_match": ",".join(hmdb_hits),
                                                "MCP_match": ",".join(mcp_hits), "match_type": "name prefix"}
                            else:
                                if len(lipid_formula_filtered)>0:
                                    hmdb_hits = [i for i in set(lipid_formula_filtered.HMDB_ID) if i]
                                    mcp_hits = [i for i in set(lipid_formula_filtered.MCP_met_ID) if i]
                                    matches[q_raw] = {"input": q_raw, "mode": mode, "cleaned": lipid_class,
                                                    "n_match": len(lipid_formula_filtered), "HMDB_match": ",".join(hmdb_hits),
                                                    "MCP_match": ",".join(mcp_hits), "match_type": "name prefix"}

    # pick out remaining unmatched chemicals
    unmatched = [k for k, v in matches.items() if v["n_match"] == 0]

    # check for name matches by stripping functional groups-- polar only
    functional_groups = ['acetyl',
                         'adenosyl',
                         'allyl',
                         'aspartyl',
                         'arachidonoyl',
                         'butyryl',
                         'carbamoyl',
                         'carboxyethyl',
                         'carboxyl',
                         'chain, or cyclopropyl',
                         'deoxy',
                         'diacetyl',
                         'dihydroxy',
                         'dimethyl',
                         'docosahexaenoyl',
                         'ethyl',
                         'formyl',
                         'formimino',
                         'fructosyl',
                         'glutamyl',
                         'glutaryl',
                         'hexanoyl',
                         'hydroxybutyryl',
                         'hydroxy',
                         'hydroxyl',
                         'hydroxyoctanoyl',
                         'hydroxyphenyl',
                         'imidazole',
                         'indole',
                         'indoxyl',
                         'isobutyryl',
                         'isocaproyl',
                         'isovaleryl',
                         'linolenoyl',
                         'linoleoyl',
                         'methyl',
                         'monomethyl',
                         'nervonoyl',
                         'octanoyl',
                         'octadecanedioyl',
                         'oleoyl',
                         'oxindolyl',
                         'palmitoyl',
                         'phenyl',
                         'prolyl',
                         'propionyl',
                         'stearoyl',
                         'succinoyl',
                         'tigloyl',
                         'tiglyl',
                         'trimethyl']
    if (mode == "polar") or (mode == "both"):
        for q_raw in unmatched:
            polar_formula_filtered = pd.DataFrame()
            query = re.sub(" \(.*", "", q_raw.lower())
            query = re.sub(".*-", "", query)
            query = re.sub(" ", "", query)
            query = re.sub("^\)", "", query)
            if query.endswith("*"):
                query = query[0:-1]
            backbone = query
            fg = ""
            if any([query.startswith(fg) for fg in functional_groups]):
                fg = max([functional_groups[i] for i in
                          np.concatenate(np.argwhere([query.startswith(fg) for fg in functional_groups]))], key=len)
                backbone = re.sub("^" + fg, "", query)
                if len(backbone) <= 3:
                    backbone = query
                    fg = ""
            backbone = re.sub("^\)", "", backbone)
            if fg != "":
                polar_db_hits = polar_met_db.loc[[(backbone in str(i).lower()) & (fg in str(i).lower()) for i in
                                                  polar_met_db.metabolite__name]].copy()
                if len(polar_db_hits) > 0:
                    if name_mass_dict is not None:
                        # print("polar hits", polar_db_hits[["index", "metabolite__name", "metabolite__monisotopic_molecular_weight"]])
                        polar_db_hits["mass_diff"] = abs(
                            polar_db_hits.metabolite__monisotopic_molecular_weight.astype(float) - name_mass_dict[
                                input_chem])
                        # filter for mass matches within mz_tol or nan
                        polar_db_hits = polar_db_hits.loc[
                            (polar_db_hits.mass_diff < mz_tol) | (polar_db_hits.mass_diff == np.nan)]
                    if name_formula_dict is not None:
                        if name_formula_dict[q_raw] != "":
                            polar_formula_filtered = polar_db_hits.loc[polar_db_hits.metabolite__chemical_formula == name_formula_dict[q_raw]].copy()
                            polar_db_hits = polar_formula_filtered.copy()
                    if len(polar_db_hits)>0:
                        hmdb_hits = [i for i in set(polar_db_hits.HMDB_ID) if i]
                        mcp_hits = [i for i in set(polar_db_hits.MCP_met_ID) if i]
                        matches[q_raw] = {"input": q_raw, "mode": mode, "cleaned": backbone, "n_match": len(polar_db_hits),
                                        "HMDB_match": ",".join(hmdb_hits), "MCP_match": ",".join(mcp_hits),
                                        "match_type": "fg strip", "fg": fg}
                    else:
                        if len(polar_formula_filtered)>0:
                            hmdb_hits = [i for i in set(polar_formula_filtered.HMDB_ID) if i]
                            mcp_hits = [i for i in set(polar_formula_filtered.MCP_met_ID) if i]
                            matches[q_raw] = {"input": q_raw, "mode": mode, "cleaned": backbone, "n_match": len(polar_formula_filtered),
                                            "HMDB_match": ",".join(hmdb_hits), "MCP_match": ",".join(mcp_hits),
                                            "match_type": "fg strip", "fg": fg}
                else:
                    matches[q_raw]["cleaned"] = backbone
                    matches[q_raw]["fg"] = fg

    # pick out remaining unmatched chemicals
    unmatched = [k for k, v in matches.items() if v["n_match"] == 0]

    # check names for fuzzy string matching among synonyms
    met_db_fuzz = met_db.copy()
    met_db_fuzz.metabolite__name = [i.lower() for i in met_db_fuzz.metabolite__name]
    for fuzz_raw in unmatched:
        fuzz_formula_filtered = pd.DataFrame()
        fuzz_db_sub = met_db_fuzz.copy()
        if name_mass_dict is not None:
            fuzz_db_sub["mass_diff"] = abs(
                fuzz_db_sub.metabolite__monisotopic_molecular_weight.astype(float) - name_mass_dict[fuzz_raw])
            # filter for mass matches within mz_tol or nan
            fuzz_db_sub = fuzz_db_sub.loc[(fuzz_db_sub.mass_diff < mz_tol) | (fuzz_db_sub.mass_diff == np.nan)]
        if name_formula_dict is not None:
            if name_formula_dict[fuzz_raw] != "":
                fuzz_formula_filtered = fuzz_db_sub.loc[fuzz_db_sub.metabolite__chemical_formula == name_formula_dict[fuzz_raw]].copy()
                fuzz_db_sub = fuzz_formula_filtered.copy()
        fuzz_db_sub["Fuzz"] = [fuzz.ratio(fuzz_raw.lower(), i) for i in fuzz_db_sub.metabolite__name]
        fuzz_db_sub = fuzz_db_sub.loc[fuzz_db_sub.Fuzz > 90]
        if len(fuzz_db_sub) > 0:
            hmdb_hits = [i for i in set(fuzz_db_sub.HMDB_ID) if i]
            mcp_hits = [i for i in set(fuzz_db_sub.reset_index().MCP_met_ID) if i]
            # print("input", fuzz_raw, "n_match", len(fuzz_db_sub), "HMDB", hmdb_hits, "MCP", mcp_hits)
            matches[fuzz_raw] = {"input": fuzz_raw, "mode": mode, "cleaned": fuzz_raw, "n_match": len(fuzz_db_sub),
                                 "HMDB_match": ",".join(hmdb_hits), "MCP_match": ",".join(mcp_hits),
                                 "match_type": "fuzzy"}
        else:
            if len(fuzz_formula_filtered)>0:
                hmdb_hits = [i for i in set(fuzz_formula_filtered.HMDB_ID) if i]
                mcp_hits = [i for i in set(fuzz_formula_filtered.reset_index().MCP_met_ID) if i]
                matches[fuzz_raw] = {"input": fuzz_raw, "mode": mode, "cleaned": fuzz_raw, "n_match": len(fuzz_formula_filtered),
                                     "HMDB_match": ",".join(hmdb_hits), "MCP_match": ",".join(mcp_hits),
                                     "match_type": "fuzzy"}

    # if any matches have one HMDB ID but multiple MCP, use the entry corresponding to the HMDB
    for k, v in matches.items():
        if len(v["HMDB_match"].split(",")) == 1 and len(v["MCP_match"].split(",")) > 1:
            matches[k]["MCP_match"] = met_db.set_index("HMDB_ID").loc["HMDB0088423"].MCP_met_ID
            matches[k]["n_match"] = 1

    return pd.DataFrame.from_dict(matches, orient="index")