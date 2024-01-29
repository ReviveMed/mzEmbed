# Merging the results from two or more alignments methods


import networkx as nx
import numpy as np
import pandas as pd
import json
import os
from .utils_eclipse import align_ms_studies_with_Eclipse
from .utils_metabCombiner import align_ms_studies_with_metabCombiner


def align_ms_studies_with_merge(origin_study,input_study,
                                origin_name='origin',input_name='input',
                                save_dir=None,params_list=None,
                                alignment_savename='Merge',verbose=True):
    
    if params_list is None:
        params_list = [{'alignment_method':'metabcombiner','alignment_params':None,
                                  'weight':2, 'method_param_name': 'default'},
                                {'alignment_method':'Eclipse','alignment_params':None,
                                 'weight':1, 'method_param_name': 'default'}]

    assert isinstance(params_list,list)
    
    # check that each element of the list has the required keys
    for params in params_list:
        assert 'alignment_method' in params.keys()
        assert 'alignment_params' in params.keys()
        assert 'weight' in params.keys()
        assert 'method_param_name' in params.keys()

    # get the alignment results for each method
    align_df_list = []
    align_name_list = []
    align_weights = []
    
    if save_dir is not None:
        intermediate_save_dir = os.path.join(save_dir,'merge_intermediate')
        os.makedirs(intermediate_save_dir,exist_ok=True)
    else:
        intermediate_save_dir = None
    
    if save_dir is not None:
        with open(os.path.join(save_dir,f'{input_name}_aligned_to_{origin_name}_Merge_params.json'), 'w') as fp:
            json.dump(alignment_params, fp)

    # check for duplicates
    for iter,params in enumerate(params_list):
        alignment_savename = params['alignment_method'] + '_' + params['method_param_name']
        align_name_list.append(alignment_savename)

    if len(set(align_name_list)) < len(align_name_list):
        raise ValueError('duplicate alignment names')

    # Get the alignment results for each method
    for iter,params in enumerate(params_list):
        if verbose: 
            print(f'Merge method{iter}: running {params["alignment_method"]} alignment with {params["method_param_name"]}')
       
        alignment_savename = params['alignment_method'] + '_' + params['method_param_name']
        alignment_params = params['alignment_params']
        weight = params['weight']
        if weight <= 0:
            continue
        if 'metabcombiner' in params['alignment_method'].lower():
            try:
                align_df = align_ms_studies_with_metabCombiner(origin_study=origin_study, 
                                                                input_study=input_study,
                                                                origin_name=origin_name,
                                                                input_name=input_name,
                                                                alignment_params=alignment_params,
                                                                save_dir=intermediate_save_dir,
                                                                alignment_savename=alignment_savename)
            except ValueError:
                print('Within Merge Alignment, metabCombiner alignment failed')
                align_df = None
                weight = 0
        
        elif 'eclipse' in params['alignment_method'].lower():
            try:
                align_df = align_ms_studies_with_Eclipse(origin_study=origin_study, 
                                                            input_study=input_study,
                                                            origin_name=origin_name,
                                                            input_name=input_name,
                                                            alignment_params=alignment_params,
                                                            save_dir=intermediate_save_dir,
                                                            alignment_savename=alignment_savename)
            except ValueError:
                print('Within Merge Alignment, Eclipse alignment failed')
                align_df = None
                weight = 0
        else:
            raise NotImplementedError('alignment method not implemented')
        
        if weight > 0:
            align_df_list.append(align_df)
            align_weights.append(weight)


    assert len(align_weights) > 0, 'no alignment methods were successful'
    assert len(align_df_list) == len(align_weights)

    # Compute the union alignment
    union_df = compute_union_merge_alignment(align_df_list,align_weights)

    if save_dir is not None:
        union_df.to_csv(filepath=os.path.join(save_dir,f'{input_name}_aligned_to_{origin_name}_with_{alignment_savename}.csv'))


    return union_df




def compute_union_merge_alignment(align_df_list,align_weights):

    if align_weights is None:
        align_weights = np.ones(len(align_df_list))

    assert len(align_df_list) == len(align_weights)
    
    # check that the alignment dfs are all dataframes
    for df in align_df_list:
        assert isinstance(df,pd.DataFrame)

    # check that the first two columns for all of the dataframes are the same
    name0 = align_df_list[0].columns[0]
    name1 = align_df_list[0].columns[1]
    assert name0 != name1, 'the first two columns of the alignment dataframes must have different names'
    for df in align_df_list:
        if name0 != df.columns[0]:
            if name0 == df.columns[1]:
                df = df.rename(columns={df.columns[1]:df.columns[0],df.columns[0]:df.columns[1]})

        assert name0 == df.columns[0]
        assert name1 == df.columns[1]
    
    # create unique feature ids
    for df in align_df_list:
        df[name0] = df[name0].apply(lambda x: name0 + '_' + str(x))
        df[name1] = df[name1].apply(lambda x: name1 + '_' + str(x))
        
    # Create a new bipartite graph to represent the alignment results
    G = nx.Graph()

    # add the nodes
    for iter,df in enumerate(align_df_list):
        G.add_nodes_from(df[name0].values, bipartite=0)
        G.add_nodes_from(df[name1].values, bipartite=1)
        G.add_edges_from(zip(df[name0], df[name1]), weight=align_weights[iter])

    #### get the maximum weight matching
    # Create a new graph to store the result
    H = nx.Graph()

    num_components = nx.number_connected_components(G)
    # Find maximum cardinality matching for each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        num_size = len(subgraph)
        matching = nx.max_weight_matching(subgraph)
        # if num_size > 2:
            # print(num_size)
            # break
        H.add_edges_from(tuple(edge) for edge in matching)


    # create a new dataframe to store the result
    union_df = pd.DataFrame(columns=[name0, name1])

    # add the edges from the matching to the dataframe
    union_df[name0] = [edge[0] if edge[0].startswith(name0) else edge[1] for edge in H.edges()]
    union_df[name1] = [edge[0] if edge[0].startswith(name1) else edge[1] for edge in H.edges()]

    # remove the study names from the feature ids
    # This only works if there are no '_' in the feature ids
    # union_df[name0] = union_df[name0].apply(lambda x: int(x.split('_')[-1]))
    # union_df[name1] = union_df[name1].apply(lambda x: int(x.split('_')[-1]))

    union_df[name0] = union_df[name0].apply(lambda x: x.replace(name0+'_',''))
    union_df[name1] = union_df[name1].apply(lambda x: x.replace(name1+'_',''))

    return union_df