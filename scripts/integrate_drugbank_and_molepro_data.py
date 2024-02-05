import xmltodict
import pandas as pd
import os, sys
import pickle
import argparse
import utils
import requests
import asyncio
import httpx
from multiprocessing import Pool
from typing import Optional, List
import traceback
from tqdm import tqdm, trange

# class MoleProData:

#     def __init__(self, logger, molepro_api_link: str = 'https://molepro-trapi.transltr.io/molepro/trapi/v1.3'):
#         """
#         Initial Method
#         """
#         ## setup basic information
#         self.molepro_api_link = molepro_api_link
#         self.logger = logger
#         self.client = httpx.Client()

#     async def _call_async_melepro(self, curie_id: str, category: str):
#             try:
#                 request_body = self._generate_query_graph(curie_id, category)
#                 resp = await self.client.post(f'{self.molepro_api_link}/query', json = request_body, headers={'accept': 'application/json'})
#             except Exception:
#                 traceback.print_exc()
#                 print(f"############### {curie_id}", flush=True)
#                 return pd.DataFrame([], columns=['subject','object','pmid'])

#             if not resp.status_code == 200:
#                 self.logger.warning(f"{curie_id} fails to call molepro api with status code {resp.status_code}")
#                 return pd.DataFrame([], columns=['subject','object','pmid'])

#             resp_res = resp.json()
#             temp_pairs = self._extract_drug_target_pairs_from_kg(resp_res['message']['knowledge_graph'])
    
#             return temp_pairs

#     async def _get_data(self, param_list: List):
#         async with httpx.AsyncClient(timeout=None) as client:
#             tasks = [asyncio.create_task(self._call_async_melepro(client, curie_id, category)) for curie_id, category in param_list]
#             self.logger.info("starting to extract data from molepro api")
#             temp_results = await asyncio.gather(*tasks)
#             self.results = pd.concat(temp_results).reset_index(drop=True)
#             self.logger.info("Extracted data from molepro api done")

#     @staticmethod
#     def _generate_query_graph(curie_id, category):
#         if type(curie_id) is str:
#             query_id = [curie_id]
#         else:
#             query_id = curie_id

#         query_graph = {
#             "message": {
#                 "query_graph": {
#                 "edges": {
#                     "e00": {
#                     "subject": "n00",
#                     "predicates": [
#                         "biolink:affects",
#                         "biolink:interacts_with"
#                     ],
#                     "object": "n01"
#                     }
#                 },
#                 "nodes": {
#                     "n00": {
#                     "ids": query_id,
#                     "categories": [
#                         category
#                     ]
#                     },
#                     "n01": {
#                     "categories": [
#                         "biolink:Gene",
#                         "biolink:Protein"
#                     ]
#                     }
#                 }
#                 }
#             }
#         }

#         return query_graph

#     @staticmethod
#     def _extract_drug_target_pairs_from_kg(kg, pmid_support=True):

#         if pmid_support:
#             res = [(kg['edges'][key]['subject'], kg['edges'][key]['object'], attr['value']) for key in kg['edges'] for attr in kg['edges'][key]['attributes'] if attr['original_attribute_name']=='publication']
#             return pd.DataFrame(res, columns=['subject','object','pmid'])
#         else:
#             res = [(kg['edges'][key]['subject'], kg['edges'][key]['object']) for key in kg['edges']]
#             return pd.DataFrame(res, columns=['subject','object'])


#     def get_molepro_data(self, res):

#         param_list = [(row[0], row[1]) for row in res.to_numpy()]
#         ## start the asyncio program
#         asyncio.run(self._get_data(param_list))


def get_melepro_data(params: tuple):

    def _generate_query_graph(curie_id, category):
        if type(curie_id) is str:
            query_id = [curie_id]
        else:
            query_id = curie_id

        query_graph = {
            "message": {
                "query_graph": {
                "edges": {
                    "e00": {
                    "subject": "n00",
                    "predicates": [
                        "biolink:affects",
                        "biolink:interacts_with"
                    ],
                    "object": "n01"
                    }
                },
                "nodes": {
                    "n00": {
                    "ids": query_id,
                    "categories": [
                        category
                    ]
                    },
                    "n01": {
                    "categories": [
                        "biolink:Gene",
                        "biolink:Protein"
                    ]
                    }
                }
                }
            }
        }

        return query_graph

    def _extract_drug_target_pairs_from_kg(kg, pmid_support=True):

        if pmid_support:
            res = [(kg['edges'][key]['subject'], kg['edges'][key]['object'], attr['value']) for key in kg['edges'] for attr in kg['edges'][key]['attributes'] if attr['original_attribute_name']=='publication']
            return pd.DataFrame(res, columns=['subject','object','pmid'])
        else:
            res = [(kg['edges'][key]['subject'], kg['edges'][key]['object']) for key in kg['edges']]
            return pd.DataFrame(res, columns=['subject','object'])


    curie_id, category, molepro_api_link = params
    try:
        request_body = _generate_query_graph(curie_id, category)
        resp = requests.post(f'{molepro_api_link}/query', json = request_body, headers={'accept': 'application/json'})
    except Exception:
        traceback.print_exc()
        print(f"ERROR: ############### {curie_id}", flush=True)
        return pd.DataFrame([], columns=['subject','object','pmid'])

    if not resp.status_code == 200:
        print(f"WARNING: {curie_id} fails to call molepro api with status code {resp.status_code}")
        return pd.DataFrame([], columns=['subject','object','pmid'])

    resp_res = resp.json()
    temp_pairs = _extract_drug_target_pairs_from_kg(resp_res['message']['knowledge_graph'])

    return temp_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step3_3.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument("--batchsize", type=int, help="Batch Size", default=50000)
    parser.add_argument("--process", type=int, help="Use number of processes to run the program", default=50)
    parser.add_argument('--molepro_api_link', type=str, help='API link of Molecular Data Provider', default='https://molepro-trapi.transltr.io/molepro/trapi/v1.3')
    parser.add_argument('--outdir_name', type=str, help='The name of output directory for storing results', default='expert_path_files')
    args = parser.parse_args()


    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)


    if not os.path.exists(os.path.join(args.data_dir, args.outdir_name, 'all_drugs.txt')):
        neo4j_bolt = os.getenv('neo4j_bolt')
        neo4j_username = os.getenv('neo4j_username')
        neo4j_password = os.getenv('neo4j_password')

        ## extract all possible drug entities from neo4j database
        conn = utils.Neo4jConnection(uri=neo4j_bolt, user=neo4j_username, pwd=neo4j_password)
        res = conn.query(f"match (n) where n.category='biolink:SmallMolecule' or n.category='biolink:Drug' return distinct n.id, n.category, n.equivalent_curies")    
        res.columns = ['id','category','equivalent_curies']
        res.to_csv(os.path.join(args.data_dir, args.outdir_name, 'all_drugs.txt'), sep='\t', index=None)
    else:
        res = pd.read_csv(os.path.join(args.data_dir, args.outdir_name, 'all_drugs.txt'), sep='\t', header=0)
        res = res.apply(lambda row: [row[0], row[1], eval(row[2])], axis=1, result_type='expand')
        res.columns = ['id','category','equivalent_curies']

    ## read drugbank processed data
    file_path = os.path.join(args.data_dir, args.outdir_name, 'p_expert_paths.txt')
    p_expert_paths = pd.read_csv(file_path, sep='\t', header=None)
    p_expert_paths = p_expert_paths.loc[~p_expert_paths[2].isna(),:]
    p_expert_paths.columns = ['drugbankid', 'subject', 'object']

    ## query molepro (Molecular Data Provider) API
    if not os.path.exists(os.path.join(args.data_dir, args.outdir_name, 'molepro_df_backup.txt')):
        # api_res = requests.get(f'{args.molepro_api_link}/meta_knowledge_graph')
        # if api_res.status_code == 200:
        #     molepro_meta_kg = api_res.json()

        # set up the batches
        pair_list = [(row[0], row[1], args.molepro_api_link) for row in res.to_numpy()]
        batch =list(range(0,len(pair_list), args.batchsize))
        batch.append(len(pair_list))
        logger.info(f'total batch: {len(batch)-1}')

        molepro_df = pd.DataFrame(columns=['subject','object','pmid'])
        ## run each batch in parallel
        for i in trange(len(batch)):
            if((i+1)<len(batch)):
                start = batch[i]
                end = batch[i+1]
                if args.process == -1:
                    with Pool() as executor:
                        out_res = executor.map(get_melepro_data, pair_list[start:end])
                else:
                    with Pool(processes=args.process) as executor:
                        out_res = executor.map(get_melepro_data, pair_list[start:end])
                temp_molepro_df = pd.concat(out_res).reset_index(drop=True)
                molepro_df = pd.concat([molepro_df,temp_molepro_df]).reset_index(drop=True)
                # save intermediate results
                molepro_df.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', index=None)

        molepro_df.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', index=None)
    else:
        molepro_df = pd.read_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', header=0)
    
    temp_dict = dict()
    for index in range(len(molepro_df)):
        source, target, pmid = molepro_df.loc[index,'subject'], molepro_df.loc[index,'object'], molepro_df.loc[index,'pmid']
        if (source, target) not in temp_dict:
            temp_dict[(source, target)] = [pmid]
        else:
            temp_dict[(source, target)].append(pmid)
    molepro_df = pd.DataFrame([(key[0], key[1], value) for key, value in temp_dict.items()])
    molepro_df.columns = ['subject','object','pmid']

    combined_table = molepro_df.merge(p_expert_paths,how='outer',on=['subject','object'])
    combined_table.loc[(~combined_table.loc[:,'pmid'].isna()) & (~combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'drugbank&molepro'
    combined_table.loc[(combined_table.loc[:,'pmid'].isna()) & (~combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'drugbank'
    combined_table.loc[(~combined_table.loc[:,'pmid'].isna()) & (combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'molepro'

    ## output the results
    combined_table.to_csv(os.path.join(args.data_dir, args.outdir_name, 'p_expert_paths_combined.txt'), sep='\t', index=None)
