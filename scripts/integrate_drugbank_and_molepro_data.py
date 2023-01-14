import xmltodict
import pandas as pd
import os, sys
import pickle
import argparse
import utils
import requests
from tqdm import tqdm

def generate_query_graph(curie_id, category):
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


def extract_drug_target_pairs_from_kg(kg, pmid_support=True):

    if pmid_support:
        res = [(kg['edges'][key]['subject'], kg['edges'][key]['object'], attr['value']) for key in kg['edges'] for attr in kg['edges'][key]['attributes'] if attr['original_attribute_name']=='publication']
        return pd.DataFrame(res, columns=['subject','object','pmid'])
    else:
        res = [(kg['edges'][key]['subject'], kg['edges'][key]['object']) for key in kg['edges']]
        return pd.DataFrame(res, columns=['subject','object'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step3_3.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--molepro_api_link', type=str, help='API link of Molecular Data Provider', default='https://translator.broadinstitute.org/molepro/trapi/v1.2')
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
        api_res = requests.get(f'{args.molepro_api_link}/meta_knowledge_graph')
        if api_res.status_code == 200:
            molepro_meta_kg = api_res.json()

        molepro_df = pd.DataFrame(columns=['subject','object','pmid'])
        for index in tqdm(range(len(res))):
            curie_id = res.loc[index,'id']
            category = res.loc[index,'category']
            request_body = generate_query_graph(curie_id, category)
            r = requests.post(f'{args.molepro_api_link}/query', json = request_body, headers={'accept': 'application/json'})
            if r.status_code == 200:
                r_res = r.json()
                temp_pairs = extract_drug_target_pairs_from_kg(r_res['message']['knowledge_graph'])
                molepro_df = pd.concat([molepro_df,temp_pairs]).reset_index(drop=True)
            else:
                logger.warning(f"{curie_id} fails to call molepro api")

            if index!=0 and index%1000==0:
                molepro_df.to_csv(os.path.join(args.data_dir, args.outdir_name, 'molepro_df_backup.txt'), sep='\t', index=None)
        molepro_df.to_csv(os.path.join(args.data_dir, args.outdir_name, 'molepro_df_backup.txt'), sep='\t', index=None)
    else:
        molepro_df = pd.read_csv(os.path.join(args.data_dir, args.outdir_name, 'molepro_df_backup.txt'), sep='\t', header=0)

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
