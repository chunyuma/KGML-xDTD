import sys, os, re
import pandas as pd
import argparse
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step1.log")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="~/data")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    neo4j_bolt = os.getenv('neo4j_bolt')
    neo4j_username = os.getenv('neo4j_username')
    neo4j_password = os.getenv('neo4j_password')

    output_path = args.output_folder

    ## Connect to neo4j database
    conn = utils.Neo4jConnection(uri=neo4j_bolt, user=neo4j_username, pwd=neo4j_password)

    ## Pull a dataframe of all graph edges
    query = "match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:BehavioralFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:SmallMolecule') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and not (m1.id in drug_ids and m2.id in disease_ids) and not (m1.id in disease_ids and m2.id in drug_ids) with distinct m1 as node1, r as edge, m2 as node2 return node2.id as source, node1.id as target, edge.predicate as predicate, edge.knowledge_source as knowledge_source"
    KG_alledges = conn.query(query)
    KG_alledges.columns = ['source','target','predicate', 'p_knowledge_source']
    logger.info(f"Total number of triples in kg after removing all edges between drug entities and disease entities: {len(KG_alledges)}")
    KG_alledges.to_csv(os.path.join(output_path, 'graph_edges.txt'), sep='\t', index=None)

    ## Pulls a dataframe of all graph nodes with category label
    query = "match (n) with distinct n.id as id, n.category as category, n.name as name, n.all_names as all_names, n.description as des return id, category, name, all_names, des"
    KG_allnodes_label = conn.query(query)
    KG_allnodes_label.columns = ['id','category','name', 'all_names', 'des']
    for index in range(len(KG_allnodes_label)):
        KG_allnodes_label.loc[index,'all_names'] = list(set([x.lower() for x in KG_allnodes_label.loc[index,'all_names']])) if KG_allnodes_label.loc[index,'all_names'] else KG_allnodes_label.loc[index,'all_names']
    logger.info(f"Total number of entities: {len(KG_allnodes_label)}")
    for i in range(len(KG_allnodes_label)):
        if KG_allnodes_label.loc[i, "des"]:
            KG_allnodes_label.loc[i, "des"] = " ".join(KG_allnodes_label.loc[i, "des"].replace("\n", " ").split())

    KG_allnodes_label = KG_allnodes_label.apply(lambda row: [row[0], row[1], utils.clean_up_name(row[2]), list(set([utils.clean_up_name(name) for name in row[3]])) if row[3] is not None else [''], utils.clean_up_desc(row[4])], axis=1, result_type='expand')
    KG_allnodes_label.columns = ['id','category','name', 'all_names', 'des']

    KG_allnodes_label.to_csv(os.path.join(output_path, 'all_graph_nodes_info.txt'), sep='\t', index=None)


