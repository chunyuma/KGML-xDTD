
"""
This script defines a class `QueryMyChem` to communicate with MyChem APIs.
"""

## Import Libraries
import os
import sys
import json
import httpx
import asyncio
import eval_utilities

from node_synonymizer import NodeSynonymizer

class QueryMyChem:
    
    ## Constructor
    def __init__(self, API_BASE_URL = 'http://mychem.info/v1', TIMEOUT_SEC = 120):
        
        ## user-defined variables
        self.API_BASE_URL = API_BASE_URL
        self.TIMEOUT_SEC = TIMEOUT_SEC
        
        ## internal variables
        self.available_id_prefix = ['CHEMBL.COMPOUND', 'CHEBI', 'UMLS', 'PUBCHEM.COMPOUND'] 
        self.synonymizer = eval_utilities.nodesynonymizer

    async def __call_async_api(self, client: httpx.AsyncClient, url: str):
        
        try:
            res = await client.get(url)
        except httpx.TimeoutException:
            print(f"Timeout in QueryMyChem for URL: {url}", flush=True)
            return None
        except BaseException as e:
            print(f"{e} received in QueryMyChem for URL: {url}", flush=True)
            return None
        
        if res.status_code != 200:
            print(f"Error {res.status_code} received in QueryMyChem for URL: {url}", flush=True)
            return None
        
        return res.json()
    
    async def __call_async_apis(self, urls):
        
        async with httpx.AsyncClient(timeout=self.TIMEOUT_SEC) as client:
            tasks = [asyncio.create_task(self.__call_async_api(client, url)) for url in urls if url]

            return await asyncio.gather(*tasks)
    
    def __get_url(self, synonym, endpoint='chem', spec=''):
        
        if synonym.split(':')[0] in ['CHEMBL.COMPOUND', 'PUBCHEM.COMPOUND']:
            endpoint = 'chem'
            query = synonym.split(':')[1]
            return f"{self.API_BASE_URL}/{endpoint}/{query}{spec}"
        elif synonym.split(':')[0] == 'CHEBI':
            endpoint = 'chem'
            query = synonym
            return f"{self.API_BASE_URL}/{endpoint}/{query}{spec}"
        elif synonym.split(':')[0] == 'UMLS':
            query = synonym.split(':')[1]
            return f"{self.API_BASE_URL}/query?q=drugcentral.drug_use.indication.umls_cui:{query}"
        else:
            return None

    def __get_all_synonyms(self, chembl_id):

        results = self.synonymizer.get_equivalent_nodes(chembl_id)[chembl_id]
        if results is None:
            return None
        else:
            synonyms = list(results.keys())
            avaiable_synonyms = [x for x in synonyms if x.split(':')[0] in self.available_id_prefix]
            if len(avaiable_synonyms) == 0:
                return None
            else:
                return avaiable_synonyms

    def search_results(self, key, results_dict):
        
        if isinstance(results_dict, dict):
            for k, v in results_dict.items():
                if k == key:
                    yield v
                elif isinstance(v, dict):
                    for result in self.search_results(key, v):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in self.search_results(key, d):
                            yield result

    def get_drug_or_chemical_class(self, chem_id):
        
        ## get all synonyms
        synonyms = self.__get_all_synonyms(chem_id)
        
        if synonyms is None:
            print(f"No available synonyms found for {chem_id}", flush=True)
            return str(chem_id)
        else:
            ## get urls for all synonyms
            urls = [self.__get_url(synonym) for synonym in synonyms]
            
            ## get results for all urls
            contents = asyncio.run(self.__call_async_apis(urls))
            drug_class = set()
            for content in contents:
                if content is not None:
                    for x in self.search_results("fda_epc", content):
                        if x:
                            if isinstance(x, list):
                                for y in x:
                                    drug_class.add((y['code'],y['description']))
                            else:
                                drug_class.add((x['code'],x['description']))
            drug_class = list(drug_class)
            if len(drug_class) == 0:
                return str(chem_id)
            else:
                return drug_class
            
if __name__ == "__main__":
    
    ## test
    query_mychem = QueryMyChem()
    print(query_mychem.get_drug_or_chemical_class('CHEMBL.COMPOUND:CHEMBL1308'))