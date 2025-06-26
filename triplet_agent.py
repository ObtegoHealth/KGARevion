import argparse
import os
from transformers import set_seed
import json
from tqdm import tqdm
import logging
from src.utils import QADataset, MedDDxLoader, BaseLLM, AfrimedLoader
from action.generate import Generate
from action.review import Review
from action.answer import Answer

set_seed(42)


class KGARevion(object):
    def __init__(self,
                 args
                 ):
        super().__init__()
        self.agent_name = "KGARevion"
        self.role = """You can answer questions by choosing Extract_Triplets, KnowledgeGraph_Classifier and Answer_Generator actions. Finish it if you find answer."""
        self.args = args
        self.llm = BaseLLM("llama3.1")
        self.triplets_generator = Generate(self.llm, args)
        self.classifier = Review(self.llm, args)
        self.answer_generator = Answer(self.llm)

    
    def call(self, query):
        generated_triplets = self.triplets_generator.call(query)
        filtered_triplets, score = self.classifier.call(generated_triplets, query)
       
        return filtered_triplets

def main(args):

    ##load llm
    set_seed(42)

    import gc
    gc.collect()

    logging.basicConfig(filename = "triplet_agent.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    query = args.query

    bioKG_agent = KGARevion(args={
        "max_round": 1,
        "is_revise": True,
        "weights_path": "fine_tuned_model/"
    })
    
    response = bioKG_agent.call(query)
    return response

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    args = parser.parse_args()
    main(args)
