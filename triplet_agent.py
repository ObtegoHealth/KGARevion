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
        self.llm = BaseLLM(args.llm_name)
        self.triplets_generator = Generate(self.llm, args)
        if args.llm_name == 'gpt-4-turbo':
            self.review_llm = BaseLLM('llama3.1')
        else:
            self.review_llm = self.llm
        self.classifier = Review(self.review_llm, args)
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

    bioKG_agent = KGARevion(args=args)
    
    response = bioKG_agent.call(args.query)
    return response

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_round", type=int, default=1)
    parser.add_argument("--is_revise", type=bool, default=True)
    parser.add_argument("--llm_name", default='llama3.1', choices=['llama3.1', 'llama3', 'gpt-4-turbo', 'llama3.1-70'], type=str)
    parser.add_argument("--weights_path", type=str, default='fine_tuned_model/')
    parser.add_argument("--query", type=str)
    args = parser.parse_args()
    main(args)

