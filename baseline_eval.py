import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import json

"""
this file run QG inference with bleu/rouge/meteor metrics

"""

class QGinference():
    def __init__(self, dataset = "squad", tokenizer = "google-t5/t5-large", model = "iarfmoose/t5-base-question-generator"):
        # evaluate metrics
        self.eval_rouge = evaluate.load('rouge')
        self.eval_bleu = evaluate.load('bleu')
        self.eval_meteor = evaluate.load('meteor')
        self.eval = {'rougeL':self.eval_rouge, 'bleu':self.eval_bleu, 'meteor':self.eval_meteor}
        
        # model init
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to('cuda')

        # load dataset
        print("start loading dataset")
        self.dataset = self.load_dataset(dataset)
        print("done dataset loading")
        
    
    def load_dataset(self, dataset):
        qa_list = []
        if dataset == "squad":
            squad = pd.read_json('/home/tzujohsu/SI630/dev-v2.0.json')
            for i in range(len(squad['data'])):
                for j in range(len(squad['data'][i]['paragraphs'])):
                    context = squad['data'][i]['paragraphs'][0]['context']
                    for k in range(len(squad['data'][i]['paragraphs'][j]['qas'])):
                        if squad['data'][i]['paragraphs'][j]['qas'][k]['answers']:
                            answer = squad['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']
                            question = squad['data'][i]['paragraphs'][j]['qas'][k]['question']
                            qa_list.append((str('<answer> '+ str(answer)+ ' <context> '+ str(context)), question))
            print("squad dev data instances: ", len(qa_list))
            print("example: ", qa_list[0])
                
        return qa_list
    
    def evaluate(self, predictions:list, references:list):
        results = [self.eval[metric].compute(predictions=predictions, references=references)[metric] for metric in self.eval]
        return results[0], results[1], results[2]


    def run_inference(self):
        data = self.dataset
        res = {'rougeL': [], 'bleu': [], 'meteor': []}
        print("start running inference")
        
        for instance in tqdm(data, total = len(data)):
            encoding = self.tokenizer.encode_plus(instance[0], return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=50,
                temperature = 1.5,
                do_sample=True,
                top_k=90,
                top_p=0.5,
                num_return_sequences=1
            )
            line = self.tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
            rougeL, bleu, meteor = self.evaluate([line], [instance[1]])
            res['rougeL'].append(rougeL)
            res['bleu'].append(bleu)
            res['meteor'].append(meteor)
        n = len(res['rougeL'])
        
        # return {'rougeL_avg':res['rougeL']/n, 'bleu_avg':res['bleu']/n, 'meteor_avg':res['meteor']}, res
        try:
            with open('result.json', 'w') as fp:
                json.dump(res, fp)
        except:
            pass
        return  {'rougeL_avg':sum(res['rougeL'])/n, 'bleu_avg':sum(res['bleu'])/n, 'meteor_avg':sum(res['meteor'])/n}
        

        
if __name__ == "__main__":
    exp1 = QGinference(tokenizer='facebook/bart-large', model='facebook/bart-large')
    print(exp1.run_inference())