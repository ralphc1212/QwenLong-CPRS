# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json

# parser = argparse.ArgumentParser()

    
def nested_load_data(path):
    data = []
    if os.path.isfile(path):
        with open(path) as f:
            for line in f.readlines():
                d = json.loads(line)
                pred = None
                if 'llm_preds' in d.keys():
                    pred = d['llm_preds']
                else:
                    for key in d.keys():
                        if key.startswith('response_'):
                            pred = d[key]
                            break
                assert pred is not None
                data.append((pred, d['outputs'] if "outputs" in d.keys() else d["ouputs"]))
        return data
    else:
        files = os.listdir(path)
        for file in files:
            data += nested_load_data(os.path.join(path, file))
        return data




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str)
    parser.add_argument('--task_type', type=str)

    args = parser.parse_args()

    def string_match_part(preds, refs):
        score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
        return round(score, 2)

    def string_match_all(preds, refs):
        score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
        return round(score, 2)
        

    TASKS = {
        'niah': {
            'metric_fn': string_match_all,
        },
        'variable_tracking': {
            'metric_fn': string_match_all,
        },
        'cwe': {
            'metric_fn': string_match_all,
        },
        'fwe': {
            'metric_fn': string_match_all
        },
        'qa': {
            'metric_fn': string_match_part,
        },
    }


    eval_data = nested_load_data(args.input_path)

    # print(eval_data[:100])
    preds = [d[0] for d in eval_data]
    refs = [d[1] for d in eval_data]

    print(args.task_type)
    score = TASKS[args.task_type]['metric_fn'](preds, refs)
    print(score)





    