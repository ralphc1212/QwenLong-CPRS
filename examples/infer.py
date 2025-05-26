import time
import argparse
import json
import random
import os
from tqdm import tqdm
from openai import OpenAI
import traceback

def compress_api_call_local(messages: list):

    import requests
    retry_cnt = 0
    while retry_cnt < 2:
        try:
            data = {
                'header':{
                    'request_id': "1111abca"
                },
                'payload':{
                    'input':{
                        'messages':messages
                    },
                    'parameters':{
                        "min_keyword_len":1,
                        "complete_sentence":False,
                        "batch_size": 1,
                        'chunk_size': 8192
                    }
                }
            }

            url = 'http://0.0.0.0:8091/qwen_long_compress_server'
            payload = json.dumps(data)
            returns = requests.request("POST", url, data=payload)
            returns = returns.json()

            return returns['payload']['output']['text']

        except:
            retry_cnt += 1

            time.sleep(2)
            print("FAILED!",returns)
            continue
    # raise ValueError
    return []

def openai_call(messages: list, model: str,streaming=False):
    client = OpenAI(
        api_key=os.environ.get("LLM_APIKEY", None),
        base_url=os.environ.get("LLM_APIURL", None)
    )
    retry_cnt = 0
    while retry_cnt < 2:
        try:
            if streaming == False:
                response = client.chat.completions.create(
                    # model="deepseek-reasoner",
                    model=model,
                    messages=messages,
                    stream=False  # 明确关闭流式传输
                )
                print(response)
                output=response.choices[0].message.content.strip()
                return output
            else:
                response = client.chat.completions.create(
                    # model="deepseek-reasoner",
                    model=model,
                    messages=messages,
                )

                # 逐步接收并处理响应
                reasoning = ""
                output = ""
                for chunk in response:
                    chunk_message = chunk.choices[0].delta
                    # print(chunk.choices[0].delta)
                    if chunk_message.reasoning_content is not None:
                        reasoning += chunk_message.reasoning_content
                    
                    if chunk_message.content is not None:
                        output += chunk_message.content
                ans = output
                print(ans)
                return ans
        except:
            retry_cnt += 1
            traceback.print_exc()

            time.sleep(2)
            print("FAILED!",response)
            continue
    # raise ValueError
    return ""
    

def process_item(d, args):
    # from .prompts import prompt

    try:
        # Choose the appropriate prompt version
        if args.use_compress:
            messages_for_compress = [
                {
                    'role': 'system',
                    'content': args.cprs_prompt,
                },
                {
                    'role': 'user',
                    'content': d['query']
                },
                {
                    'role': 'context',
                    'content': d['context']
                }
            ]
            cprs_res = compress_api_call_local(messages_for_compress)
            doc_content = "Doc content:\n\n" + '\n'.join(cprs_res)
            d['cprs_preds'] = cprs_res
        else:
            doc_content = d['context']

        messages = [
            {'role': 'system', 'content': doc_content},
            {'role': 'user', 'content': d['query']},
        ]

        ans = openai_call(messages=messages, model=args.model, streaming=args.streaming)

        if ans != "":
            d['llm_preds'] = ans
            return  d
        else:
            return None

    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error processing id {d.get('id', 'N/A')}: {e}")
        return None



def main():
    
    random.seed(10)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='gpt-4-1106-preview')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=99999)
    parser.add_argument('--cprs_prompt', type=str, required=True)
    parser.add_argument('--use_compress', type=str, default='False')
    parser.add_argument('--streaming', type=str, default='False')
    args = parser.parse_args()

    def str2bool(text):
        if text.lower() in ['true', 'yes', '1']:
            return True
        else:
            return False
    
    args.use_compress = str2bool(args.use_compress)
    args.streaming = str2bool(args.streaming)

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load existing IDs to skip

    import json

    # Load existing IDs to skip
    existed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    existed_ids.add(json.loads(line).get('id'))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
   
    # Load and filter input data
    with open(args.input_path, 'r', encoding='utf-8') as f_in:
        all_lines = f_in.readlines()

    total_lines = len(all_lines)
    print(f"Total input lines: {total_lines}")


    filtered_data = []
    for idx, line in enumerate(all_lines):
        if idx < args.start or idx >= args.end:
            continue
        try:
            data = json.loads(line)
            if data['id'] not in existed_ids:
                filtered_data.append(data)
        except json.JSONDecodeError:
            continue  # Skip malformed lines
    

    print(f"Data to process after filtering: {len(filtered_data)}")


    # Use partial to fix the args parameter
    gen_cnt = 0
    with open(args.output_path, 'a', encoding='utf-8') as fw:
        # Use imap_unordered for better performance with large datasets
        for d in tqdm(filtered_data, total=len(filtered_data), desc="Processing"):
            result= process_item(d, args)
            print(f'=============={result["id"]}==============')
            if result is not None:
                print(result['query'])
                print('--------------------')
                print(result['llm_preds'])
                fw.write(json.dumps(result, ensure_ascii=False) + '\n')
                fw.flush()
            else:
                print('GOT UNEXPECTED SUM!!!!!!!!!!!')
            gen_cnt += 1


    fw.close()


if __name__ == "__main__":
    main()