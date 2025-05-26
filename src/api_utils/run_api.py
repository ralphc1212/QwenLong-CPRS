# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import time

from qwen_long_cprs.modeling_qwen2 import Qwen2ForCausalLMandBIO
import os
import logging
from transformers import AutoTokenizer, AutoConfig
import torch

from exceptions import *

from dataset import build_request_samples

from fastapi import FastAPI, BackgroundTasks
from fastapi import Request
import traceback
import os
# parser = argparse.ArgumentParser()
model_dir = os.environ.get("MODEL_DIR", None)


logger = logging.getLogger('qwenlong_compress')
logger.setLevel(logging.INFO)



def build_tag_from_data(dev_data, tokenizer):
    """
    returns:
        tag2vocab_id_list: list, 每个tag对应的vocab id的列表，用于找到其对应的vocab_id进行gather操作。
        tag2cls_id_map: dict, 每个tag对应的跟最终抽出来的cls一样的列表，用与最终找到tag distribution
    """
    tag_list = []
    tag_token_list = []
    for d in dev_data:
        for k in d['labels'].keys():
            tag_seq = tokenizer.decode(tokenizer.encode(k)[0])
            tag_token = tokenizer.tokenize(k)[0]
            if tag_seq not in tag_list:
                tag_list.append(tag_seq)
                tag_token_list.append(tag_token)
    tag_list.append('outside')
    tag_token_list.append('outside')
    tag2vocab_id_list = tokenizer.convert_tokens_to_ids(tag_token_list)
    tag2cls_id_map = {}
    cnt = 0
    for tag in tag_list:
        tag2cls_id_map[tag] = cnt
        cnt += 1
    return tag_list, tag2vocab_id_list, tag2cls_id_map



def correct_tag_pred(tag_pred):
    # 找B位置，找E位置，相隔小于xxx token，中间字符全部打为I
    new_tag_pred = tag_pred
    cur_offset = 0
    while cur_offset < len(new_tag_pred):
        try:
            B_idx = new_tag_pred[cur_offset:].index('B')
            B_idx = B_idx + cur_offset
        except:
            break
        
        if B_idx == len(new_tag_pred) - 1:
            break
        else:
            left_no_o_idx = B_idx
            right_no_o_idx = B_idx + 1
            while right_no_o_idx < len(new_tag_pred):
                if new_tag_pred[right_no_o_idx] == 'O':
                    right_no_o_idx += 1
                else:
                    if right_no_o_idx - left_no_o_idx < 20:
                        for j in range(left_no_o_idx+1, right_no_o_idx):
                            new_tag_pred[j] = 'I'
                        if new_tag_pred[right_no_o_idx] in ['B', 'E']:
                            break
                        else:
                            left_no_o_idx = right_no_o_idx
                            right_no_o_idx += 1
                    else:
                        break
        cur_offset = B_idx + 1

    cur_offset = len(new_tag_pred)
    while cur_offset > 0:
        try:
            E_idx = new_tag_pred[: cur_offset].rindex('E')
        except:
            break
        
        if E_idx == 0:
            break
        else:
            right_no_o_idx = E_idx
            left_no_o_idx = E_idx - 1
            while left_no_o_idx > 0:
                if new_tag_pred[left_no_o_idx] == 'O':
                    right_no_o_idx -= 1
                else:
                    if right_no_o_idx - left_no_o_idx < 20:
                        for j in range(left_no_o_idx+1, right_no_o_idx):
                            new_tag_pred[j] = 'I'
                        if new_tag_pred[left_no_o_idx] in ['B', 'E']:
                            break
                        else:
                            right_no_o_idx = left_no_o_idx
                            left_no_o_idx -= 1
                    else:
                        break
        cur_offset = E_idx
    
    return new_tag_pred


def get_pred_set_bi(tag_pred, context_offset, context, min_keyword_len, complete_sent=False, return_offset=False):
    pred_set = []
    start_idx = 0
    # try:
    #     assert len(tag_pred) == len(context_offset)
    # except:
    #     print('length mismatch')
    #     print(len(tag_pred))
    #     print(len(context_offset))
    pre_end = -1
    while start_idx < len(tag_pred):
        if tag_pred[start_idx] == "O":
            start_idx += 1
            continue
        else:
            end_idx = start_idx + 1
            # 不断往后加移位
            while end_idx < len(tag_pred):
                if tag_pred[end_idx] != 'O' and tag_pred[end_idx][0] != 'B':
                    end_idx += 1
                    continue
                else:
                    break
            char_start = context_offset[start_idx][0]
            char_end = context_offset[end_idx][0] if end_idx < len(context_offset) else len(context)
            # pred_set.append(tokenizer.decode(context_ids[start_idx:end_idx]).strip())
            if end_idx - start_idx > min_keyword_len:
                if complete_sent:
                    while char_start > -1:
                        if char_start == 0:
                            break
                        if context[char_start - 1] not in '.。：？！"”?!\n' and char_start > pre_end:
                            char_start -= 1
                        else:
                            break
                    while char_end < len(context):
                        if char_end == len(context) - 1:
                            break
                        if context[char_end - 1] not in '.。：？！"”?!\n':
                            char_end += 1
                        else:
                            break
                if not return_offset:
                    pred_set.append(context[char_start:char_end].strip())
                else:
                    pred_set.append((context[char_start:char_end].strip(), char_start, char_end))
            start_idx = end_idx
            pre_end = char_end
    return pred_set

def clean_preds(keywords, return_offset):
    new_keywords = []
    for kw in keywords:
        if return_offset:
            kw_ = kw[0]
        else:
            kw_ = kw
        if kw_.strip() == "":
            continue
        if kw_ in "（），。！":
            continue
        new_keywords.append(kw)
    return new_keywords


model_dir = '/nas-wulanchabu/shenweizhou.swz/qwen-long/context-compression/v1/BiLLM-main/saved_models/Qwen2-7B/20250320_infbench_v2/tech_report_outside_drop_0.5_len_8192_tagmode_normal_clean_merge_False_bi_label_type_BIOE_new_attn/Bilayer_21_bi_weight_10_outside_weight_0.5'

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})


config = AutoConfig.from_pretrained(model_dir)
model = Qwen2ForCausalLMandBIO.from_pretrained(
    model_dir,config=config
).half().eval().to(device)


def call_model(messages):
    prompt = messages[0]['content']
    query = messages[1]['content']
    context = messages[2]['content']

    query_str = ""
    query_str += '<|im_start|>system\n' + prompt + "<|im_end|>\n"
    query_str += '<|im_start|>user\n' + query + "<|im_end|>\n"
    query_str += '<|im_start|>context\n'


    input_ids = tokenizer.encode(query_str)
    input_ids_context = tokenizer.encode(context)
    input_ids = input_ids + input_ids_context
    input_ids = torch.LongTensor([input_ids]).to(device)
    #input_ids = torch.cat((input_ids, input_ids), dim=0).to(device)
    
    attention_mask = torch.ones_like(input_ids).to(device)

    outputs = model(input_ids,attention_mask).detach().cpu().numpy().tolist()

    return outputs


app = FastAPI()

@app.post("/qwen_long_compress_server")
async def qwen_long_compress_server(request: Request, progress_task: BackgroundTasks):
    try:
        data = await request.json()
        start_time = time.time()
        
        request_id = data['header']['request_id']
        # print("data:", data)
        messages = data['payload']['input']['messages']
        parameters = data['payload']['parameters']
        split_with_sent = parameters['split_with_sent'] if "split_with_sent" in parameters.keys() else True
        min_keyword_len = parameters['min_keyword_len'] if "min_keyword_len" in parameters.keys() else 4
        return_offset = parameters['return_offset'] if "return_offset" in parameters.keys() else False
        complete_sentence = parameters['complete_sentence'] if "complete_sentence" in parameters.keys() else False
        batch_size = parameters['batch_size'] if "batch_size" in parameters.keys() else 1
        # split_with_sent = True
        request_samples = build_request_samples(messages, tokenizer=tokenizer, max_len=8192, split_with_sent=split_with_sent)

        logger.info(f"prompt: {request_samples[0]['messages'][0]['content']}, query: {request_samples[0]['messages'][1]['content']}, split into {len(request_samples)} sub requests", extra={'request_id': request_id})

        results = []

        query_len = request_samples[0]['query_len']

        if batch_size == 1:
            for sample in request_samples:
                sub_message = sample['messages']
                score = call_model(messages=sub_message)
                score = score[0]
                if len(score) < (query_len + len(sample['offset_mapping'])):
                    score = score + [2] * (query_len + len(sample['offset_mapping']) - len(score))
                elif len(score) > (query_len + len(sample['offset_mapping'])):
                    raise BackendError(f"Got illegal score length: {len(score)}, expect {query_len + len(sample['offset_mapping'])}")     
                sub_res = {
                    'chunk_id': sample['chunk_id'],
                    'score': score[query_len:],
                    'offset_mapping': sample['offset_mapping']
                }
                results.append(sub_res)
        else:
            for step in range(0, len(request_samples), batch_size):
                sub_sample_group = request_samples[step: step+batch_size]
                sub_message_group = [sample['messages'] for sample in sub_sample_group]
                total_messages = []
                total_offset = []
                for msg in sub_message_group:
                    total_messages += msg
                for sample in sub_sample_group:
                    total_offset += sample['offset_mapping']
                score = call_model(messages=total_messages)

                assert len(score) == len(sub_sample_group)


                # print(score)

                total_scores = []
                for i,score_ in enumerate(score):
                    if len(score_) < (query_len + len(sub_sample_group[i]['offset_mapping'])):
                        score_ = score_ + [2] * (query_len + len(sub_sample_group[i]['offset_mapping']) - len(score_))
                    total_scores += score_[query_len:][:len(sub_sample_group[i]['offset_mapping'])]

                # print("total_scores",total_scores)
                sub_res = {
                    'chunk_id': sub_sample_group[0]['chunk_id'],
                    'score': total_scores,
                    'offset_mapping': total_offset
                }
                results.append(sub_res)

        results = sorted(results, key = lambda x: x['chunk_id'])
        scores = []
        bi_id_to_label = {0:'B', 1:'I', 2:'O', 3:'E'}
        context_offset = []
        context = messages[2]['content']
        for res in results:
            scores += res['score']
            context_offset += res['offset_mapping']
        tag_pred = [bi_id_to_label[s] for s in scores]
        tag_pred = correct_tag_pred(tag_pred)     
        pred_set = get_pred_set_bi(tag_pred, context_offset, context, min_keyword_len, complete_sent=complete_sentence, return_offset = return_offset)
        pred_set = clean_preds(pred_set, return_offset=return_offset)

        input_token_num = query_len * len(request_samples) + sum([len(sample['offset_mapping']) for sample in request_samples])
        
        data['payload']['output'] = {"text": pred_set}
        data['payload']['usage'] = {"input_tokens": input_token_num}
        
        data['header']['status_code'] = 200
        data['header']['status_name'] = "Success"
        data['header']['status_message'] = "Success"
        data['header']['finished'] = True
        end_time = time.time()
        # print("new_data:", data)
        elapsed_time = end_time - start_time  # 计算经过的时间，转换为毫秒
        print(f"Elapsed time: {elapsed_time} seconds")


        return data

    except Exception as e:

        if isinstance(e, InvalidInputError):
            data['header']['status_message'] = str(e)
            data['header']['status_name'] = 'InvalidInputError'
            data['header']['status_code'] = 400
        elif isinstance(e, BackendError):
            data['header']['status_message'] = e.error_body
            data['header']['status_name'] = "BackendError"
            data['header']['status_code'] = 500
        elif isinstance(e, SchemaError):
            data['header']['status_message'] = e.message
            data['header']['status_name'] = 'SchemaError'
            data['header']['status_code'] = 440
        else:
            data['header']['status_message'] = "internal error"
            data['header']['status_name'] = "InternalError"
            data['header']['status_code'] = 500
        traceback.print_exc()
        err_msg = traceback.format_exc()
        result = {'is_match': False}
        data['payload']['output'] = result
        data['payload']['usage'] = {"input_tokens": -1, "output_tokens": -1}
        print("new_data:", data)
        screenshot = f"./screenshot/screenshot_{request_id}.jpg"
        try:
            os.remove(screenshot)
        except FileNotFoundError:
            pass
        logger.error(f'error_message: {err_msg}', extra={'request_id': request_id})
        return data


@app.get('/readiness')
async def readiness():
    return 'success'

@app.get('/liveness')
async def liveness():
    return 'success'
