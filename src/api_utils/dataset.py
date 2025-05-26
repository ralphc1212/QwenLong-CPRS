# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
from exceptions import *
import re

def detect_chinese(text):
    # 去掉文本中的空白字符
    text = text.strip()
    
    # 中文字符的正则表达式范围
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')

    # 计算中文字符的数量
    chinese_chars = chinese_char_pattern.findall(text)
    # 计算英文字母的数量

    num_chinese = len(chinese_chars)
    # 通过数量判断是中文还是英文
    if num_chinese > 0:
        return True
    else:
        return False


def build_request_sample_with_sent(messages, tokenizer, max_len=8192):
    prompt = messages[0]['content']
    query = messages[1]['content']
    context = messages[2]['content']

    query_str = ""
    query_str += '<|im_start|>system\n' + prompt + "<|im_end|>\n"
    query_str += '<|im_start|>user\n' + query + "<|im_end|>\n"
    query_str += '<|im_start|>context\n'

    query_len = len(tokenizer.tokenize(query_str))

    cur_window_len = query_len # 当前窗口长度计数
    cur_seq_len = 0 #当前已加入的序列长度
    left_context = context # 剩余未加入的序列
    prev_offset = 0 #上一个 sample的偏移量

    cur_context = ""

    
    ret = []
    cnt = 0

    while cur_seq_len < len(context):
        # 找到当前要加入的位置
        # 几种情况
        # - 。 ！ ？
        # - 。”， ！”，？”
        # - 上面两种情况加\n
        next_punkt = None
        next_punkt_idx = len(left_context) - 1

        for punkt in [".", "\n", "。”","！”", "？”","”", "。","！","？", "|"]:
            if punkt not in left_context:
                next_punkt_idx_temp = len(left_context) - 1
            else:
                next_punkt_idx_temp = left_context.index(punkt)

            if next_punkt_idx_temp < next_punkt_idx:
                next_punkt = punkt
                next_punkt_idx = next_punkt_idx_temp
        
        # 继续往后包进\n
        if next_punkt is not None:
            end_of_sent = next_punkt_idx + len(next_punkt)
            while end_of_sent < len(left_context) and (left_context[end_of_sent] == '\n' or left_context[end_of_sent] == " "):
                end_of_sent += 1
        else:
            end_of_sent = len(left_context)

        
        # 判断长度，加进去
        sent_encoding = tokenizer.encode_plus(left_context[:end_of_sent], return_offsets_mapping=True)
        sent_passage_ids= sent_encoding['input_ids']
        if cur_window_len + len(sent_passage_ids) <= max_len:
            #没超长
            cur_context += left_context[:end_of_sent]
        else:
            encoding = tokenizer.encode_plus(cur_context, return_offsets_mapping=True)

            passage_ids = encoding["input_ids"]
            passage_offsets = encoding["offset_mapping"]
            passage_offsets = [(x[0]+prev_offset, x[1]+prev_offset) for x in passage_offsets]
            # print(cur_context)
            # print("add sent len:",len(passage_offsets))
            if len(passage_offsets) + query_len > max_len:
                # print('++++++++++++++++length over size+++++++++++++')
                # print(len(passage_offsets) + query_len)
                passage_ids = passage_ids[:max_len - query_len]
                cur_context = tokenizer.decode(passage_ids)

            
            new_msg = [
                {
                    'role':'system',
                    'content': prompt
                },
                {
                    'role': 'user',
                    'content': query
                },
                {
                    'role': 'context',
                    'content': cur_context
                }
            ]
            new_sample = {
                'chunk_id': cnt,
                'messages': new_msg,
                "offset_mapping": passage_offsets,
                'query_len': query_len
            }
            ret.append(new_sample)
            cnt += 1

            #更新相关变量
            prev_offset += len(cur_context)
            cur_context = left_context[:end_of_sent]
            cur_window_len = query_len
        
        cur_window_len += len(sent_passage_ids)
        cur_seq_len += len(left_context[:end_of_sent])
        if cur_seq_len == len(context):
            break
        else:
            left_context = left_context[end_of_sent:]

    if cur_context != "":
        encoding = tokenizer.encode_plus(cur_context, return_offsets_mapping=True)
        # passage_ids = encoding["input_ids"]
        passage_offsets = encoding["offset_mapping"]
        passage_offsets = [(x[0]+prev_offset, x[1]+prev_offset) for x in passage_offsets]
        # print("add sent len:",len(passage_ids))
        new_msg = [
            {
                'role':'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': query
            },
            {
                'role': 'context',
                'content': cur_context
            }
        ]
        new_sample = {
            'chunk_id': cnt,
            'messages': new_msg,
            "offset_mapping": passage_offsets,
            'query_len': query_len
        }
        ret.append(new_sample)
        cnt += 1

    return ret




def build_request_samples(messages, tokenizer, max_len=8192, split_with_sent=False, max_document_len=2048000):
    if len(messages) != 3:
        raise SchemaError('Messages should len=3 list, the first role should be "system", second role should be "user", third role should be "context"')
    if messages[0]['role'] != 'system' and messages[1]['role'] != 'user' and messages[2]['role'] != 'context':
        raise SchemaError('Messages should len=3 list, the first role should be "system", second role should be "user", third role should be "context"')
    
    if len(tokenizer.encode(messages[-1]['content'])) > max_document_len:
        messages[-1]['content'] = tokenizer.decode(tokenizer.encode(messages[-1]['content'])[:max_document_len])[:-1]
        # print(messages[-1]['content'], flush=True)

    if split_with_sent and detect_chinese(messages[-1]['content']):
        return build_request_sample_with_sent(messages, tokenizer, max_len)
    
    prompt = messages[0]['content']
    query = messages[1]['content']
    context = messages[2]['content']

    query_str = ""
    query_str += '<|im_start|>system\n' + prompt + "<|im_end|>\n"
    query_str += '<|im_start|>user\n' + query + "<|im_end|>\n"
    query_str += '<|im_start|>context\n'

    query_len = len(tokenizer.tokenize(query_str))

    encoding = tokenizer.encode_plus(context, return_offsets_mapping=True)

    passage_ids = encoding["input_ids"]

    passage_offsets = encoding["offset_mapping"] # 每个token对应的字符位置

    chunk_label_len = max_len - query_len

    cnt = 0
    ret = []
    for j in range(0, len(passage_ids), chunk_label_len):
        temp_input_ids= passage_ids[j: j+chunk_label_len]
        temp_context = tokenizer.decode(temp_input_ids)

        new_msg = [
            {
                'role':'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': query
            },
            {
                'role': 'context',
                'content': temp_context
            }
        ]
        # assert len(self.tokenizer.encode(temp_context, add_special_tokens=False)) == len(temp_input_ids)
        new_sample = {
            'chunk_id': cnt,
            'messages': new_msg,
            "offset_mapping": passage_offsets[j: j+chunk_label_len],
            'query_len': query_len
        }
        cnt += 1
        ret.append(new_sample)
    return ret


