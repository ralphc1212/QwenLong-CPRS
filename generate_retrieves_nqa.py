import torch
from sentence_transformers import SentenceTransformer, util
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk, Dataset, load_dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing



def retrieve_from_tokens(working_seq, model, tokenizer, k_size = 128, q_size = 32, topk = 100, query_chunk_size = 1000, step = 0):
    """
    给定一条输入序列，输出每个index上的retrieves结果，

    """
    import time

    working_dict = dict(working_seq)

    context = working_dict['document']['summary']['text']
    question = working_dict['question']['text']

    context_tokens = tokenizer.encode(context)

    # context = tokenizer.decode(context_tokens)

    # tokens_from_ids = tokenizer.convert_ids_to_tokens(context_tokens)

    context_chunks = []
    for idx in range(0, len(context_tokens) - k_size, k_size//4):
        context_chunks.append(context_tokens[idx : idx + k_size])

    context_emb_chunks = model.encode(tokenizer.batch_decode(context_chunks), batch_size=256)
    context_emb_chunks = torch.tensor(context_emb_chunks, dtype=torch.float16, device="cuda")

    question_emb = model.encode([question])
    question_emb = torch.tensor(question_emb, dtype=torch.float16, device="cuda")

    # retrieval results: index in the key list pool
    retrieved_results = util.semantic_search(question_emb, context_emb_chunks, top_k=topk, query_chunk_size=query_chunk_size)

    # print("step 4 search: ", time.time() - start)
    # start = time.time()
    # print(retrieved_results)
    top1_idx = retrieved_results[0][0]['corpus_id'] * (k_size // 4)
    top1_score = retrieved_results[0][0]['score']
    # top2_idx = retrieved_results[0][1]['corpus_id'] * (k_size//2)
    # top3_idx = retrieved_results[0][2]['corpus_id'] * (k_size//2)

    retrieved_tokens = context_tokens[top1_idx:top1_idx+k_size]

    return retrieved_tokens, top1_score

# 设置多进程的启动方法为 'spawn'
multiprocessing.set_start_method('spawn', force=True)

split = 'train'

# 加载模型和 tokenizer
model_path = "/media/nvme/yufei/projects/language-modelling/embedding_model/all-MiniLM-L6-v2"
tokenizer_path = "/media/nvme/yufei/projects/language-modelling/tokenizer/mamba-2.8b-instruct-openhermes-tokenizer.json"
dataset_path = "/media/nvme/yufei/projects/language-modelling/data/deepmind___narrativeqa"
output_path = "/media/nvme/yufei/projects/language-modelling/data/narrativeqa_retrieved/" + split + "/pieces/"


# 单 GPU 处理函数（放在 main 函数外部）
def process_split_on_gpu(data_split, gpu_id, inside_id):
    device = f'cuda:{gpu_id}'
    local_model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": "float16"}).to(device)  # 每个进程加载自己的模型到指定 GPU
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # 应用到数据集分片
    def add_retrieve_results(example):
        example["retrieved_tokens"], example["score"] = retrieve_from_tokens(example, local_model, tokenizer)
        return example

    # processed_split = data_split.map(add_retrieve_results, batched=True, batch_size=32)
    processed_split = data_split.map(add_retrieve_results)

    print(processed_split)
    processed_split.save_to_disk(f"{output_path}_gpu_{gpu_id}_{inside_id}")  # 保存每个 GPU 的处理结果
    return f"GPU {gpu_id} processing complete"

def main():
    # 加载数据集

    dataset = load_dataset(dataset_path)
    dataset_len = len(dataset[split])
    num_per_gpu = 4

    # print(dataset['train'][0].keys())
    # 分割数据集
    num_gpus = torch.cuda.device_count()  # 假设为 8
    split_size = dataset_len // (num_gpus * num_per_gpu)
    datasets_split = [dataset[split].select(range(i * split_size, (i + 1) * split_size)) for i in range(num_gpus * num_per_gpu)]

    # process_split_on_gpu(datasets_split[0], 0, 0)

    processes = []
    for gpu_id in range(num_gpus):
        for i in range(num_per_gpu):
            p = multiprocessing.Process(target=process_split_on_gpu, args=(datasets_split[gpu_id*num_per_gpu+i], gpu_id, i))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
