import os
import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import test_collate_fn


def get_predictions(dataset, embeddings):
    gt = []
    for i in range(dataset.len_q):
        positives = dataset.get_non_negatives(i)
        gt.append(positives)

    # get distance
    qFeat = embeddings[: dataset.len_q].cpu().numpy().astype("float32")
    dbFeat = embeddings[dataset.len_q :].cpu().numpy().astype("float32")

    print("==> Building faiss index")
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    dis, predictions = faiss_index.search(qFeat, 20)

    return predictions, gt

def evaluate_4drad_dataset(model, device, dataset, params):

    model.eval()
    quantizer = params.model_params.quantizer
    val_collate_fn = test_collate_fn(
        dataset,
        quantizer,
        params.batch_split_size,
        params.model_params.input_representation,
    )

    dataloder = DataLoader(
        dataset,
        batch_size=params.val_batch_size,
        collate_fn=val_collate_fn,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )
    
    embeddings_dataset = torch.empty((len(dataset), 256))
    
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(dataloder, desc="==> Computing embeddings")):
            embeddings_l = []
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                ys = model(minibatch)
                embeddings_l.append(ys["global"])
                del ys

            embeddings = torch.cat(embeddings_l, dim=0)
            del embeddings_l
            embeddings_dataset[
                iteration
                * params.val_batch_size : (iteration + 1)
                * params.val_batch_size
            ] = embeddings.detach()
            torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    predictions, gt = get_predictions(dataset, embeddings_dataset)
    recalls = compute_recall(predictions, gt)
    print("==> Evaluation completed!")

    return recalls

def compute_recall(predictions, gt, n_values=[1, 5, 10, 20]):
    numQ = 0
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) == 0:
            continue
        else:
            numQ += 1
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
    return all_recalls

def save_recall_results(model_name, dataset_name, recall_metrics, result_dir):
    # Create a directory for the results if it doesn't exist
    
    # Construct filename and filepath
    filename = f"{model_name}.txt"
    filepath = os.path.join(result_dir, filename)
    
    # Prepare the line to be appended to the file
    recall_results_str = f"Dataset: {dataset_name}, Recall@1: {recall_metrics[1]:.4f}, Recall@5: {recall_metrics[5]:.4f}, Recall@10: {recall_metrics[10]:.4f}\n"
    
    # Check if file exists and append the results; if not, create the file and write the results
    with open(filepath, 'a') as file:
        file.write(recall_results_str)
    print(f"==> Results saved to {filepath}")