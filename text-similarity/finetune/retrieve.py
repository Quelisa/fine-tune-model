import torch
import logging
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Retrieve(object):
    def __init__(
        self,
        model,
        device,
        index_dataset,
        num_cells=100,
        num_cells_in_search=10,
    ):
        self.model = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

    def encode(self, data_loader, batch_size=64):

        embeddings_list = []
        labels_list = []

        for step, (inputs_ids, attention_mask,
                   labels) in tqdm(enumerate(data_loader)):
            ids, att = inputs_ids.to(self.device), attention_mask.to(
                self.device)
            outputs = self.model(ids, att)
            embeddings_list.append(outputs.cpu())
            labels_list.append(labels)

        embeddings = torch.cat(embeddings_list, 0)
        labels = torch.cat(labels_list, 0)
        return embeddings, labels

    def build_index(self, index_loader):

        self.index["index"], self.index["labels"] = self.encode(index_loader)

    def similarity(self, query_vecs, key_vecs):

        single_query, single_key = len(query_vecs.shape) == 1, len(
            key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        # N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
        if single_key:
            similarities = float(similarities[0])

        return similarities

    def search(self, queries, device, threshold=0.1, top_k=5):

        similarities = self.similarity(queries, self.index["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))
        id_and_score = sorted(id_and_score, key=lambda x: x[1],
                              reverse=True)[:top_k]
        results = [(self.index["labels"][idx], score)
                   for idx, score in id_and_score]
        return results

    def evalute(self, data_loader):

        embeddings, labels = self.encode(data_loader)
        top1_acc = 0
        for i in range(embeddings.size(0)):
            res = self.search(embeddings[i])
            if len(res) > 1 and res[1][0] == labels[i]:
                top1_acc += 1
        return top1_acc / embeddings.size(0)
