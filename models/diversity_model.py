from sentence_transformers import SentenceTransformer

import torch
from torch.nn import functional as F


class DiversityModel():
  def __init__(self, modelname, device):
    self.model = SentenceTransformer(modelname, device = device, trust_remote_code= True)
    self.device = device

  def get_embedding(self, texts):
    return self.model.encode(texts, convert_to_tensor=True)

  def get_avg_distance(self, text, compared_texts, metric='cosine'):
    embeddings = self.get_embedding([text]+compared_texts)
    if metric == 'cosine':
      embeddings = F.normalize(embeddings, p=2, dim=1)
      similarity = torch.mm(embeddings, embeddings.T)
      distance = 1 - similarity
    elif metric == 'euclidean':
      distance = torch.cdist(embeddings, embeddings, p=2)
    return torch.mean(distance[0][1:]).item()

  def get_diversity(self, texts, metric='cosine'):
    embeddings = self.get_embedding(texts)
    if metric == 'cosine':
      embeddings = F.normalize(embeddings, p=2, dim=1)
      similarity = torch.mm(embeddings, embeddings.T)
      distance = 1 - similarity
    elif metric == 'euclidean':
      distance = torch.cdist(embeddings, embeddings, p=2)
    
    # turn the eye of distance to 0
    distance[torch.eye(distance.shape[0]).bool()] = 0

    div_scores = []

    for i in range(distance.shape[0]):
      cur_distance = torch.sum(distance[i])/(distance.shape[0]-1)
      div_scores.append(cur_distance.item())
    return div_scores
  
  def pair_dists(self, texts, metric='cosine'):
    embeddings = self.get_embedding(texts)
    if metric == 'cosine':
      embeddings = F.normalize(embeddings, p=2, dim=1)
      similarity = torch.mm(embeddings, embeddings.T)
      distance = 1 - similarity
    elif metric == 'euclidean':
      distance = torch.cdist(embeddings, embeddings, p=2)
    
    # turn the eye of distance to 0
    distance[torch.eye(distance.shape[0]).bool()] = 0


    # turn all distance elements into numpy
    return distance.detach().cpu().float().numpy()

    

