# Graph Embeddings & Graph Neural Networks

Exploration de méthodes de représentation de graphes et de réseaux de neurones sur graphes,
appliquées à deux jeux de données de référence : le club de karaté de Zachary et le dataset CORA.

**Auteurs :** Maram Nasr et Skander Haj Mabrouk

---

## Contenu du dépôt

| Notebook | Description |
|---|---|
| `deepwalk_node2vec.ipynb` | DeepWalk et Node2Vec — embeddings par marches aléatoires |
| `gcn_gat.ipynb` | GCN et GAT — réseaux de neurones sur graphes |

---

## Données

**Club de karaté de Zachary** — disponible directement via NetworkX :
```python
import networkx as nx
G = nx.karate_club_graph()
```

**CORA** — dataset de publications scientifiques :
```python
# Téléchargement
# https://raw.githubusercontent.com/AdrienGuille/adrienguille.github.io/main/assets/cora.npz

import numpy as np
data = np.load("cora.npz", allow_pickle=True)
# Clés : "adjacency_matrix", "tfidf_vectors", "labels"
```

---

## Notebook 1 — DeepWalk & Node2Vec

### Tâches

- **Club de karaté** : partitionner les sommets pour anticiper la scission du club (k-moyennes)
- **CORA** : prédire le domaine de publication des articles (régression logistique multinomiale)

### Expériences menées

**Partie 1 — Club de karaté**

- Implémentation from scratch de DeepWalk (marches aléatoires uniformes + Word2Vec)
- Implémentation from scratch de Node2Vec (marches aléatoires biaisées par les paramètres p et q)
- Clustering K-means sur les embeddings appris, évalué par Silhouette Score, ARI et pureté
- Comparaison K-means vs Label Propagation sur le graphe

**Partie 2 — CORA**

- Classification avec DeepWalk seul (structure du graphe uniquement)
- Classification avec Node2Vec seul, avec différentes configurations de biais (p, q)
- Classification avec TF-IDF seul (contenu textuel uniquement)
- Classification combinée : embeddings de graphe + TF-IDF
- Étude de l'impact des hyperparamètres sur Node2Vec + TF-IDF :
  - Longueur des marches aléatoires
  - Taille de la fenêtre de contexte
  - Dimension des embeddings
  - Nombre de marches par nœud

### Résultats clés (CORA)

| Méthode | Accuracy |
|---|---|
| DeepWalk — graphe seul | 0.809 |
| Node2Vec — graphe seul (meilleur config) | 0.818 |
| TF-IDF — texte seul | 0.764 |
| DeepWalk + TF-IDF | 0.834 |
| Node2Vec + TF-IDF | **0.840** |

---

## Notebook 2 — GCN & GAT

### Partie 1 — Club de karaté et GCN aléatoire

- Calcul explicite des matrices A̋ (adjacence avec auto-boucles) et D̋ (degré normalisé)
- Implémentation from scratch d'un GCN à 3 couches avec activation ReLU, paramétré par des matrices orthogonales tirées aléatoirement (sans entraînement)
- Visualisation des représentations apprises en 2D et 3D, colorées selon les deux communautés de terrain

### Partie 2 — CORA et classification semi-supervisée

- Classification semi-supervisée avec un **GCN** à 2 couches (via `torch-geometric`)
- Classification semi-supervisée avec un **GAT** à 2 couches, mécanisme d'attention multi-têtes (8 têtes)
- Démonstration mathématique et expérimentale qu'un GCN sans activation non-linéaire est équivalent à une **régression logistique** appliquée sur les features transformées A²_norm · X

### Résultats clés (CORA, 100 epochs)

| Modèle | Train | Val | Test |
|---|---|---|---|
| GCN (avec ReLU) | ~95.7% | ~90.7% | ~88.2% |
| GAT (multi-têtes) | — | — | ~89.3% |
| GCN sans activation | — | — | ~88.0% |
| Régression logistique sur A²·X | — | — | ~81.3% |

---

## Installation

```bash
pip install networkx numpy matplotlib seaborn scikit-learn gensim
pip install torch torch-geometric
```
---

## Références

- Perozzi et al. (2014) — *DeepWalk: Online Learning of Social Representations*
- Grover & Leskovec (2016) — *node2vec: Scalable Feature Learning for Networks*
- Kipf & Welling (2017) — *Semi-Supervised Classification with Graph Convolutional Networks*
- Veličković et al. (2018) — *Graph Attention Networks*
- Zachary (1977) — *An Information Flow Model for Conflict and Fission in Small Groups*
