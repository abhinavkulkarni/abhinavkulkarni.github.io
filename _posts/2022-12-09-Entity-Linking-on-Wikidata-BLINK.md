---
layout: post
title:  "Entity Linking on Wikidata - BLINK from Facebook AI Research"
excerpt: "BLINK is an entity linking framework from Facebook AI Research which implements a classic neural network based retriever-reader architecture for entity linking."
date:   2022-12-09 19:21:13 -0800
comments: true
categories: jekyll update
---

In this blog post, I will try to summarize an Entity Linking method for Wikidata from Facebook AI Research- [BLINK][blink-link]. It is open source and has Python libraries available.

### Entity Linking

Entity Linking (EL) is identifying mentions of entities in text and linking them to the corresponding entities in a knowledge base. The knowledge base can be Wikipedia or Wikidata. The task is also known as Entity Disambiguation (ED).

![el-image]

*In entity linking, each named entity is linked to a unique identifier. Often, this identifier corresponds to a Wikipedia page.* ([source](https://en.wikipedia.org/wiki/File:Entity_Linking_-_Short_Example.png))

Most state-of-the-art EL systems are two staged - candidate selection followed by candidate ranking / rescoring. 

The candidate selection stage generates a set of candidate entities for each mention. E.g., for mention "Paris", the candidate selection stage may generate the following collection of candidate entities:

- Paris (capital of France)
- Paris Hilton (American businesswoman, socialite, model, actress, singer, DJ, and author)
- Paris Saint-Germain F.C. (French professional football club)
- Paris Agreement (international agreement within the United Nations Framework Convention on Climate Change), etc.

The candidate ranking stage ranks the candidate entities in the order of their likelihood of being the correct entity for the mention. E.g., for the mention "Paris", we hope that the candidate ranking stage will rank the entity "Paris (capital of France)" as the most likely entity.

### BLINK EL Framework

BLINK is an EL method that uses a neural network in a classic retriever-reader fashion. The biencoder-based retriever stage uses two BERT-like transformer encoders to generate top-k candidates. The mention-encoder produces fixed-length vector embeddings for the mention and surrounding context whereas entity-encoder produces similar embeddings from the entity title and description. Top-k candidate entities are retrieved based on similarity (dot-product based).

These candidates and the mention are then jointly scored by a cross-encoder, a BERT-like transformer encoder.

### Biencoder

<!-- Write a LaTex block that explains how mention and entity encodings are calculated -->
$$
\begin{align*}
& \text{Mention input: } \\
& \mathbf{x_m} = \text{[CLS] ctx$_l$ [M$_s$] mention [M$_e$] ctxt$_r$ [SEP]} \\
& \text{Mention encoding: } \\
& \mathbf{y_m} = \text{BERT}_m(\mathbf{x_m}) \\
\\
& \text{Entity input: } \\
& \mathbf{x_e} = \text{[CLS] title [ENT] description [SEP]} \\
& \text{Entity encoding: } \\
& \mathbf{y_e} = \text{BERT}_e(\mathbf{x_e})\\
\\
& sim(m, e) = \mathbf{y_m} \cdot \mathbf{y_e}
\end{align*}
$$

where, 

$$ \text{[CLS], [M$_s$], [M$_e$], [ENT], [SEP]} $$ are [special symbols](https://stackoverflow.com/questions/62452271/understanding-bert-vocab-unusedxxx-tokens) reserved in BERT architecture,

$$ \text{ctx$_l$, ctx$_r$}$$ are left and right context of the mention and 

$\mathbf{y_m}$ and $\mathbf{y_e}$ are obtained by taking output embedding from $\text{[CLS]}$ token of respective encoders. 

For e.g., for the mention _"Elon Musk"_ in the following sentence:

> Twitter’s new owner Elon Musk on Thursday said he plans to introduce an option to make it possible for users to determine if the company has limited how many other users can view their posts.

```python
max_ctx_len = 32 - 2    # 2 for [CLS] and [SEP]

ctx_l = "Twitter’s new owner"
mention = "Elon Musk"
ctx_r = "on Thursday said he plans to introduce an option to make it possible for users to determine if the company has limited how many other users can view their posts."

ctx_l_tokens = tokenizer.tokenize(ctx_l)
ctx_r_tokens = tokenizer.tokenize(ctx_r)
mention_tokens = tokenizer.tokenize(mention)
mention_tokens = ["[unused0]"] + mention_tokens + ["[unused1]"]


n = len(mention_tokens)
n_left = (max_ctx_len - n) // 2
n_left = min(n_left, len(ctx_l_tokens))
n_right = max_ctx_len - n - n_left

ctx_tokens = ["[CLS]"] + \
ctx_l_tokens[-n_left:] + \
mention_tokens + \
ctx_r_tokens[:n_right] + \
["[SEP]"]

# Pass this through the context encoder to obtain contaxt embeddings y_m
```

BLINK only uses 32 tokens for the context (including mention tokens). This was found to be sufficient for most mentions. The context embeddings are contextual, so they are carrying semantic sigmanls from wider context (due to attention mechanism).

To obtain Elon Musk's entity encoding from his [Wikipedia entry](https://en.wikipedia.org/wiki/Elon_Musk), we use the following input to the entity encoder:

```python
max_cand_len = 128 - 2    # 2 for [CLS] and [SEP]

title = "Elon Musk"
description = "Elon Reeve Musk FRS is a business magnate, industrial designer, engineer, and philanthropist. He is the founder, CEO, CTO and chief designer of SpaceX; early investor, CEO and product architect of Tesla, Inc.; founder of The Boring Company; co-founder of Neuralink; and co-founder and initial co-chairman of OpenAI. A centibillionaire, Musk is one of the richest people in the world."

title_tokens = tokenizer.tokenize(title)
description_tokens = tokenizer.tokenize(description)
description_tokens = ["[unused2]"] + description_tokens

n = len(title_tokens)
n_desc = max_cand_len - n

cand_tokens = ["[CLS]"] + \
title_tokens + \
description_tokens[:n_desc] + \
["[SEP]"]

# Pass this through the entity encoder to obtain entity embeddings y_e
```

For encoding candidates (entities from Wikipedia with their titles and abstract), BLINK uses maximum of 128 tokens. This was found to be sufficient for most entities.

### Cross-encoder

After retrieving top-k candidates, we can jointly model a mention and entity as follows:

$$
\begin{align*}
& \text{Input: } \\
& \mathbf{x} = \text{[CLS] ctx$_l$ [M$_s$] mention [M$_e$] ctx$_r$ [SEP] title [ENT] description [SEP]} \\
& \mathbf{y_{m,e}} = \text{BERT}_{cross}(\mathbf{x}) \\
& s_{cross}(m, e) = \mathbf{y_{m,e} \cdot W}
\end{align*}
$$

where, $\mathbf{y_{m,e}}$ is the embedding of $\text{[CLS]}$ token of the cross-encoder and $\mathbf{W}$ is a linear layer applied to it in order to obtain a single score. For cross-encoder, BLINK uses maximum of 256 tokens. This was found to be sufficient for most mention-entity pairs.

### Training

Biencoders are trained to maximize the similarity between mention and entity pairs that are linked in the training data and minimize the similarity between (randomly sampled) mention-entity pairs that are not linked. This is done by using a contrastive loss function.

Crossencoder is trained to maximize the similarity between mention-entity pairs linked in the training data and minimize the similarity between other entities (among `top-k`) that were retrieved for the same mention by biencoder. This is done by using a contrastive loss function.

### Nearest Neighbor Index

Candidate enocdings can be pre-computed and stored in a nearest neighbor index such as [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/). Unlike, many other entity linking frameworks, BLINK can update candidate encodings periodically and re-index them. This does not require re-training the biencoder and cross-encoder.

[blink-link]: https://github.com/facebookresearch/BLINK
[el-image]: /assets/entity-linking/640px-Entity_Linking_-_Short_Example.png