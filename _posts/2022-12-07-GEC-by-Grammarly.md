---
layout: post
title:  "Grammatical Error Correction & Text Simplification by Grammarly"
date:   2022-12-07 19:21:13 -0800
comments: true
categories: jekyll update
---

A short explaination of the Grammatical Error Correction (GEC) and Text Simplification (TS) tasks and the GECToR model by Grammarly.
-----

In this paper, I try to summarize the main ideas behind the two excellent papers published by Grammarly's research team for the tasks of [GEC][GEC paper] (Grammatical Error Correction) and [Text Simplification][Text simplification paper] tasks. Both papers are very well written and easy to understand.

GEC Task
-----
[CoNLL][CoNLL website] is a yearly conference organized by SIGNLL (ACL's Special Interest Group on Natural Language Learning). [CoNLL-2014][CoNLL-2014 task] introduced the GEC task which aims to evaluate systems for automatically detecting and correcting grammatical errors present in English essays written by second language learners of English. 

To illustrate, consider the following sentence:

```
Social network plays a role in providing and also filtering information.
```

The noun number error _networks_ needs to be corrected (network → networks). This necessitates the correction of a subject-verb agreement error (plays → play).

As an another example, consider the following sentence:

```
Nothing is absolute right or wrong.
```

There is a word form error (absolute → absolutely)
in this sentence.

The performance is measured by how well the two sets **g** and **e** match, in the form of recall **R**, precision **P**, and **F0.5** measure, where **g** set of gold-standard edits and **e** is the set of system edits.

For example, consider the following sentence:

```
There is no a doubt , tracking system has brought many benefits in this information age .

g = {a doubt → doubt, system → systems, has → have}
```

Suppose, the system edits are:

```
There is no doubt , tracking system has brought many benefits in this information age .

e = {a doubt → doubt}
```

In this case,

$$
R = \frac{1}{3}, P = \frac{1}{1}
, F0.5 = \frac{1 + 0.5^2}{\frac{1}{P} + \frac{0.5^2}{R}} = \frac{5}{7}
$$

GECToR Model by Grammarly
-----
While Neural Machine Translation (NMT)-based approaches have become
the preferred method for GEC, they suffer from the lack of training data and slow inference speeds. NMT proposed to solve GEC task by learning the maping between _source sentence_ → _target sentence_ where the target sentence is the one with the errors corrected.

GECToR proposes possible 5000 edits for each token in the sentence. There are a few token independent edits such as 

```
KEEP (no edit), 

DELETE (delete the token) 
```

and few token dependent edits such as 

```
REPLACE(t) (replace the token with a with token t),

APPEND(t) (append the token with token t), 

VERB FORM (VB → VBD) (change the verb form from VB to VBD)
for e.g. play → played
```

![gector-edit-tags]

Model Architecture
-----

GECToR proposes an encoder-only model. For every token in the model, it predicts the probability of the token needing to be corrected and probabilities of possible edits.

$$
\begin{aligned}
\mathbf{\tilde{X}} &= \text{ENCODER}(\mathbf{X}) \\
\mathbf{p_{e,i}} &= \text{softmax}(\mathbf{W_e} \mathbf{\tilde{X_i}} + \mathbf{b_e}) \\
\mathbf{p_{t,i}} &= \text{softmax}(\mathbf{W_t} \mathbf{\tilde{X_i}} + \mathbf{b_t}) \\
\end{aligned}
$$

where,

 $\mathbf{X}$ are the input token embeddings, 

$\mathbf{\tilde{X}}$ are the output contextual token embeddings from a BERT-like transformer encoder, 

$\mathbf{p_e}$ are the binary probabilities of each token needing to be corrected, and 

$\mathbf{p_t}$ are the probabilities of edits for each token

The error probabilities $\mathbf{p_e}$ act as a gate for the edit probabilities $\mathbf{p_t}$, that is, if $\mathbf{p_e}$ is lower than a threshold, then the edit probabilities $\mathbf{p_t}$ are not used and the token is kept as is. Moreover, among edit tag probabilities, probability of the tag KEEP is set to a fixed threshold value - both of these mechanisms force the model to make more accurate predictions. Other than this, the model also calculates overall sentence error probability - if it is above a certain threshold, only then further token-wise edits are made. `sentence_error_probability`, `min_edit_probability` and `keep_probability` are the hyperparameters in the model.

Since each word in the source sentence may be split into multiple sub-word units, we pass the first sub-word unit of each word into linear-softmax layers to obtain $\mathbf{p_{e,i}}$ and $\mathbf{p_{t,i}}$ probabilities.

The model performs iterative inference on a given source sentence until it arrives at an error-free sentence. The maximum number of such iterations is a hyperparameter that trades precision for latency.

![gector-inference]

The paper has more details about various BERT-like encoder architectures that were tried and respective results.

[GEC paper]: https://arxiv.org/abs/2005.12592
[Text simplification paper]: https://arxiv.org/abs/2103.05070
[CoNLL website]: https://conll.org/
[CoNLL-2014 task]: https://aclanthology.org/W14-1701.pdf
[gector-inference]: /assets/gector_inference.png
[gector-edit-tags]: /assets/gector_edit_tags.png