# TCotMechanism

This repository contains code for the paper： 

Unveiling the Mechanisms of Explicit CoT Training: How Chain-of-Thought Enhances Reasoning Generalization

## Requirements

To install the experiment, please install the pip file. 

```setup
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

cd transformers
pip install -e .
cd ..

cd simpletransformers
pip install -e .
cd ..
```

## Data

You can run the following example  to generate data:

```data
TCotMechanism/composition.ipynb
```
>📋 You can adjust the following hyperparameters.
>
>NUM_ENTITY_IN = 2000  #  $|\mathcal{E}|$
>
>NUM_RELATION = 200  #  $|\mathcal{R}|$
>
>OOD_ratio = 0.05  # $|S_{\text{OOD}}|: |S_{\text{ID}}|=5\%:95\%$. 
>
>lambda_noise = 0.4 $ noise ratio
>
>phi=7.2  # $|S_{\text{ID}_{\text{train}}}^{(2)}|/|S_{\text{ID}}|$

## Training

```train example
chmod +x TCotMechanism/run.sh  #ensure execute permissions
TCotMechanism/run.sh
```

## Evaluation

```test example
TCotMechanism/eval_qa.py --dir <path_to_saved_checkpoints>
```

##  Two-stage Generalizing Circuit

```test example
TCotMechanism/tracing_composition.py  #check the first stage
TCotMechanism/tracing_composition1.py  #check the second stage
```

## Contributing

>📋  Training large language models (LLMs) with high-quality Chain-of-Thought (CoT) annotations has emerged as a widely adopted strategy in the industry because it significantly enhances reasoning capabilities. To fully understand this strategy, two questions naturally arise: (Q1) What advantages does training with CoT offer compared to training without CoT? (Q2) If there are advantages, what are the underlying mechanisms of explicit CoT training? Analyzing the advantages and mechanisms of CoT training is challenging due to the many factors involved. To tackle this, we conduct a detailed analysis with clear, controllable data distributions and identify the following interesting phenomena:  (1) The advantages:  (i) Training with CoT markedly improves reasoning generalization, extending it from in-distribution (ID) to both ID and out-of-distribution (OOD) scenarios, while also speeding up convergence; (ii) Even when training with CoT includes a certain range (noise ratio $\xi$) of erroneous reasoning steps, it still enables the model to learn reasoning patterns, leading to systematic generalization. (2) The internal mechanisms:  (i) The data distribution (e.g., ratio $\lambda$ and pattern) plays a crucial role in influencing the model's systematic generalization; (ii) CoT training (with two-hop facts) internalizes reasoning into a two-stage generalizing circuit, where the number of stages corresponds to the explicit reasoning steps during training. Our findings elucidate the mechanisms underlying explicit CoT training and offer critical insights into tuning strategies for LLMs to achieve robust generalization.

<img src="pics\noise_only_t.jpg" alt="ex1" style="zoom:72%;" />

<img src="pics\circuit.jpg" alt="results" style="zoom:85%;" />
