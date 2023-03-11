# Controlling-LLMs-Consistency

## [Experiment 1](https://github.com/harshraj172/Controlling-LLMs-Consistency/blob/main/adv_pp-rule_based.ipynb)
To improve the consistency scorer function and the paraphrase generation method as described in our [previous work](https://arxiv.org/abs/2211.05853).

### Dataset - [TruthfulQA](https://arxiv.org/abs/2109.07958)
[see sample](https://huggingface.co/datasets/truthful_qa)

### Rule based Paraphraser
In the previous work one of the method to generate paraphrases was to prompt LLMs (GPT3) with few shot prompts of a sentence and its respective paraphrase. But now we introduced some grammatical rules according to which a sentences can be paraphrased. This helps in producing more diverse paraphrases.

### Consistency Scorer
In the previous work we were using a mix of several semantic and lexical sentence similarity metrics like [ROUGE1](https://aclanthology.org/W04-1013/), [BLEURT](https://arxiv.org/abs/2004.04696), [BERTs](https://arxiv.org/abs/1904.09675), etc. 
Now we tried to improve the scoring function using LLMs as the consistency scorer itself, with some chain of prompts. The prompt (see [notebook](https://github.com/harshraj172/Controlling-LLMs-Consistency/blob/main/adv_pp-rule_based.ipynb)) consistes of 2 templates.

1. "template_eval_step1" (see 4th cell [notebook](https://github.com/harshraj172/Controlling-LLMs-Consistency/blob/main/adv_pp-rule_based.ipynb)) - designed to extract the facts & figures of an answer. To compare the consistencies of the answers generated from both the original and the paraphrased question we compare the basic content. The template gives some few shot examples on how to get the basic content of an answer based on the question.
In the template:
```
Context - Answer (from paraphrased question / original question)

Question - Question (paraphrased / original)

Answer - Basic Content (asked to generate)
```
Using the prompt, retrive the main content of the answers for both the original and the paraphrased versions of the questions.

2. "template_eval_step2" (see 5th cell [notebook](https://github.com/harshraj172/Controlling-LLMs-Consistency/blob/main/adv_pp-rule_based.ipynb)) - designed to compare the retrieved facts (Basic Content) for the both the type of answers. 
In the template:
```
Question - original question

Answer1 - Basic Content (of original answer)

Answer2 - Basic Content (of paraphrased answer)
```
Via few-shot prompting the comparison yields "yes" if the two texts talk the same thing (are consistent) else "no" through which we decide the consistency score of 1 and 0 respectively.


- Using the above methods of paraphrasing and consistency scoring, we constructed a dataset consisting of the question, paraphrased question, type of paraphrasing rule used, original output (from question), paraphrased output (from paraphrased question) and consistecy score. 
