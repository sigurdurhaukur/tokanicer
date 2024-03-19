# tokenicer

Tinkering with tokenization and NLP for Icelandic.

Visualizations from https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer

The project is based on the video by Andrej Karpathy on tokenization: https://www.youtube.com/watch?v=zduSFxRajkE

How do we make great language models for smaller languages such as Icelandic?

I believe that to develop a good language model, the tokenizer must be great.

Currently, the architecture of language models such as GPT and BERT depends on tokenization. Before the training of language models all text must be given a numerical representation, or a respective token. The act of converting text into tokens is called tokenization, and the algorithm that performs tokenization is called a tokenizer.

If the training data is badly tokenized it becomes difficult for the language model to converge. Therefore, to develop a good language model, the tokenizer must be great.

The question is: how good are the tokenizers of state-of-the-art language models for Icelandic and what can we do to improve them?

I measured the performance of 3 different tokenizers, here are the results:

1. The GPT-2 tokenizer

   -

2. The GPT-4 tokenizer

3. Jón Friðrik Daðason's gpt-2-igc-is tokenizer
