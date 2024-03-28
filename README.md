# 🌋 LLaVA-NeXT (aka LLaVA-1.6) is here! 

LLaVA (Large Language and Vision Assistant) is an open-source multimodal AI assistant capable of processing both text and images, boasts enhanced reasoning, optical character recognition (OCR), and world knowledge capabilities.

Here are the key improvements in LLaVA-NeXT compared to the previous LLaVA-1.5 version:

 - 🏆 According to the authors, on several benchmarks, LLaVA-NeXT-34B outperforms Gemini Pro, a state-of-the-art multimodal model. It achieves state-of-the-art performance across 11 benchmarks with simple modifications to the original LLaVA model. 

 - 👩🏾‍🏫 The authors curated high-quality user instruction data that meets two criteria: diverse task instructions amd superior responses. 

They combined two data sources for this: 

(1) Existing GPT-V data in [LAION-GPT-V](https://huggingface.co/datasets/laion/gpt4v-dataset) and [ShareGPT-4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V). 

(2) A 15K visual instruction tuning dataset from the LLaVA demo covering various applications. 

 - Major improvements include enhanced:
   - 🧠 reasoning capabilities 
   - 👀 optical character recognition (OCR)
   - 🌎 world knowledge compared to the previous LLaVA-1.5 model.

🧧 LLaVA-NeXT has an emerging zero-shot capability in Chinese **despite only being trained on English multimodal data**. Its Chinese multimodal performance is surprisingly good.

LLaVA-NeXT has been [open-sourced and is available in the Hugging Face Transformers library](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf), making it one of the best open-source vision-language models currently available.

# What's this repo doing?

We'll explore for ourselves how the `llava-hf/llava-v1.6-mistral-7b-hf` and `llava-hf/llava-v1.6-vicuna-7b-hf` models perform!

Aggregrate metrics are nice and all, but I just wanna get hands on with these models and see for myself how they perform. To that end, I've wrote some scripts for inference and evaluation based on the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework the authors of the paper used. 

I then use `fiftyone` to visuale how the model responds to visual and text prompts to get a better sense of the model performs.

Basically, vibe-checking a LMM!