import os
import sys
import statistics

import torch
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

default_quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def load_model_and_processor(model_id, quantization_config=default_quantization_config):
    """Loads a model and processor for LlavaNext.

    Args:
        model_id (str): model id from Hugging Face 
        quantization_config (optional): The quantization configuration for the model. Defaults to default_quantization_config.

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

    processor = LlavaNextProcessor.from_pretrained(model_id)

    return model, processor

def ask_question_of_image(
        image,
        prompt_template,
        question,
        model,
        processor):
    """
    Run inference using a given model and processor on an image and question.

    Parameters:
    - image: a PIL Image object
    - prompt_template: A template string for the prompt, which must include `{question}` for formatting.
    - question: The question related to the image.
    - model: The model to use for inference.
    - processor: The processor to use for preparing the inputs for the model.

    Returns:
    - The inference result, excluding the part of the prompt.
    """

    # Prepare the prompt using the template and the question
    prompt = prompt_template.format(question=question)

    # Prepare the inputs for the model
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # Run the inference
    output = model.generate(**inputs, max_new_tokens=256)

    # Decode the output
    full_output = processor.decode(output[0], skip_special_tokens=True)

    # Extract everything after the last word in the prompt
    result = full_output.split(prompt.split()[-1], 1)[-1].strip()

    print(result)


def prepare_dataset(dataset_id, split=None, seed=42, subset_size=250):
    """Prepare a dataset for use with LlavaNext.

    Args:
        dataset_id (str): The ID of the dataset to be loaded.
        split (str, optional): The split of the dataset to be loaded. Defaults to None.
        seed (int, optional): The seed for shuffling the dataset. Defaults to 42.
        subset_size (int, optional): The number of examples to be loaded. Defaults to 250.
    """
    dataset = load_dataset(dataset_id, split=split) if split else load_dataset(dataset_id)
    dataset = dataset.shuffle(seed=seed).select(range(subset_size))
    dataset.save_to_disk(f'./{dataset_id}_subset')

def run_inference_hf_single(item, prompt_template, output_key, model, processor):
    """
    Adapted function to work with a single item from the dataset for use with the HF datasets map function.

    Note: This function is designed to be used inside a wrapper function for datasets.map()
    """
    # Prepare the prompt
    prompt = prompt_template.format(question=item["question"])

    # Prepare inputs
    inputs = processor(prompt, images=item["image"], return_tensors="pt").to("cuda:0")

    # Run inference
    output = model.generate(**inputs, max_new_tokens=256)

    # Decode the output
    result = processor.decode(output[0], skip_special_tokens=True)

    # Update the item
    item[output_key] = result.split(prompt.split()[-1], 1)[-1].strip()

    return item

def run_inference_on_dataset(dataset, prompt_template, output_key, model, processor):
    """
    Apply inference to the entire dataset using the HF datasets map function.

    Parameters:
    - dataset: The dataset to process.
    - prompt_template: Template string for the prompt.
    - output_key: Key for storing inference results in the dataset.
    - model: The model to use for inference.
    - processor: The processor for preparing inputs.
    """
    # Define a wrapper function for use with dataset.map
    def map_function(item):
        return run_inference_hf_single(item, prompt_template, output_key, model, processor)

    # Apply the map function to the entire dataset
    result_dataset = dataset.map(map_function)
    return result_dataset

def textvqa_process_results(doc, result):
    """
    Calculate the accuracy of a single model's answer against a set of ground truth answers.

    This function processes both the model's answer and the ground truth answers through a specified
    answer processor. It calculates the accuracy based on the presence of the ground truth answer
    within the model's answer, rather than requiring an exact match.

    Parameters:
    - doc (dict): A dictionary representing a single row from a dataset, expected to contain
      a list of ground truth answers under the key "answers".
    - result (str): The model's answer as a string.

    Returns:
    - float: The accuracy of the model's answer, calculated as the mean of matches between
      the processed model's answer and the processed ground truth answers.

    Note:
    - The function assumes the existence of an EvalAIAnswerProcessor class that is used to
      process the answers for comparison.
    """
    eval_ai_processor = EvalAIAnswerProcessor()
    # Process model output once
    processed_model_answer = eval_ai_processor(doc[result])

    if "answers" in doc and doc["answers"] is not None:
        # Process ground truth answers once
        processed_gt_answers = [eval_ai_processor(answer) for answer in doc["answers"]]
        # Calculate accuracy using list comprehension
        gtAcc = [1 if gt_ans in processed_model_answer else 0 for gt_ans in processed_gt_answers]
        # Safely compute the mean, avoiding division by zero
        accuracy = statistics.mean(gtAcc) if gtAcc else 0
    else:
        accuracy = 0

    return accuracy

def add_accuracy_score(example, columns_to_evaluate):
    """
    Calculate and add accuracy scores for specified columns in a single dataset example.

    Parameters:
    - example (dict): A single example from a dataset.
    - columns_to_evaluate (list of str): Columns for which to calculate and add accuracy scores.

    Returns:
    - dict: The original example dictionary updated with new accuracy score columns.
    """
    for column in columns_to_evaluate:
        if column in example:
            score_key = f"{column}_score"
            example[score_key] = textvqa_process_results(example, column)
    return example