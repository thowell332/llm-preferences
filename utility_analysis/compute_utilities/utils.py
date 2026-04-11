# utils.py

import asyncio
import json
import os
import yaml
import numpy as np
import random
from typing import List, Dict, Any, Optional, Union
from .llm_agent import HuggingFaceAgent, HuggingFaceAgentLogitsPrediction, vLLMAgent, vLLMAgentBaseModel
import re
from tqdm import tqdm


# ========================== GENERAL HELPER FUNCTIONS ========================== #

def convert_numpy(obj):
    """
    Recursively convert numpy data types in the object to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int_, np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def load_config(config_path: Optional[str], config_key: str, default_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with default path handling.
    
    Args:
        config_path: Optional path to config file. If None, uses default path
        config_key: Key to use in the config file
        default_filename: Default filename to use if config_path is None
        
    Returns:
        Dictionary containing configuration for the specified key
        
    Raises:
        ValueError: If config file doesn't exist or key not found
    """
    if config_path is None:
        if default_filename is None:
            raise ValueError("config_path is None and default_filename is None")
        config_path = os.path.join(os.path.dirname(__file__), default_filename)
        
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    if config_key not in config:
        raise ValueError(f"Config key '{config_key}' not found in {config_path}")
        
    return config[config_key]


def flatten_hierarchical_options(hierarchical_options):
    """
    Flattens a hierarchical options dictionary into a list of options.
    """
    flattened = []
    for category, options in hierarchical_options.items():
        flattened.extend(options)
    return flattened


# ========================== GENERATE AND PARSE RESPONSES ========================== #

def create_agent(model_key, temperature=0.0, max_tokens=10, trust_remote_code=True, **kwargs):
    """
    Creates a local inference agent based on the model key from models.yaml.

    Args:
        model_key: Key of the model in models.yaml (e.g., 'llama-32-1b-instruct')
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum number of tokens to generate
        trust_remote_code: Whether to trust remote code (for HuggingFace/vLLM)
        **kwargs: Ignored (kept for backward-compatible YAML configs)
    """
    models_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models.yaml')
    with open(models_yaml_path, 'r') as f:
        models_config = yaml.safe_load(f)

    model_config = models_config.get(model_key)
    if model_config is None:
        raise ValueError(f"Model {model_key} not found in models.yaml")

    model_type = model_config['model_type']
    accepts_system_message = model_config.get('accepts_system_message', True)

    if model_type == 'huggingface':
        return HuggingFaceAgent(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get('tokenizer_path')
        )
    elif model_type == 'huggingface_logits':
        return HuggingFaceAgentLogitsPrediction(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=model_config.get('accepts_system_message', False),
        )
    elif model_type == 'vllm':
        return vLLMAgent(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get('tokenizer_path')
        )
    elif model_type == 'vllm_base_model':
        return vLLMAgentBaseModel(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get('tokenizer_path')
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of "
            "['huggingface', 'huggingface_logits', 'vllm', 'vllm_base_model']."
        )



# ========================== GENERATE AND PARSE RESPONSES ========================== #
def parse_responses_forced_choice(
    raw_results,
    with_reasoning=False,
    choices=['A', 'B'],
    verbose=True
):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_raw_responses]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [raw_response_1, raw_response_2, ...]}
    :param with_reasoning:  if True, parse based on "Answer: X" or "Answer: Y" in text
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable

    Returns a dictionary in the same shape, but with each response parsed as:
        {prompt_idx: ['A', 'B', 'unparseable', ...]}
    Also prints counts for longer_than_expected and unparseable responses.
    """
    parsed_results = {}
    counts = {
        'longer_than_expected': 0,
        'unparseable': 0
    }

    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert len(choices[0]) == 1 and len(choices[1]) == 1, (
        "each choice in `choices` must be a single character."
    )
    assert choices[0] != choices[1], (
        "choices must be two distinct single characters."
    )

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = '|'.join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf'Answer:\s*({pattern_str})', re.IGNORECASE)

    # Precompile patterns for non-reasoning mode
    choice_patterns = [re.compile(rf'(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)') for c in choices]

    for prompt_idx, responses in raw_results.items():
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            continue

        parsed_list = []
        for response in responses:
            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append('unparseable')
                counts['unparseable'] += 1
                continue

            if with_reasoning:
                # Reasoning mode: must find "Answer: X" or "Answer: Y".
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Normalize the matched choice by matching it to one of choices[0] or choices[1].
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')
                else:
                    counts['unparseable'] += 1
                    parsed_list.append('unparseable')
            else:
                # Non-reasoning mode
                # First check if response is exactly one of the choices
                response = response.strip()
                if response == choices[0]:
                    parsed_list.append(choices[0])
                elif response == choices[1]:
                    parsed_list.append(choices[1])
                else:
                    # Check if response is longer than expected
                    if len(response) > max(len(choices[0]), len(choices[1])):
                        counts['longer_than_expected'] += 1
                    
                    # Check for choices appearing with space/newline before them
                    matches = [bool(pattern.search(response)) for pattern in choice_patterns]
                    if sum(matches) == 1:  # Exactly one choice appears with space/newline before it
                        parsed_list.append(choices[matches.index(True)])
                    else:  # Neither or both choices appear with space/newline before them
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(f"Number of responses longer than expected: {counts['longer_than_expected']}")
        print(f"Number of unparseable responses: {counts['unparseable']}")

    return parsed_results


async def parse_responses_forced_choice_freeform(
    raw_results,
    system_prompt,
    user_prompt,
    preference_data,
    with_reasoning=False,
    choices=['A', 'B'],
    verbose=True,
    free_form_mode=False,
    lmjudge_client=None
):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_raw_responses]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [raw_response_1, raw_response_2, ...]}
    :param with_reasoning:  if True, parse based on "Answer: X" or "Answer: Y" in text
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable
    :param free_form_mode:  no longer supported (raises if True)
    :param lmjudge_client:  no longer supported (raises if not None)

    Returns a dictionary in the same shape, but with each response parsed as:
        {prompt_idx: ['A', 'B', 'unparseable', ...]}
    Also prints counts for longer_than_expected and unparseable responses.
    """
    if free_form_mode or lmjudge_client is not None:
        raise ValueError(
            "free_form_mode and lmjudge_client are no longer supported (remote LLM judge removed). "
            "Use parse_responses_forced_choice with with_reasoning=True or structured outputs."
        )

    parsed_results = {}
    counts = {
        'longer_than_expected': 0,
        'unparseable': 0
    }

    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert len(choices[0]) == 1 and len(choices[1]) == 1, (
        "each choice in `choices` must be a single character."
    )
    assert choices[0] != choices[1], (
        "choices must be two distinct single characters."
    )

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = '|'.join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf'Answer:\s*({pattern_str})', re.IGNORECASE)

    # Precompile patterns for non-reasoning mode
    choice_patterns = [re.compile(rf'(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)') for c in choices]

    for prompt_idx, responses in tqdm(raw_results.items(), desc="Processing responses"):
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            continue

        parsed_list = []
        for response in responses:
            # print(f"Model's raw response is: {response}")
            
            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append('unparseable')
                counts['unparseable'] += 1
                continue
            
            if with_reasoning:
                # Reasoning mode: must find "Answer: X" or "Answer: Y".
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Normalize the matched choice by matching it to one of choices[0] or choices[1].
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')
                else:
                    counts['unparseable'] += 1
                    parsed_list.append('unparseable')
            else:
                # Non-reasoning mode
                # First check if response is exactly one of the choices
                response = response.strip()
                if response == choices[0]:
                    parsed_list.append(choices[0])
                elif response == choices[1]:
                    parsed_list.append(choices[1])
                else:
                    # Check if response is longer than expected
                    if len(response) > max(len(choices[0]), len(choices[1])):
                        counts['longer_than_expected'] += 1
                    
                    # Check for choices appearing with space/newline before them
                    matches = [bool(pattern.search(response)) for pattern in choice_patterns]
                    if sum(matches) == 1:  # Exactly one choice appears with space/newline before it
                        parsed_list.append(choices[matches.index(True)])
                    else:  # Neither or both choices appear with space/newline before them
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(f"Number of responses longer than expected: {counts['longer_than_expected']}")
        print(f"Number of unparseable responses: {counts['unparseable']}")

    return parsed_results



async def generate_responses(agent, prompts, system_message=None, K=10, timeout=5, use_cached_responses=False, prompt_idx_to_key=None, cached_responses_mapping=None, verbose=True):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        prompts: List of prompt strings
        system_message: The system message to include in each prompt (if supported)
        K: Number of completions to generate for each prompt
        timeout: Timeout in seconds for each API call
        use_cached_responses: Whether to use cached responses
        prompt_idx_to_key: Mapping from prompt indices to cache keys
        cached_responses_mapping: Dictionary of cached responses
        verbose: Whether to print verbose output

    Returns:
        A dictionary mapping prompt indices to their generated responses.
    """
    
    # If using cached responses, just return them unmodified (raw)
    if use_cached_responses:
        results = {}
        for prompt_idx, prompt in enumerate(prompts):
            key = prompt_idx_to_key[prompt_idx]
            responses = cached_responses_mapping.get(key, [])
            if not responses and verbose:
                print(f"No cached responses found for prompt index {prompt_idx}, key {key}")
            results[prompt_idx] = responses[:K]
        return results
    
    # Prepare messages
    messages = []
    for prompt in prompts:
        message = []
        # Only add system message if the model accepts it
        if system_message is not None and agent.accepts_system_message:
            message.append({'role': 'system', 'content': system_message})
        message.append({'role': 'user', 'content': prompt})
        messages.append(message)
    
    # Duplicate messages K times to get K completions for each prompt
    messages_k = messages * K

    responses = agent.completions_batch(messages_k)
    
    # Reshape responses into groups of K for each prompt
    num_prompts = len(prompts)
    responses_by_prompt = {}
    for i in range(num_prompts):
        responses_by_prompt[i] = responses[i::num_prompts]
    return responses_by_prompt


async def evaluate_holdout_set(
    graph,
    agent,
    utility_model,
    utilities,
    comparison_prompt_template,
    system_message=None,
    with_reasoning=False,
    K=10
):
    """
    Evaluate model performance on holdout set.
    
    Args:
        graph: PreferenceGraph instance containing holdout edges
        agent: Agent instance for generating responses
        utility_model: UtilityModel instance for processing responses
        utilities: Dictionary of computed utilities
        comparison_prompt_template: Template for comparison prompts
        system_message: Optional system message for the agent
        with_reasoning: Whether to use reasoning-based response parsing
        K: Number of responses to generate per prompt
        
    Returns:
        Dictionary containing holdout metrics (or None if no holdout edges)
    """
    if not graph.holdout_edge_indices:
        print("Evaluating utility model on holdout set, but no holdout edges found; returning None.")
        return None
        
    # Generate prompts for holdout edges
    holdout_preference_data, holdout_prompts, holdout_prompt_idx_to_key = graph.generate_prompts(
        list(graph.holdout_edge_indices),
        comparison_prompt_template
    )
    
    # Generate responses for holdout edges
    holdout_responses = await generate_responses(
        agent=agent,
        prompts=holdout_prompts,
        system_message=system_message,
        K=K
    )
    
    # Parse responses and process them into preference data
    parsed_responses = parse_responses_forced_choice(holdout_responses, with_reasoning=with_reasoning)
    processed_preference_data = utility_model.process_responses(
        graph=graph,
        responses=holdout_responses,
        parsed_responses=parsed_responses,
        prompt_idx_to_key=holdout_prompt_idx_to_key
    )
    
    # Add edges to graph
    graph.add_edges(processed_preference_data)
    
    # Compute holdout metrics
    holdout_metrics = utility_model.evaluate(
        graph=graph,
        utilities=utilities,
        edge_indices=list(graph.holdout_edge_indices)
    )
    
    print("\nHoldout Set Metrics:")
    print(f"Log Loss: {holdout_metrics['log_loss']:.4f}")
    print(f"Accuracy: {holdout_metrics['accuracy'] * 100:.2f}%")
    
    return holdout_metrics

async def generate_responses_from_messages(
    agent: Union[HuggingFaceAgent, HuggingFaceAgentLogitsPrediction, vLLMAgent, vLLMAgentBaseModel],
    messages=None,
    timeout=5,
    verbose=True,
    structured_json: str = None,
):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        messages: List of messages to use for completions
        timeout: Timeout in seconds for each API call
        verbose: Whether to print verbose output

    Returns:
        A dictionary mapping prompt indices to their generated responses.
    """
    
    if isinstance(agent, HuggingFaceAgentLogitsPrediction):
        responses = agent.completions(messages)
    else:
        responses = agent.completions_batch(messages, structured_json=structured_json)
    
    if isinstance(responses, str):
        return [responses]
    return responses