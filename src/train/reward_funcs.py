import re

try:
    from math_verify import LatexExtractionConfig, parse, verify
    from latex2sympy2_extended import NormalizationConfig
except ImportError:
    LatexExtractionConfig = None
    NormalizationConfig = None
    parse = None
    verify = None

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    rewards = []

    for completion, sol in zip(completions, assistant):
        if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
            rewards.append(float(completion.strip().lower() == sol.strip().lower()))
            continue

        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards
