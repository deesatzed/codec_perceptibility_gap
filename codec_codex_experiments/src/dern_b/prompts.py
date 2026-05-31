"""Fixed local prompt set for Stage-B measurement (no network, committed data).

A mix of short-factual prompts (where the cheap model often agrees with the
reference -> cheap served) and longer-reasoning prompts (where it more often
escalates). Deliberately small to keep a real run tractable on a laptop.
"""

FACTUAL = [
    "What is the capital of France? Answer in one word.",
    "What is 2 plus 2? Answer with just the number.",
    "Name the largest planet in our solar system. One word.",
    "What color is a clear daytime sky? One word.",
    "How many days are in a week? Just the number.",
    "What is the chemical symbol for water? One token.",
]

REASONING = [
    "Explain step by step why dividing by zero is undefined.",
    "Reason carefully: if a train travels 60 km in 1.5 hours, what is its average speed, and why?",
    "Explain in detail how a binary search algorithm achieves logarithmic time.",
]

ALL_PROMPTS = FACTUAL + REASONING
