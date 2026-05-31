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


# A stream with REPEATED distinction-regions, so the experience graph fills and
# replay (amortized audit) actually engages. Many of these share a probe key
# (short factual), which is the whole point: the first hit in a region audits;
# later hits in the same region replay the cheap route without re-auditing.
REPEATED_REGION_STREAM = [
    "What is the capital of France? One word.",
    "What is the capital of Japan? One word.",
    "What is the capital of Italy? One word.",
    "What is the capital of Spain? One word.",
    "What is the capital of Germany? One word.",
    "What is 2 plus 2? Just the number.",
    "What is 3 plus 5? Just the number.",
    "What is 7 minus 4? Just the number.",
    "What is 6 times 2? Just the number.",
    "What is 10 divided by 2? Just the number.",
    "Name the largest planet. One word.",
    "Name the closest planet to the sun. One word.",
    "Name the red planet. One word.",
    "What color is grass? One word.",
    "What color is the sun? One word.",
    "What color is snow? One word.",
]

