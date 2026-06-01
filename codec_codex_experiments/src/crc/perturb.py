"""Arm-2 perturbation: counterfactual question editing to isolate memorized
recall from reasoning-on-input (CRC higher-concept test).

A perturbed item embeds a NEW premise in the prompt so the *memorized* answer is
provably wrong and the correct answer follows only from reading the given premise.
A model that parrots the memorized answer is provably NOT reasoning on the input.

Each PerturbedItem carries:
- prompt: the counterfactual question (premise stated in-prompt)
- answer: the correct answer GIVEN THE PREMISE (the new key; committed data,
  human-verifiable — NOT model-generated)
- memorized_trap: the old memorized answer that a non-reasoning model would emit
The pairs are hand-authored data, not generated, so the key is ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PerturbedItem:
    prompt: str
    answer: str          # correct given the in-prompt premise (the new key)
    memorized_trap: str  # the memorized answer a non-reasoning model parrots
    base_fact: str       # the original fact being perturbed (for provenance)


# Hand-authored counterfactual items. The premise is stated IN the prompt, so the
# correct answer is unambiguous from reading it; the memorized_trap is what a model
# emits if it ignores the premise and recalls the real-world fact.
PERTURBED_ITEMS: List[PerturbedItem] = [
    PerturbedItem(
        prompt="In my story, the capital of France is Brindle. "
               "Per my story, what is the capital of France? One word.",
        answer="Brindle", memorized_trap="Paris", base_fact="capital of France = Paris"),
    PerturbedItem(
        prompt="Suppose 2 plus 2 equals 5 in this puzzle's rules. "
               "Under these rules, what does 2 plus 2 equal? Just the number.",
        answer="5", memorized_trap="4", base_fact="2+2=4"),
    PerturbedItem(
        prompt="In this fictional solar system, the largest planet is named Vorn. "
               "What is the largest planet here? One word.",
        answer="Vorn", memorized_trap="Jupiter", base_fact="largest planet = Jupiter"),
    PerturbedItem(
        prompt="In my alternate chemistry, the symbol for water is Wx. "
               "What is the symbol for water in my alternate chemistry? One token.",
        answer="Wx", memorized_trap="H2O", base_fact="water = H2O"),
    PerturbedItem(
        prompt="In this calendar, a week has 9 days. "
               "How many days are in a week in this calendar? Just the number.",
        answer="9", memorized_trap="7", base_fact="week = 7 days"),
    PerturbedItem(
        prompt="In my story grass is purple. Per my story, what color is grass? One word.",
        answer="purple", memorized_trap="green", base_fact="grass = green"),
    PerturbedItem(
        prompt="In this fictional world, the closest planet to the sun is Kael. "
               "What is the closest planet to the sun here? One word.",
        answer="Kael", memorized_trap="Mercury", base_fact="closest planet = Mercury"),
    PerturbedItem(
        prompt="Suppose the boiling point of water is 80 degrees Celsius in this "
               "scenario. What is water's boiling point here? Just the number.",
        answer="80", memorized_trap="100", base_fact="boiling point = 100C"),
    PerturbedItem(
        prompt="In my novel, the play Romeo and Juliet was written by Carmody. "
               "Per my novel, who wrote it? Last name only.",
        answer="Carmody", memorized_trap="Shakespeare", base_fact="author = Shakespeare"),
    PerturbedItem(
        prompt="In this fictional geography, the largest ocean is the Verdant Ocean. "
               "What is the largest ocean here? One word.",
        answer="Verdant", memorized_trap="Pacific", base_fact="largest ocean = Pacific"),
    PerturbedItem(
        prompt="In my story the chemical symbol for gold is Gx. "
               "Per my story, what is the symbol for gold? One token.",
        answer="Gx", memorized_trap="Au", base_fact="gold = Au"),
    PerturbedItem(
        prompt="Suppose a triangle has 5 sides in this puzzle. "
               "How many sides does a triangle have here? Just the number.",
        answer="5", memorized_trap="3", base_fact="triangle = 3 sides"),
    PerturbedItem(
        prompt="In this fictional country, the capital of Japan is Suno. "
               "What is the capital of Japan here? One word.",
        answer="Suno", memorized_trap="Tokyo", base_fact="capital of Japan = Tokyo"),
    PerturbedItem(
        prompt="In my alternate world, the red planet is called Ember. "
               "What planet is known as the red planet here? One word.",
        answer="Ember", memorized_trap="Mars", base_fact="red planet = Mars"),
    PerturbedItem(
        prompt="In this scenario, the freezing point of water is 10 degrees Celsius. "
               "What is water's freezing point here? Just the number.",
        answer="10", memorized_trap="0", base_fact="freezing point = 0C"),
    PerturbedItem(
        prompt="In my story the sky is described as orange. "
               "Per my story, what color is the clear daytime sky? One word.",
        answer="orange", memorized_trap="blue", base_fact="sky = blue"),
    PerturbedItem(
        prompt="Suppose 6 times 7 equals 50 in this game's math. "
               "Under these rules, what is 6 times 7? Just the number.",
        answer="50", memorized_trap="42", base_fact="6x7=42"),
    PerturbedItem(
        prompt="In this fictional world, the tallest mountain is Mount Cirra. "
               "What is the tallest mountain here? One word.",
        answer="Cirra", memorized_trap="Everest", base_fact="tallest mountain = Everest"),
    PerturbedItem(
        prompt="In my alternate chemistry, oxygen's symbol is Qo. "
               "What is oxygen's symbol in my alternate chemistry? One token.",
        answer="Qo", memorized_trap="O", base_fact="oxygen = O"),
    PerturbedItem(
        prompt="In this calendar, a year has 15 months. "
               "How many months are in a year in this calendar? Just the number.",
        answer="15", memorized_trap="12", base_fact="year = 12 months"),
    PerturbedItem(
        prompt="In my story, the capital of Italy is Doria. "
               "Per my story, what is the capital of Italy? One word.",
        answer="Doria", memorized_trap="Rome", base_fact="capital of Italy = Rome"),
    PerturbedItem(
        prompt="In this fiction, the capital of Spain is Estra. "
               "What is the capital of Spain here? One word.",
        answer="Estra", memorized_trap="Madrid", base_fact="capital of Spain = Madrid"),
    PerturbedItem(
        prompt="In my alternate world, the capital of Germany is Halden. "
               "Per my world, what is the capital of Germany? One word.",
        answer="Halden", memorized_trap="Berlin", base_fact="capital of Germany = Berlin"),
    PerturbedItem(
        prompt="Suppose 5 plus 7 equals 20 in this puzzle. "
               "Under these rules, what is 5 plus 7? Just the number.",
        answer="20", memorized_trap="12", base_fact="5+7=12"),
    PerturbedItem(
        prompt="Suppose 9 minus 3 equals 1 in this game. "
               "Under these rules, what is 9 minus 3? Just the number.",
        answer="1", memorized_trap="6", base_fact="9-3=6"),
    PerturbedItem(
        prompt="Suppose 8 times 8 equals 70 in this scenario. "
               "Under these rules, what is 8 times 8? Just the number.",
        answer="70", memorized_trap="64", base_fact="8x8=64"),
    PerturbedItem(
        prompt="In my alternate chemistry, sodium's symbol is Sx. "
               "What is sodium's symbol in my alternate chemistry? One token.",
        answer="Sx", memorized_trap="Na", base_fact="sodium = Na"),
    PerturbedItem(
        prompt="In my story, a ripe banana is blue. "
               "Per my story, what color is a ripe banana? One word.",
        answer="blue", memorized_trap="yellow", base_fact="banana = yellow"),
    PerturbedItem(
        prompt="In my story, fresh snow is black. "
               "Per my story, what color is fresh snow? One word.",
        answer="black", memorized_trap="white", base_fact="snow = white"),
    PerturbedItem(
        prompt="In this fictional country, the primary language of Brazil is Korlan. "
               "What language is primarily spoken in Brazil here? One word.",
        answer="Korlan", memorized_trap="Portuguese", base_fact="Brazil = Portuguese"),
    PerturbedItem(
        prompt="In my alternate world, the longest river is the Tanis. "
               "What is the longest river here? One word.",
        answer="Tanis", memorized_trap="Nile", base_fact="longest river = Nile"),
]


def perturbed_benchmark() -> Dict[str, str]:
    """Return {prompt: correct_answer_given_premise} — the Arm-2 answer key."""
    return {it.prompt: it.answer for it in PERTURBED_ITEMS}


def memorized_traps() -> Dict[str, str]:
    """Return {prompt: memorized_answer} — what a non-reasoning model parrots."""
    return {it.prompt: it.memorized_trap for it in PERTURBED_ITEMS}
