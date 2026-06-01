"""Tiny factual benchmark for Phase-1 calibration (real questions + answer keys,
committed as data, no network). One-word/short factual answers so the
exact/substring correctness scorer applies cleanly. ~40 items to clear MIN_N=30.

This is a real known-answer set used ONLY to prove (or refuse) the disagreement
signal in Phase 1; it is not a mock — the models really answer these.
"""
from __future__ import annotations

TINY_FACTUAL = {
    "What is the capital of France? Answer in one word.": "Paris",
    "What is the capital of Japan? Answer in one word.": "Tokyo",
    "What is the capital of Italy? Answer in one word.": "Rome",
    "What is the capital of Spain? Answer in one word.": "Madrid",
    "What is the capital of Germany? Answer in one word.": "Berlin",
    "What is the capital of Russia? Answer in one word.": "Moscow",
    "What is the capital of Canada? Answer in one word.": "Ottawa",
    "What is the capital of Egypt? Answer in one word.": "Cairo",
    "What is the capital of Greece? Answer in one word.": "Athens",
    "What is the capital of Portugal? Answer in one word.": "Lisbon",
    "What is 2 plus 2? Answer with just the number.": "4",
    "What is 5 plus 7? Answer with just the number.": "12",
    "What is 9 minus 3? Answer with just the number.": "6",
    "What is 6 times 7? Answer with just the number.": "42",
    "What is 100 divided by 4? Answer with just the number.": "25",
    "What is 8 times 8? Answer with just the number.": "64",
    "What is 15 minus 9? Answer with just the number.": "6",
    "What is 3 times 9? Answer with just the number.": "27",
    "What is the chemical symbol for water? Answer in one token.": "H2O",
    "What is the chemical symbol for gold? Answer in one token.": "Au",
    "What is the chemical symbol for oxygen? Answer in one token.": "O",
    "What is the chemical symbol for sodium? Answer in one token.": "Na",
    "How many days are in a week? Just the number.": "7",
    "How many months are in a year? Just the number.": "12",
    "How many sides does a triangle have? Just the number.": "3",
    "How many continents are there? Just the number.": "7",
    "What color is a clear daytime sky? One word.": "blue",
    "What color is fresh snow? One word.": "white",
    "What color is grass? One word.": "green",
    "What color is a ripe banana? One word.": "yellow",
    "What is the largest planet in our solar system? One word.": "Jupiter",
    "What is the closest planet to the sun? One word.": "Mercury",
    "What planet is known as the red planet? One word.": "Mars",
    "What is the largest ocean on Earth? One word.": "Pacific",
    "What is the tallest mountain on Earth? One word.": "Everest",
    "What is the longest river in the world? One word.": "Nile",
    "Who wrote the play Romeo and Juliet? Last name only.": "Shakespeare",
    "What language is primarily spoken in Brazil? One word.": "Portuguese",
    "What is the freezing point of water in Celsius? Just the number.": "0",
    "What is the boiling point of water in Celsius? Just the number.": "100",
}
