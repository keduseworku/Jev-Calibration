"""The two questions sent with every example (one request, two questions)."""
from typesafe_sdk import Choice, Noul

QUESTIONS = {
    "positive": Noul(instructions="Is the overall sentiment of this text positive?"),
    "sentiment": Choice(
        instructions="What is the overall sentiment of this text?",
        criteria={"positive": None, "negative": None},
    ),
}
