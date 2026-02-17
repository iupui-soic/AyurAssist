def doc_to_text(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <option1>
    B. <option2>
    C. <option3>
    D. <option4>
    Answer:
    """
    choices = [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]
    option_choices = {
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"

    return prompt


def doc_to_target(doc) -> int:
    """
    Returns the index of the correct answer in the list of choices
    """
    answer_text = doc["correct_answer"]
    options = ["A", "B", "C", "D"]
    option_key = options.index(answer_text)
    return option_key


def doc_to_text_genai(doc) -> str:
    """
    Prompt for generate_until tasks that explicitly requests only the answer letter.
    """
    choices = [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]
    option_choices = {
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer with ONLY the letter (A, B, C, or D):"

    return prompt


def doc_to_target_letter(doc) -> str:
    """
    Returns the correct answer letter (A, B, C, or D) for generate_until tasks.
    """
    return doc["correct_answer"]


import re


def process_results_genai(doc, results):
    """
    Custom result processor that handles both letter-based and text-based answers.
    First tries to extract a letter (A-D), then falls back to text matching
    against the option values.
    """
    response = results[0].strip()
    target = doc["correct_answer"]  # "A", "B", "C", or "D"

    options = {
        "A": doc["option_a"].strip(),
        "B": doc["option_b"].strip(),
        "C": doc["option_c"].strip(),
        "D": doc["option_d"].strip(),
    }

    # Try 1: Extract a standalone letter (last match, handles thinking models)
    letter_match = re.findall(r"\b([A-Da-d])\b", response)
    if letter_match:
        predicted = letter_match[-1].upper()
        return {"exact_match": 1.0 if predicted == target else 0.0}

    # Try 2: Match generated text against option values (case-insensitive)
    response_lower = response.lower().strip().rstrip(".")
    for letter, option_text in options.items():
        if response_lower == option_text.lower().strip():
            return {"exact_match": 1.0 if letter == target else 0.0}

    # Try 3: Check if response starts with or contains an option value
    for letter, option_text in options.items():
        if option_text.lower().strip() in response_lower:
            return {"exact_match": 1.0 if letter == target else 0.0}

    return {"exact_match": 0.0}
