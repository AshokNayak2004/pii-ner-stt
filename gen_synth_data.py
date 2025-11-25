import json
import random
import os
from typing import List, Dict, Tuple

random.seed(42)

PII_LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"]
NON_PII_LABELS = ["CITY", "LOCATION"]

DIGIT_WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine"]
MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]
FIRST_NAMES = ["rahul","ashok","neha","anita","vivek","rohan","priya","sneha","arjun","kiran"]
LAST_NAMES = ["sharma","patel","naik","iyer","mehta","gupta","rao","joshi","khan"]
CITIES = ["mumbai","pune","delhi","bangalore","chennai","kolkata","hyderabad","ahmedabad"]
LOCATIONS = ["central mall","main street","sector seventeen","park street","marine drive","tech park"]
EMAIL_PROVIDERS = ["gmail","yahoo","outlook","hotmail","protonmail"]
FILLERS = ["uh","like","you know","actually","basically","i mean","sort of","kind of"]
RANDOM_WORDS = ["today","meeting","booking","payment","ticket","hotel","office",
                "friend","flight","order","account","number","details","info","help"]

def random_filler():
    if random.random() < 0.4:
        return [random.choice(FILLERS)]
    return []

def random_words(n: int):
    return [random.choice(RANDOM_WORDS) for _ in range(n)]

def gen_phone_tokens() -> List[str]:
    # e.g. "nine eight seven six five four three two one zero"
    length = random.choice([10, 11])
    digits = [random.choice(DIGIT_WORDS) for _ in range(length)]
    return digits

def gen_credit_card_tokens() -> List[str]:
    # 16-digit card spoken in groups of 4
    digits = [random.choice(DIGIT_WORDS) for _ in range(16)]
    # optionally insert "space" grouping with pauses (just more spaces between groups)
    return digits

def gen_email_tokens() -> List[str]:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    provider = random.choice(EMAIL_PROVIDERS)
    # "ashok dot sharma at gmail dot com"
    tokens = [first, "dot", last, "at", provider, "dot", "com"]
    # add a tiny chance of noise "gmeil"
    if random.random() < 0.2 and provider == "gmail":
        tokens[4] = "gmeil"
    return tokens

def gen_name_tokens() -> List[str]:
    return [random.choice(FIRST_NAMES), random.choice(LAST_NAMES)]

def gen_date_tokens() -> List[str]:
    # e.g. "twenty three january twenty twenty four"
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    year = random.randint(2018, 2025)
    # simple day words:
    day_words = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth", 21: "twenty first", 22: "twenty second",
        23: "twenty third", 24: "twenty fourth", 25: "twenty fifth",
        26: "twenty sixth", 27: "twenty seventh", 28: "twenty eighth"
    }
    day_tokens = day_words[day].split()
    year_tokens = ["twenty", str(year % 100)]  # "twenty 24"
    return day_tokens + [month] + year_tokens

def gen_city_tokens() -> List[str]:
    return [random.choice(CITIES)]

def gen_location_tokens() -> List[str]:
    return random.choice(LOCATIONS).split()

def build_utterance(entities_to_include: List[str]) -> Tuple[str, List[Dict]]:
    """
    Build a single noisy STT utterance containing some entities.
    Returns (text, entities) where entities have char offsets.
    """
    segments = []  # each segment: ("text", [tokens]) or ("ent", label, [tokens])
    # random order of appearance
    random.shuffle(entities_to_include)

    # start with some random words
    segments.append(("text", random_words(random.randint(2, 5)) + random_filler()))

    for label in entities_to_include:
        # random filler before each entity
        segments.append(("text", random_filler() + random_words(random.randint(1, 3))))

        if label == "PHONE":
            tokens = gen_phone_tokens()
        elif label == "CREDIT_CARD":
            tokens = gen_credit_card_tokens()
        elif label == "EMAIL":
            tokens = gen_email_tokens()
        elif label == "PERSON_NAME":
            tokens = gen_name_tokens()
        elif label == "DATE":
            tokens = gen_date_tokens()
        elif label == "CITY":
            tokens = gen_city_tokens()
        elif label == "LOCATION":
            tokens = gen_location_tokens()
        else:
            tokens = ["unknown"]

        segments.append(("ent", label, tokens))

    # end with some random words
    segments.append(("text", random_filler() + random_words(random.randint(1, 4))))

    # Now build final text + entity spans
    text = ""
    spans = []
    for seg in segments:
        if seg[0] == "text":
            tokens = seg[1]
            for tok in tokens:
                if not tok:
                    continue
                if text:
                    text += " "
                text += tok
        else:
            _, label, tokens = seg
            # entity start
            if tokens:
                if text:
                    text += " "
                ent_start = len(text)
                for i, tok in enumerate(tokens):
                    if i > 0:
                        text += " "
                    text += tok
                ent_end = len(text)
                spans.append({"start": ent_start, "end": ent_end, "label": label})

    return text, spans

def sample_entity_combo(split: str) -> List[str]:
    """
    Decide which entities to include in a given example.
    We ensure at least one PII entity per utterance.
    Stress set tends to include more entities per utterance.
    """
    entities = []

    # always at least one PII
    main_pii = random.choice(PII_LABELS)
    entities.append(main_pii)

    # sometimes add another PII
    if random.random() < (0.4 if split != "stress" else 0.7):
        extra = random.choice(PII_LABELS)
        if extra not in entities:
            entities.append(extra)

    # sometimes add CITY / LOCATION
    if random.random() < 0.5:
        entities.append(random.choice(NON_PII_LABELS))

    return entities

def generate_split(n: int, split: str) -> List[Dict]:
    data = []
    for i in range(n):
        entities_to_include = sample_entity_combo(split)
        text, spans = build_utterance(entities_to_include)
        ex = {
            "id": f"{split}_{i:05d}",
            "text": text,
            "entities": spans
        }
        data.append(ex)
    return data

def write_jsonl(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    os.makedirs("data", exist_ok=True)

    train_data = generate_split(800, "train")
    dev_data = generate_split(150, "dev")
    stress_data = generate_split(100, "stress")

    write_jsonl("data/train.jsonl", train_data)
    write_jsonl("data/dev.jsonl", dev_data)
    write_jsonl("data/stress.jsonl", stress_data)

    print("Wrote:")
    print("  data/train.jsonl  (800 examples)")
    print("  data/dev.jsonl    (150 examples)")
    print("  data/stress.jsonl (100 examples)")

if __name__ == "__main__":
    main()
