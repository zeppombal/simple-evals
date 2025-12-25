#!/usr/bin/env python3
"""
Convert IFEval raw.jsonl to hb_short.jsonl format.

Usage:
    python convert_to_hb_format.py [--input INPUT] [--output OUTPUT]
"""

import argparse
import json
import uuid
from pathlib import Path

# Language code to language name mapping
LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "pl": "Polish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "sw": "Swahili",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "zu": "Zulu",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "el": "Greek",
    "fa": "Persian",
    "am": "Amharic",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "ka": "Georgian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "uz": "Uzbek",
    "mn": "Mongolian",
}

# All known instruction types
KNOWN_INSTRUCTION_IDS = {
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
    "combination:two_responses",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:title",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "startend:end_checker",
    "startend:quotation",
}


def instruction_to_rubric(instruction_id: str, kwargs: dict) -> str:
    """
    Convert an IFEval instruction_id and its kwargs to a human-readable rubric criterion.

    Raises:
        ValueError: If instruction_id is not recognized.
    """
    if instruction_id not in KNOWN_INSTRUCTION_IDS:
        raise ValueError(f"Unknown instruction_id: {instruction_id}")

    # Helper to safely get numeric values
    def get_int(key):
        val = kwargs.get(key)
        return int(val) if val is not None else None

    def get_str(key):
        return kwargs.get(key)

    def get_list(key):
        return kwargs.get(key, [])

    # Map instruction_id to rubric criterion
    if instruction_id == "change_case:capital_word_frequency":
        relation = get_str("capital_relation") or "at least"
        frequency = get_int("capital_frequency") or 1
        return (
            f"Response contains {relation} {frequency} word(s) in all capital letters"
        )

    elif instruction_id == "change_case:english_capital":
        return "Response is entirely in capital letters (no lowercase letters)"

    elif instruction_id == "change_case:english_lowercase":
        return "Response is entirely in lowercase letters (no capital letters)"

    elif instruction_id == "combination:repeat_prompt":
        return (
            "Response starts by repeating the prompt exactly before giving the answer"
        )

    elif instruction_id == "combination:two_responses":
        return "Response contains exactly two different responses separated by six asterisks (******)"

    elif instruction_id == "detectable_content:number_placeholders":
        num_placeholders = get_int("num_placeholders") or 1
        return f"Response contains at least {num_placeholders} placeholder(s) in square brackets (e.g., [name], [address])"

    elif instruction_id == "detectable_content:postscript":
        postscript_marker = get_str("postscript_marker") or "P.S."
        return f"Response ends with a postscript starting with '{postscript_marker}'"

    elif instruction_id == "detectable_format:constrained_response":
        return "Response is one of the constrained valid responses (e.g., 'My answer is yes.', 'My answer is no.')"

    elif instruction_id == "detectable_format:json_format":
        return "Response is in valid JSON format"

    elif instruction_id == "detectable_format:multiple_sections":
        section_splitter = get_str("section_spliter") or "SECTION"
        num_sections = get_int("num_sections") or 1
        return f"Response contains exactly {num_sections} section(s) marked with '{section_splitter}'"

    elif instruction_id == "detectable_format:number_bullet_lists":
        num_bullets = get_int("num_bullets") or 1
        return f"Response contains exactly {num_bullets} bullet point(s) using '* ' markdown format"

    elif instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = get_int("num_highlights") or 1
        return f"Response highlights at least {num_highlights} section(s) using *markdown italics* (e.g., *highlighted section*)"

    elif instruction_id == "detectable_format:title":
        return "Response contains a title wrapped in double angular brackets (e.g., <<title>>)"

    elif instruction_id == "keywords:existence":
        keywords = get_list("keywords")
        if keywords:
            keywords_str = ", ".join(f'"{k}"' for k in keywords)
            return f"Response contains all of the following keywords: {keywords_str}"
        return "Response contains the required keywords"

    elif instruction_id == "keywords:forbidden_words":
        forbidden_words = get_list("forbidden_words")
        if forbidden_words:
            words_str = ", ".join(f'"{w}"' for w in forbidden_words)
            return f"Response does not contain any of the following forbidden words: {words_str}"
        return "Response does not contain the forbidden words"

    elif instruction_id == "keywords:frequency":
        keyword = get_str("keyword") or "keyword"
        relation = get_str("relation") or "at least"
        frequency = get_int("frequency") or 1
        return f'Response contains the word "{keyword}" {relation} {frequency} time(s)'

    elif instruction_id == "keywords:letter_frequency":
        letter = get_str("letter") or "a"
        let_relation = get_str("let_relation") or "at least"
        let_frequency = get_int("let_frequency") or 1
        return f"Response contains the character '{letter}' {let_relation} {let_frequency} time(s)"

    elif instruction_id == "language:response_language":
        lang_code = get_str("language") or "en"
        lang_name = LANGUAGE_CODES.get(lang_code, lang_code)
        return f"Response is entirely in {lang_name}"

    elif instruction_id == "length_constraints:nth_paragraph_first_word":
        nth_paragraph = get_int("nth_paragraph") or 1
        first_word = get_str("first_word") or "The"
        return f'Paragraph {nth_paragraph} starts with the word "{first_word}"'

    elif instruction_id == "length_constraints:number_paragraphs":
        num_paragraphs = get_int("num_paragraphs") or 1
        return f"Response contains exactly {num_paragraphs} paragraph(s) separated by '***'"

    elif instruction_id == "length_constraints:number_sentences":
        relation = get_str("relation") or "at least"
        num_sentences = get_int("num_sentences") or 1
        return f"Response contains {relation} {num_sentences} sentence(s)"

    elif instruction_id == "length_constraints:number_words":
        relation = get_str("relation") or "at least"
        num_words = get_int("num_words") or 1
        return f"Response contains {relation} {num_words} word(s)"

    elif instruction_id == "punctuation:no_comma":
        return "Response does not contain any commas"

    elif instruction_id == "startend:end_checker":
        end_phrase = get_str("end_phrase") or ""
        return f'Response ends with the exact phrase: "{end_phrase}"'

    elif instruction_id == "startend:quotation":
        return "Response is wrapped in double quotation marks"


def convert_entry(entry: dict) -> dict:
    """
    Convert a single IFEval entry to hb_short.jsonl format.
    """
    # Generate deterministic UUID from the key
    prompt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ifeval-{entry['key']}"))

    # Build rubrics from instruction_id_list and kwargs
    rubrics = []
    for instruction_id, kwargs in zip(entry["instruction_id_list"], entry["kwargs"]):
        criterion = instruction_to_rubric(instruction_id, kwargs)
        rubrics.append({"criterion": criterion, "points": 1, "tags": []})

    return {
        "example_tags": [],
        "ideal_completions_data": None,
        "prompt": [{"content": entry["prompt"], "role": "user"}],
        "prompt_id": prompt_id,
        "rubrics": rubrics,
        "canary": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert IFEval raw.jsonl to hb_short.jsonl format"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=Path(__file__).parent / "raw.jsonl",
        type=Path,
        help="Input raw.jsonl file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=Path(__file__).parent / "converted.jsonl",
        type=Path,
        help="Output converted.jsonl file path",
    )
    args = parser.parse_args()

    # Read input
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {args.input}")

    # Convert entries
    converted = []
    for entry in entries:
        converted.append(convert_entry(entry))

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(converted)} entries to {args.output}")

    # Print summary of rubric types
    rubric_counts = {}
    for entry in entries:
        for inst_id in entry["instruction_id_list"]:
            rubric_counts[inst_id] = rubric_counts.get(inst_id, 0) + 1

    print(f"\nInstruction type distribution:")
    for inst_id, count in sorted(rubric_counts.items(), key=lambda x: -x[1]):
        print(f"  {inst_id}: {count}")


if __name__ == "__main__":
    main()
