#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Convert HB outputs to IFEval format")
    parser.add_argument("input_path", help="Path to input jsonl file")
    parser.add_argument("output_path", help="Path to output jsonl file")
    args = parser.parse_args()

    with open(args.input_path, "r") as f_in, open(args.output_path, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            output_row = {
                "prompt": row["prompt"][0]["content"],
                "response": row["response"],
            }
            f_out.write(json.dumps(output_row) + "\n")


if __name__ == "__main__":
    main()
