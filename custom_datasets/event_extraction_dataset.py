import pandas as pd
from transformers import AutoTokenizer

from custom_datasets import combining_data


def convert_relation_extraction_df_to_event_extraction(df, tokenizer):
    examples = []
    events_in_text = {}
    documents_ids = {}
    for i, row in df.iterrows():
        text = row["text"]
        if text not in events_in_text:
            events_in_text[text] = set()
            documents_ids[text] = row["document_id"]
        events_in_text[text].add((row["event1_start"], row["event1_end"]))
        events_in_text[text].add((row["event2_start"], row["event2_end"]))
    for text in events_in_text:
        tokens = tokenizer.encode_plus(text, max_length=512)
        token_ids = tokens["input_ids"]
        labels = []
        for token_idx in range(len(token_ids)):
            chars = tokens.token_to_chars(token_idx)
            if chars is None:
                labels.append(False)
            else:
                # check if the token is part of any event
                start = chars.start
                end = chars.end
                events = events_in_text[text]
                is_event = False
                for event in events:
                    if event[0] <= start and event[1] >= end:
                        is_event = True
                labels.append(is_event)
        examples.append({"text": text, "labels": labels, "tokens": tokens, "document_id": documents_ids[text]})
    return pd.DataFrame(examples)

if __name__ == '__main__':
    df = combining_data.read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    tokenizer = AutoTokenizer.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
    convert_relation_extraction_df_to_event_extraction(df, tokenizer)