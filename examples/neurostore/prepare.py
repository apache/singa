"""
prepare.py

This script downloads models from Hugging Face and exports them to ONNX format.
In our example, we use three DistilBERT models:
1. google-bert/bert-base-uncased
2. lorenzozan/ME2-BERT
3. prithivMLmods/Spam-Bert-Uncased

To download and export other models, modify: 1). the `models` list in the main function. 2). the
dummy_inputs and torch.onnx.export function parameters.
"""

import os
import torch
from transformers import AutoModel

DUMMY_SEQ_LENGTH = 32


def export_model_to_onnx(model_name: str, save_dir: str):
    """
    Downloads and exports a Hugging Face model to ONNX format.
    """
    print(f"Processing {model_name}...")
    try:
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        dummy_inputs = {
            "input_ids": torch.randint(0, 30522, (1, DUMMY_SEQ_LENGTH)),
            "attention_mask": torch.ones((1, DUMMY_SEQ_LENGTH), dtype=torch.long)
        }

        safe_model_name = model_name.replace("/", "_")
        export_path = os.path.join(save_dir, f"{safe_model_name}.onnx")

        torch.onnx.export(
            model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            export_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14,
        )
        print(f"Exported ONNX to {export_path}")
    except Exception as e:
        print(f"Failed to export {model_name}: {e}")


if __name__ == "__main__":
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    models = [
        "distilbert/distilbert-base-uncased",
        "winegarj/distilbert-base-uncased-finetuned-sst2",
        "eliasalbouzidi/distilbert-nsfw-text-classifier"
    ]

    for model_name in models:
        export_model_to_onnx(model_name, model_dir)
