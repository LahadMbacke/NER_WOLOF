"""
GLiNER Fine-tuning Script for Wolof NER
Compatible with GLiNER 0.2.x API
"""

import os
import yaml
import argparse
from datasets import load_dataset
from torch.utils.data import Dataset
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing import UniEncoderSpanDataCollator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def convert_to_gliner_format(examples, label_mapping):
    """
    Convert MasakhaNER format to GLiNER format.
    
    GLiNER expects:
    {
        "tokenized_text": ["word1", "word2", ...],
        "ner": [[start_idx, end_idx, "LABEL"], ...]
    }
    """
    gliner_data = []
    
    for tokens, ner_tags in zip(examples['tokens'], examples['ner_tags']):
        entities = []
        current_entity = None
        
        for idx, tag in enumerate(ner_tags):
            if tag == 0:  # O tag
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
            else:
                label = label_mapping[tag]
                # Check if it's a B- tag (odd numbers in BIO scheme)
                if tag % 2 == 1:  # B- tag
                    if current_entity is not None:
                        entities.append(current_entity)
                    entity_type = label.replace('B-', '')
                    current_entity = [idx, idx, entity_type]
                else:  # I- tag
                    if current_entity is not None:
                        current_entity[1] = idx  # Extend entity
        
        # Don't forget the last entity
        if current_entity is not None:
            entities.append(current_entity)
        
        gliner_data.append({
            "tokenized_text": tokens,
            "ner": entities
        })
    
    return gliner_data


class GLiNERDataset(Dataset):
    """
    Custom Dataset for GLiNER training.
    Returns raw data that will be processed by the collator.
    """
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main(config_path: str):
    """Main training function."""
    
    # Load configuration
    config = load_config(config_path)
    
    print("=" * 60)
    print("GLiNER Fine-tuning for Wolof NER")
    print("=" * 60)
    
    # 1. Load MasakhaNER dataset
    print("\n[1/6] Loading MasakhaNER Wolof dataset...")
    dataset = load_dataset('masakhaner', 'wol')
    print(f"  - Train samples: {len(dataset['train'])}")
    print(f"  - Validation samples: {len(dataset['validation'])}")
    print(f"  - Test samples: {len(dataset['test'])}")
    
    # 2. Get label mapping
    print("\n[2/6] Creating label mapping...")
    label_names = dataset['train'].features['ner_tags'].feature.names
    label_mapping = {i: name for i, name in enumerate(label_names)}
    
    # Extract entity types (without B-/I- prefix)
    entity_types = list(set([
        name.replace('B-', '').replace('I-', '') 
        for name in label_names if name != 'O'
    ]))
    print(f"  - Entity types: {entity_types}")
    
    # 3. Convert data to GLiNER format
    print("\n[3/6] Converting data to GLiNER format...")
    train_gliner = convert_to_gliner_format(dataset['train'], label_mapping)
    val_gliner = convert_to_gliner_format(dataset['validation'], label_mapping)
    
    print(f"  - Converted {len(train_gliner)} training samples")
    print(f"  - Converted {len(val_gliner)} validation samples")
    
    # Show example
    print("\n  Example converted data:")
    example = train_gliner[0]
    print(f"    Tokens: {example['tokenized_text'][:10]}...")
    print(f"    Entities: {example['ner']}")
    
    # 4. Load GLiNER model
    print("\n[4/6] Loading GLiNER model...")
    model_name = config['model']['name']
    print(f"  - Model: {model_name}")
    
    model = GLiNER.from_pretrained(model_name)
    print("  - Model loaded successfully!")
    
    # 5. Prepare datasets and collator
    print("\n[5/6] Preparing datasets and data collator...")
    
    train_dataset = GLiNERDataset(train_gliner)
    val_dataset = GLiNERDataset(val_gliner)
    
    data_collator = UniEncoderSpanDataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )
    
    print(f"  - Train dataset: {len(train_dataset)} samples")
    print(f"  - Validation dataset: {len(val_dataset)} samples")
    
    # 6. Configure training
    print("\n[6/6] Configuring training...")
    
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=float(config['training']['warmup_ratio']),
        num_train_epochs=int(config['training']['num_epochs']),
        per_device_train_batch_size=int(config['training']['batch_size']),
        per_device_eval_batch_size=int(config['training']['batch_size']),
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=int(config['training']['eval_steps']),
        save_steps=int(config['training']['save_steps']),
        save_total_limit=int(config['training']['save_total_limit']),
        dataloader_num_workers=int(config['training']['num_workers']),
        use_cpu=config['training'].get('use_cpu', False),
        focal_loss_alpha=float(config['training'].get('focal_loss_alpha', 0.75)),
        focal_loss_gamma=int(config['training'].get('focal_loss_gamma', 2)),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    print(f"  - Output directory: {output_dir}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    
    # 7. Initialize Trainer
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    
    # 8. Train
    trainer.train()
    
    # 9. Save the model
    print("\n" + "=" * 60)
    print("Training complete! Saving model...")
    print("=" * 60)
    
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    print(f"  - Model saved to: {final_model_path}")
    
    # 10. Quick test
    print("\n" + "=" * 60)
    print("Quick inference test...")
    print("=" * 60)
    
    test_text = "Ousmane Sonko jàngal na ci Université Cheikh Anta Diop ci Dakar."
    test_labels = entity_types
    
    predictions = model.predict_entities(test_text, test_labels)
    print(f"  Text: {test_text}")
    print(f"  Predictions: {predictions}")
    
    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune GLiNER for Wolof NER')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    main(args.config)
