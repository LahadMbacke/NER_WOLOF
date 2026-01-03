"""
Test script to verify the trained model works correctly.
"""

import argparse
from gliner import GLiNER


# Test sentences in Wolof
TEST_SENTENCES = [
    "Ousmane Sonko jàngal na ci Université Cheikh Anta Diop ci Dakar.",
    "Macky Sall moo nekk président bu Senegaal.",
    "Sadio Mané dafa ligééy ci Liverpool.",
    "Demba ba dem na Thiès ci 15 janvier 2024.",
    "Assemblée Nationale bi nekk na ci Place Soweto.",
]

ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE"]


def test_model(model_path: str, threshold: float = 0.5):
    """
    Test the trained model on sample sentences.
    
    Args:
        model_path: Path to the model or Hugging Face model ID
        threshold: Confidence threshold for predictions
    """
    print("=" * 60)
    print("Testing GLiNER Wolof NER Model")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = GLiNER.from_pretrained(model_path)
    print("✅ Model loaded successfully!")
    
    # Test on each sentence
    print(f"\n{'=' * 60}")
    print("Running predictions...")
    print("=" * 60)
    
    total_entities = 0
    
    for i, text in enumerate(TEST_SENTENCES, 1):
        print(f"\n[{i}] {text}")
        print("-" * 50)
        
        entities = model.predict_entities(text, ENTITY_TYPES, threshold=threshold)
        
        if entities:
            for ent in entities:
                print(f"    → {ent['text']:20s} | {ent['label']:6s} | {ent['score']:.2%}")
                total_entities += 1
        else:
            print("    → No entities detected")
    
    print(f"\n{'=' * 60}")
    print(f"Total entities detected: {total_entities}")
    print("=" * 60)
    
    return total_entities > 0


def main():
    parser = argparse.ArgumentParser(description='Test GLiNER Wolof NER model')
    parser.add_argument(
        '--model-path',
        type=str,
        default='outputs/ner-wolof/final_model',
        help='Path to the model or Hugging Face model ID'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--from-hub',
        action='store_true',
        help='Load model from Hugging Face Hub'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='Lahad/gliner_wolof_NER',
        help='Hugging Face repository ID (used with --from-hub)'
    )
    
    args = parser.parse_args()
    
    model_path = args.repo_id if args.from_hub else args.model_path
    
    success = test_model(model_path, args.threshold)
    
    if success:
        print("\n✅ Model test passed!")
    else:
        print("\n⚠️ No entities detected - model may need more training")


if __name__ == "__main__":
    main()
