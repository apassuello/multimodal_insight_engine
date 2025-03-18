import torch
from PIL import Image
import os
import argparse

def multimodal_chat_demo(model, tokenizer, safety_evaluator=None):
    """
    Simple demo application for multimodal interaction with text and images.
    
    Args:
        model: Multimodal model capable of processing text and images
        tokenizer: Tokenizer for text processing
        safety_evaluator: Optional safety evaluator for content checks
    """
    print("\n===== Multimodal Chat Demo =====")
    print("Upload an image and ask questions about it. Type 'exit' to quit.\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    while True:
        # Get image path from user
        image_path = input("\nEnter path to image (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break
            
        # Validate image path
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found.")
            continue
            
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"Image loaded: {image_path} ({image.width}x{image.height})")
            
            # User interaction loop for this image
            while True:
                # Get user query
                query = input("\nYour question about the image (or 'new' for different image): ")
                if query.lower() == 'exit':
                    return
                if query.lower() == 'new':
                    break
                    
                # Safety check if safety evaluator is provided
                if safety_evaluator is not None:
                    safety_result = safety_evaluator.evaluate_multimodal_content(query, image)
                    
                    if safety_result["flagged"]:
                        print("\nI'm sorry, but your request couldn't be processed due to safety concerns.")
                        print(f"Flagged categories: {safety_result['flagged_categories']}")
                        continue
                
                # Generate response using the model
                try:
                    print("\nGenerating response...")
                    response = generate_caption_with_beam_search(
                        model=model,
                        image=image,
                        tokenizer=tokenizer,
                        prompt=query + " ",  # Add space after query
                        beam_size=3,
                        device=device
                    )
                    
                    # Safety check on output
                    if safety_evaluator is not None:
                        output_safety = safety_evaluator.text_safety_evaluator.evaluate_text(response)
                        
                        if output_safety["flagged"]:
                            print("I apologize, but I couldn't generate an appropriate response.")
                            continue
                    
                    # Display response
                    print("\nResponse:", response)
                    
                except Exception as e:
                    print(f"\nError generating response: {str(e)}")
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            
    print("\nThank you for using the Multimodal Chat Demo!")

def main():
    """Command-line entry point for the demo application"""
    parser = argparse.ArgumentParser(description="Multimodal Chat Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--safety", action="store_true", help="Enable safety evaluation")
    args = parser.parse_args()
    
    # Load model, tokenizer, and safety evaluator
    # Note: This would need to be implemented according to your specific model architecture
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    safety_evaluator = None
    if args.safety:
        safety_evaluator = load_safety_evaluator()
    
    # Launch the demo
    multimodal_chat_demo(model, tokenizer, safety_evaluator)

if __name__ == "__main__":
    main()