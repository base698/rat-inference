#!/usr/bin/env python3
"""
Vertex AI Gemini Dataset Generator
Uses Google Cloud's Vertex AI to generate training images from reference photos
Optionally validates realism with Gemini 2.5 Pro
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
import io

# --- Configuration ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ratpack")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GENERATION_MODEL = "gemini-2.5-flash-image"
VALIDATION_MODEL = "gemini-2.5-pro"
# --- End of Configuration ---

def setup_vertex_ai():
    """Setup and verify Vertex AI authentication"""
    print("Setting up Vertex AI...")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    
    # Check for authentication
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and PROJECT_ID == "your-project-id":
        print("\n⚠️  WARNING: No authentication configured!")
        print("\nYou need to set up Google Cloud authentication:")
        print("\nOption 1: Use gcloud CLI (easiest for local development)")
        print("  1. Install gcloud: https://cloud.google.com/sdk/install")
        print("  2. Run: gcloud auth application-default login")
        print("  3. Set project: gcloud config set project YOUR_PROJECT_ID")
        print("\nOption 2: Use Service Account (for production)")
        print("  1. Create service account in GCP Console")
        print("  2. Download JSON key file")
        print("  3. Set: export GOOGLE_APPLICATION_CREDENTIALS='path/to/key.json'")
        print("\nAlso set these environment variables:")
        print("  export GCP_PROJECT_ID='your-actual-project-id'")
        print("  export GCP_LOCATION='us-central1'  # or your preferred region")
        return False
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print("✅ Vertex AI initialized successfully")
        return True
    except ImportError:
        print("\n❌ Vertex AI SDK not installed!")
        print("Install it with: pip install google-cloud-aiplatform pillow")
        return False
    except Exception as e:
        print(f"\n❌ Failed to initialize Vertex AI: {e}")
        return False

def generate_image_from_reference(reference_image_path, prompt):
    """
    Generate a new image from a reference image using Gemini 2.5 Flash Image

    Args:
        reference_image_path: Path to the reference image
        prompt: Text prompt for image generation (e.g., "Make a rat appear facing right in the middle of screen")

    Returns:
        bytes: Generated image data, or None if failed
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part

        # Initialize the generation model
        model = GenerativeModel(GENERATION_MODEL)

        # Load the reference image
        with open(reference_image_path, "rb") as f:
            image_bytes = f.read()

        mime_type = "image/png" if str(reference_image_path).endswith('.png') else "image/jpeg"
        image_part = Part.from_data(data=image_bytes, mime_type=mime_type)

        # Generate new image based on reference + prompt
        print(f"  🎨 Generating image with prompt: '{prompt}'")
        response = model.generate_content([image_part, prompt])

        # Extract generated image data
        generated_image_data = response.candidates[0].content.parts[0].inline_data.data
        return generated_image_data

    except Exception as e:
        print(f"  ❌ Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_image_realism(image_data):
    """
    Validate if generated image looks realistic using Gemini 2.5 Pro

    Args:
        image_data: Image data as bytes

    Returns:
        tuple: (is_realistic: bool, explanation: str)
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part

        # Initialize the validation model
        model = GenerativeModel(VALIDATION_MODEL)

        # Create image part from bytes
        image_part = Part.from_data(data=image_data, mime_type="image/png")

        # Validation prompt
        validation_prompt = """Analyze this image carefully and determine if it looks realistic and natural.

Consider:
- Does the subject (rat) look photorealistic?
- Are there obvious AI artifacts, glitches, or distortions?
- Does the lighting and composition look natural?
- Are the proportions and anatomy correct?
- Does it look like a real photograph?

Respond with ONLY a JSON object in this exact format:
{
    "realistic": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}"""

        print("  🔍 Validating image realism with Gemini 2.5 Pro...")
        response = model.generate_content([image_part, validation_prompt])

        # Parse JSON response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        is_realistic = result.get("realistic", False)
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "No reason provided")

        status = "✅ REALISTIC" if is_realistic else "❌ NOT REALISTIC"
        print(f"  {status} (confidence: {confidence:.2f})")
        print(f"  Reason: {reason}")

        return is_realistic, reason

    except Exception as e:
        print(f"  ⚠️  Error validating image: {e}")
        print("  Defaulting to accepting the image...")
        return True, "Validation failed, accepted by default"


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset images from reference photos using Vertex AI Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single image with validation
  python generate-dataset.py --reference ref.jpg --prompt "Make a rat appear facing right in the middle" --dataset rat --validate

  # Generate multiple variations without validation
  python generate-dataset.py --reference ref.jpg --prompts prompts.txt --dataset rat --count 10

  # Use custom prompts directly
  python generate-dataset.py --reference ref.jpg --prompt "Add a rat in the center" --prompt "Add a rat on the left side" --dataset rat
        """
    )

    # Required arguments
    parser.add_argument("--reference", "-r", type=str, required=True,
                       help="Path to reference image")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                       help="Dataset name (saves to ./datasets/{name}/unsorted)")

    # Prompt arguments (either --prompt or --prompts file)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", "-p", action="append",
                             help="Text prompt for generation (can be used multiple times)")
    prompt_group.add_argument("--prompts-file", type=str,
                             help="File containing prompts (one per line)")

    # Optional arguments
    parser.add_argument("--validate", action="store_true",
                       help="Validate image realism with Gemini 2.5 Pro before saving")
    parser.add_argument("--count", "-c", type=int, default=1,
                       help="Number of images to generate per prompt (default: 1)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip if output file already exists")

    args = parser.parse_args()

    print("="*60)
    print("Vertex AI Gemini Dataset Generator")
    print("="*60)

    # Check setup
    if not setup_vertex_ai():
        print("\n⚠️  Please configure authentication first (see instructions above)")
        return

    # Verify reference image exists
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"\n❌ Reference image not found: {args.reference}")
        return

    # Load prompts
    prompts = []
    if args.prompt:
        prompts = args.prompt
    elif args.prompts_file:
        prompts_file = Path(args.prompts_file)
        if not prompts_file.exists():
            print(f"\n❌ Prompts file not found: {args.prompts_file}")
            return
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print("\n❌ No prompts provided!")
        return

    # Create output directory
    output_dir = Path(f"./datasets/{args.dataset}/unsorted")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Reference image: {args.reference}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Prompts: {len(prompts)}")
    print(f"🔢 Count per prompt: {args.count}")
    print(f"🔍 Validation: {'ENABLED' if args.validate else 'DISABLED'}")
    print()

    # Process images
    total_attempts = 0
    total_generated = 0
    total_saved = 0
    total_rejected = 0

    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {prompt_idx}/{len(prompts)}]: {prompt}")

        for iteration in range(1, args.count + 1):
            total_attempts += 1

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_filename = f"gen_{timestamp}.png"
            output_path = output_dir / output_filename

            # Skip if exists
            if args.skip_existing and output_path.exists():
                print(f"  ⏭️  Skipping (already exists): {output_filename}")
                continue

            print(f"  [{iteration}/{args.count}] Generating...")

            # Generate image
            image_data = generate_image_from_reference(reference_path, prompt)

            if image_data is None:
                print(f"  ❌ Failed to generate image")
                continue

            total_generated += 1

            # Validate if requested
            if args.validate:
                is_realistic, reason = validate_image_realism(image_data)

                if not is_realistic:
                    print(f"  ❌ Image rejected (not realistic)")
                    total_rejected += 1
                    continue

            # Save image
            try:
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"  ✅ Saved: {output_filename}")
                total_saved += 1
            except Exception as e:
                print(f"  ❌ Failed to save: {e}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total attempts:    {total_attempts}")
    print(f"Successfully generated: {total_generated}")
    if args.validate:
        print(f"Rejected (not realistic): {total_rejected}")
    print(f"Saved to disk:     {total_saved}")
    print(f"\n✅ Check '{output_dir}' for generated images")
    print("="*60)

if __name__ == "__main__":
    main()
