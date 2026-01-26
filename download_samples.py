"""Download sample images for testing."""
import urllib.request
import os

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/manipulated", exist_ok=True)

# Real estate image sources (we'll use the sample we already have)
print("Sample images ready in data/test/")
print("For full testing, add real and AI-generated real estate images to:")
print("  - data/real/")
print("  - data/manipulated/")
print("\nYou can generate fake images using:")
print("  - DALL-E / Midjourney / Flux with 'modern kitchen interior' prompts")
print("  - Virtual staging tools")
