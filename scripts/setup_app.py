import os
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path so we can import from src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("VAY Assistant - Environment Setup & Startup Sequence")
    print("=" * 60)
    
    # 1. Seed the Customer Database
    print("\n[1/4] Ensuring Customer Database is seeded...")
    try:
        from vay.tools.db_queries import init_db
        init_db()
        print("✓ Database is ready.")
    except Exception as e:
        print(f"✗ Failed to seed database: {e}")
        
    # 2. Build the Knowledge Base (Idempotent)
    print("\n[2/4] Building/Updating ChromaDB Knowledge Base...")
    try:
        # Run it as a subprocess to keep the environment clean and match old behavior
        result = subprocess.run(
            ["uv", "run", "python", "scripts/build_kb.py"],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        print("✓ Knowledge Base is ready.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build KB: {e}")
        
    # 3. Cache ASR Models (IndicConformer)
    print("\n[3/4] Caching ASR Models (IndicConformer)...")
    try:
        from vay.asr.indic import IndicConformerASR
        from vay.config import settings
        
        # Instantiate to force the Hugging Face cache to download/load them
        _ = IndicConformerASR(model_id=settings.indic_asr_model)
        
        print("✓ ASR Models are cached and ready.")
    except Exception as e:
        print(f"✗ Failed to load ASR models: {e}")
        
    # 4. Start the Streamlit Application
    print("\n[4/4] Starting the VAY Streamlit Interface...")
    print("=" * 60)
    try:
        subprocess.run(
            ["uv", "run", "streamlit", "run", "app.py"],
            cwd=str(PROJECT_ROOT)
        )
    except KeyboardInterrupt:
        print("\nShutting down VAY Assistant...")

if __name__ == "__main__":
    main()
