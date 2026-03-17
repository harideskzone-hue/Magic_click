
import os
import sys
import asyncio
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.face_service import FaceService
from app.core.vector_db import get_vector_db

async def verify_fix():
    print("Verifying FaceService logic tweak...")
    
    # Setup
    fs = FaceService()
    vector_db = get_vector_db()
    
    target_image_path = r"c:\projects\mc_database\data\images\1b50434d-9aea-40f6-82de-83382d393363\b7bc89ce-f650-4d01-af2f-796d6c1af7b3.jpg"
    
    if not os.path.exists(target_image_path):
        print("Image not found, cannot verify.")
        return

    # Read image bytes
    with open(target_image_path, "rb") as f:
        img_bytes = f.read()
        
    # Case 1: cropped=True with Large Image
    # BEFORE FIX: This would likely fail to match (create new person) or error if we were watching logs
    # AFTER FIX: This should detect the face and find the match (since we know it matches from debug_matching.py)
    
    print("\n--- Test: calling process_and_add_image with cropped=True and Large Image ---")
    # We use a distinct score to identify this add if needed, but we mostly care about the return 'action'
    # Since the person likely already exists now (or maybe multiple versions), we expect 'added_to_existing' 
    # OR if my debug script showed it matched 1b50..., we expect it to match that ID.
    
    result = fs.process_and_add_image(img_bytes, score=0.99, is_cropped=True)
    
    print("Result:", result)
    
    if result['success']:
        if result.get('action') == 'added_to_existing':
            print("SUCCESS: Matched existing person even with cropped=True flag!")
            print(f"Matched Person ID: {result.get('person_id')}")
        else:
            print("WARNING: Created new person. Check if the person is currently in DB.")
            # If the person isn't in DB, 'created_new' is correct BEHAVIOR but implies detection worked.
            # If detection FAILED (old behavior), the embedding would be garbage. 
            # Garbage embeddings usually don't match anything, so 'created_new' happens.
            # We need to verify the embedding is GOOD.
            
            # Let's verify the embedding quality by searching for itself
            pid = result['person_id']
            vec = vector_db.get_embedding(pid)
            # A garbage embedding from a 4k image resize usually has very specific properties (smooth noise)
            # But simpler: we know the 'correct' embedding for this file matches 1b50...
            # If it created a NEW person, that might be because 1b50... was deleted or something?
            pass
            
            # Actually, to be sure, we can check if the code printed "Ignoring is_cropped flag..." if we add a print there.
    else:
        print("FAILED: Service returned error.")

if __name__ == "__main__":
    asyncio.run(verify_fix())
