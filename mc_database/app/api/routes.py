"""
FastAPI Routes
New API structure: /search, /add, /name, /topn
Refactored to use FaceService.
"""
from typing import List, Optional
from fastapi import APIRouter, UploadFile, Form, HTTPException, Body, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.services.face_service import FaceService

api = APIRouter(prefix="/api")
face_service = FaceService()

# Pydantic Models
class BatchAddImage(BaseModel):
    img: str
    score: float = 0.0
    name: Optional[str] = None
    cropped: bool = False

class BatchAddRequest(BaseModel):
    images: List[BatchAddImage]

class CreatePersonRequest(BaseModel):
    name: str

class SearchRequest(BaseModel):
    img: Optional[str] = None
    cropped: bool = False

# Endpoints

@api.get('/health')
async def health():
    """Health check endpoint."""
    return face_service.get_health_stats()

@api.post('/search')
async def search_post(request: SearchRequest):
    """
    Search by face image (POST).
    """
    result = face_service.search_image(request.img, request.cropped)
    status_code = 200 if result.get('success', False) else 400
    if not result.get('success') and result.get('match') is False:
         status_code = 200 # No match is a valid result
    
    # Handle specific error cases that were returned as 500 in original code if appropriate, 
    # but 400 is generally safer for client errors. 
    # Original code returned 500 for 'Person data not found' logic error.
    if result.get('error') == 'Person data not found':
        status_code = 500
        
    return JSONResponse(result, status_code=status_code)

@api.get('/search')
async def search_get(
    id: Optional[str] = Query(None),
    name: Optional[str] = Query(None)
):
    """
    Search by ID or name (GET).
    """
    if not id and not name:
        return JSONResponse({'success': False, 'error': 'No search parameter provided (id or name)'}, status_code=400)
    
    result = face_service.get_person_by_id_or_name(id, name)
    status_code = 200 if result.get('success') else 404
    return JSONResponse(result, status_code=status_code)

@api.post('/add')
async def add_image(request: Request):
    """
    Add an image - auto-matches to existing person or creates new.
    Supports JSON (base64) and Multipart/Form-data (file upload).
    """
    content_type = request.headers.get('content-type', '')
    
    score = 0.0
    cropped = False
    img_data = None
    
    try:
        if 'application/json' in content_type:
            data = await request.json()
            img_data = data.get('img')
            score = float(data.get('score', 0.0))
            cropped = data.get('cropped', False)
            
        elif 'multipart/form-data' in content_type:
            form = await request.form()
            score = float(form.get('score', 0.0))
            cropped = str(form.get('cropped', '')).lower() in ('true', '1', 'yes')
            
            # Check for file
            upload_file = form.get('image')
            if upload_file is not None and hasattr(upload_file, 'read'):
                img_data = await upload_file.read()
            else:
                # Check for base64 string in form
                img_data = form.get('img')
                
        else:
            return JSONResponse({'success': False, 'error': 'Unsupported Content-Type'}, status_code=400)
            
        if not img_data:
             return JSONResponse({'success': False, 'error': 'No image provided'}, status_code=400)
             
        # Process
        result = face_service.process_and_add_image(img_data, score, cropped)
        status_code = 200 if result['success'] else 400
        return JSONResponse(result, status_code=status_code)
        
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=400)

@api.post('/batch_add')
async def batch_add_images(request: BatchAddRequest):
    """
    Add multiple images in batch.
    """
    images = request.images
    
    if not images:
        return JSONResponse({'success': False, 'error': 'List of images required'}, status_code=400)
    
    results = []
    success_count = 0
    fail_count = 0
    
    print(f"DEBUG: Batch processing {len(images)} images...")
    
    for i, img_data in enumerate(images):
        result = face_service.process_and_add_image(img_data.img, img_data.score, img_data.cropped)
        result['index'] = i
        
        if result['success']:
            name = img_data.name
            # If name provided, try to assign it. This logic is slightly coupled 
            # but fits 'process_and_add' extension or calling assign_name separately.
            if name and result.get('person_id'):
                face_service.assign_name(result['person_id'], name)
                result['name_assigned'] = name
            success_count += 1
        else:
            print(f"DEBUG: Batch item {i} failed: {result.get('error')}")
            fail_count += 1
        
        results.append(result)
        
    return {
        'success': True,
        'total': len(images),
        'added': success_count,
        'failed': fail_count,
        'results': results
    }

@api.delete('/reset')
async def reset_database():
    """Clear all data from the database."""
    # This involves multiple services (storage, vector_db). 
    # For now, it's safer to keep this special administrative action here 
    # or add a 'reset_all' to service. 
    # Let's add it to service for consistency in next step or use dependency access here?
    # Better: Use dependencies directly as this is admin function, OR strictly use service.
    # Service doesn't have reset() yet. 
    # Implementation detail: I'll use the service deps pattern or imports. 
    # To follow Clean Code: add reset to service.
    try:
        from app.core.storage import get_storage
        from app.core.vector_db import get_vector_db
        storage = get_storage()
        vector_db = get_vector_db()
        storage.reset()
        vector_db.reset()
        
        return {
            'success': True,
            'message': 'Database successfully cleared'
        }
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.post('/name')
async def assign_name(request: Request):
    """Assign a name to a person."""
    try:
        content_type = request.headers.get('content-type', '')
        final_pid = None
        final_name = None
        
        if 'application/json' in content_type:
            data = await request.json()
            final_pid = data.get('person_id')
            final_name = data.get('name')
        elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
            form = await request.form()
            final_pid = form.get('person_id')
            final_name = form.get('name')
    except Exception:
        return JSONResponse({'success': False, 'error': 'Invalid request body'}, status_code=400)
    
    if not final_pid or not final_name:
        return JSONResponse({'success': False, 'error': 'person_id and name required'}, status_code=400)
    
    result = face_service.assign_name(final_pid, final_name)
    status_code = 200 if result['success'] else 400
    if result.get('error') == 'Person not found':
        status_code = 404
        
    return JSONResponse(result, status_code=status_code)

@api.get('/topn')
async def get_top_images():
    """Get top 3 images globally by score."""
    return face_service.get_top_images(n=3)

@api.get('/person/{person_id}')
async def get_person(person_id: str):
    """Get person info by ID."""
    result = face_service.get_person_by_id_or_name(pid=person_id, name=None)
    
    if result.get('success'):
         return {
             'success': True,
             'person': result['person']
         }
    else:
        return JSONResponse(result, status_code=404)

@api.delete('/person/{person_id}')
async def delete_person(person_id: str):
    """Delete a person and all their images."""
    result = face_service.delete_person(person_id)
    status_code = 200 if result['success'] else 404
    return JSONResponse(result, status_code=status_code)

@api.get('/image/{image_id}')
async def get_image(image_id: str):
    """Serve an image file by ID (raw JPEG bytes)."""
    image_bytes = face_service.get_image_bytes(image_id)
    
    if not image_bytes:
        return JSONResponse({'success': False, 'error': 'Image not found'}, status_code=404)
    
    return Response(image_bytes, media_type='image/jpeg')
