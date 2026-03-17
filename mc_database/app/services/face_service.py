from typing import Dict, List, Optional, Any, Union
import numpy as np

from app.config import SIMILARITY_THRESHOLD
from app.core.encoder import get_encoder
from app.core.vector_db import get_vector_db
from app.core.storage import get_storage
from app.core.image_utils import decode_base64_image, encode_image_to_bytes

class FaceService:
    """
    Service for handling face recognition business logic.
    Follows SRP by encapsulating database interactions and core processing.
    """
    
    def __init__(self):
        pass

    def _get_deps(self):
        return get_storage(), get_vector_db(), get_encoder()

    def get_health_stats(self) -> Dict[str, Any]:
        """Get system health statistics."""
        storage, vector_db, _ = self._get_deps()
        return {
            'status': 'healthy',
            'person_count': storage.get_person_count(),
            'vector_count': vector_db.count
        }

    def process_and_add_image(self, image_data: Union[str, bytes], score: float, is_cropped: bool = False) -> Dict[str, Any]:
        """
        Process and add a single image to the database.
        """
        storage, vector_db, encoder = self._get_deps()
        
        image = None
        img_bytes = None
        
        # Prepare image data
        if isinstance(image_data, str):
            image = decode_base64_image(image_data)
            if image is not None:
                 img_bytes = encode_image_to_bytes(image)
        elif isinstance(image_data, bytes):
            img_bytes = image_data
            nparr = np.frombuffer(image_data, np.uint8)
            import cv2
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if image is None or img_bytes is None:
            return {'success': False, 'error': 'Invalid image data'}
        
        # Encode image
        embedding = None
        
        # Safety check: Only trust is_cropped if image dimensions are actually 112x112
        if is_cropped and image.shape[0] == 112 and image.shape[1] == 112:
            embedding = encoder.encode_cropped_face(image)
        else:
            if is_cropped:
                # print(f"DEBUG: Ignoring is_cropped=True for image with shape {image.shape}")
                pass
            embedding, _ = encoder.detect_and_encode(image)
            
        if embedding is None:
            return {'success': False, 'error': 'No face detected or encoding failed'}
        
        # Search for existing match
        matches = vector_db.search(embedding, top_k=1)
        
        if matches and matches[0][1] >= SIMILARITY_THRESHOLD:
            # Match found
            person_id, similarity = matches[0]
            person = storage.get_person(person_id)
            
            if person:
                # Update running average embedding
                current_embedding = vector_db.get_embedding(person_id)
                if current_embedding is not None:
                    # Calculate new weighted average
                    # new = (old * N + current) / (N + 1)
                    # Note: person['image_count'] is N (before this new image)
                    n = person['image_count']
                    new_embedding = (current_embedding * n + embedding) / (n + 1)
                    # Normalize
                    new_embedding = new_embedding / np.linalg.norm(new_embedding)
                    vector_db.update_embedding(person_id, new_embedding)

                image_id = storage.save_image(person_id, img_bytes, score)
                new_count = person['image_count'] + 1
                vector_db.save()
                
                return {
                    'success': True,
                    'action': 'added_to_existing',
                    'person_id': person_id,
                    'name': person.get('name'),
                    'similarity': round(similarity, 3),
                    'image_count': new_count,
                    'image_id': image_id
                }
        
        # No match - create new
        faiss_index = vector_db.add_embedding('temp', embedding)
        person_id = storage.create_person(faiss_index)
        
        # Update FAISS mapping
        vector_db.person_to_id[person_id] = vector_db.person_to_id.pop('temp')
        vector_db.id_to_person[faiss_index] = person_id
        
        # Save image
        image_id = storage.save_image(person_id, img_bytes, score)
        vector_db.save()
        
        return {
            'success': True,
            'action': 'created_new',
            'person_id': person_id,
            'image_count': 1,
            'image_id': image_id
        }

    def search_image(self, base64_img: str, is_cropped: bool) -> Dict[str, Any]:
        """
        Search for a face in the database.
        """
        if not base64_img:
            return {'success': False, 'error': 'No image provided'}
        
        image = decode_base64_image(base64_img)
        if image is None:
            return {'success': False, 'error': 'Invalid image data'}
        
        storage, vector_db, encoder = self._get_deps()
        
        embedding = None
        
        # Safety check: search_image
        if is_cropped and image.shape[0] == 112 and image.shape[1] == 112:
            embedding = encoder.encode_cropped_face(image)
        else:
            embedding, _ = encoder.detect_and_encode(image)
        
        if embedding is None:
            error_msg = 'Failed to encode cropped face' if is_cropped else 'No face detected'
            return {'success': False, 'error': error_msg}
        
        matches = vector_db.search(embedding, top_k=1)
        
        if not matches or matches[0][1] < SIMILARITY_THRESHOLD:
            return {
                'success': True,
                'match': False,
                'message': 'No matching person found'
            }
        
        person_id, similarity = matches[0]
        person = storage.get_person(person_id)
        
        if not person:
            return {'success': False, 'error': 'Person data not found'}
        
        images = storage.get_all_images(person_id)
        image_data = [
            {
                'image': storage.get_image_as_base64(img['id']),
                'score': img['score'],
                'id': img['id']
            }
            for img in images
        ]
        
        return {
            'success': True,
            'match': True,
            'person_id': person_id,
            'name': person.get('name'),
            'similarity': round(similarity, 3),
            'images': image_data
        }

    def get_person_by_id_or_name(self, pid: Optional[str], name: Optional[str]) -> Dict[str, Any]:
        """
        Retrieve person details by ID or Name
        """
        storage, _, _ = self._get_deps()
        
        person = None
        if pid:
            person = storage.get_person(pid)
        elif name:
            person = storage.get_person_by_id_or_name(name)
            
        if not person:
            return {'success': False, 'error': 'Person not found'}
            
        images = storage.get_all_images(person['id'])
        image_data = [
            {
                'image': storage.get_image_as_base64(img['id']),
                'score': img['score'],
                'id': img['id']
            }
            for img in images
        ]
        
        return {
            'success': True,
            'person': {
                 'id': person['id'],
                 'name': person.get('name'),
                 'image_count': person['image_count'],
                 'created_at': person['created_at']
            },
            'person_id': person['id'], # For backward compatibility in response
            'name': person.get('name'),
            'images': image_data
        }

    def assign_name(self, person_id: str, name: str) -> Dict[str, Any]:
        """
        Assign a name to a person ID.
        """
        storage, _, _ = self._get_deps()
        
        person = storage.get_person(person_id)
        if not person:
             return {'success': False, 'error': 'Person not found'}
             
        success = storage.assign_name(person_id, name)
        
        if not success:
            return {'success': False, 'error': 'Name already taken or update failed'}
            
        return {
            'success': True,
            'person_id': person_id,
            'name': name
        }

    def delete_person(self, person_id: str) -> Dict[str, Any]:
        """
        Delete a person and all valid references.
        """
        storage, vector_db, _ = self._get_deps()
        
        person = storage.get_person(person_id)
        if not person:
            return {'success': False, 'error': 'Person not found'}
        
        vector_db.remove(person_id)
        vector_db.save()
        storage.delete_person(person_id)
        
        return {'success': True, 'message': f'Person {person_id} deleted'}

    def get_top_images(self, n: int = 3) -> Dict[str, Any]:
        """Get top N images globally by score."""
        storage, _, _ = self._get_deps()
        images = storage.get_top_images_global(n=n)
        
        image_list = []
        for img in images:
            image_list.append({
                'id': img['id'],
                'person_id': img['person_id'],
                'person_name': img.get('person_name'),
                'score': img['score'],
                'image': storage.get_image_as_base64(img['id'])
            })
        
        return {
            'success': True,
            'images': image_list
        }

    def get_image_bytes(self, image_id: str) -> Optional[bytes]:
        """Get raw image bytes."""
        storage, _, _ = self._get_deps()
        return storage.get_image_bytes(image_id)
