from typing import Dict, List, Optional, Any, Union
import logging
import uuid
import threading
import numpy as np  # type: ignore

from app.config import SIMILARITY_THRESHOLD, MIN_DET_SCORE  # type: ignore
from app.core.encoder import get_encoder  # type: ignore
from app.core.vector_db import get_vector_db  # type: ignore
from app.core.storage import get_storage  # type: ignore
from app.core.image_utils import decode_base64_image, encode_image_to_bytes  # type: ignore

log = logging.getLogger("face_service")

# Global lock to prevent concurrent FastAPI threads from corrupting FAISS or SQLite
_service_lock = threading.Lock()


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
        Using a global lock to prevent concurrent FastAPI threads from corrupting FAISS or SQLite.
        """
        with _service_lock:
            return self._process_and_add_image_internal(image_data, score, is_cropped)

    def _process_and_add_image_internal(self, image_data: Union[str, bytes], score: float, is_cropped: bool = False) -> Dict[str, Any]:
        """
        Internal unlocked method for process_and_add_image.
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
            import cv2  # type: ignore
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or img_bytes is None:
            return {'success': False, 'error': 'Invalid image data'}

        # ── Encode face ─────────────────────────────────────────────────────────
        embedding = None

        assert image is not None

        if is_cropped and image.shape[0] == 112 and image.shape[1] == 112:
            embedding = encoder.encode_cropped_face(image)
        else:
            embedding, face_info = encoder.detect_and_encode(image)
            if face_info is not None:
                det_score = face_info.get('det_score', 1.0)
                if det_score < MIN_DET_SCORE:
                    log.warning("add: face detection score %.3f < MIN_DET_SCORE %.3f — rejected",
                                det_score, MIN_DET_SCORE)
                    return {'success': False, 'error': f'Low detection confidence: {det_score:.3f}'}

        if embedding is None:
            log.warning("add: embedding is None (no face detected) for image shape %s",
                        getattr(image, 'shape', 'unknown'))
            return {'success': False, 'error': 'No face detected or encoding failed'}

        # ── Search for match ─────────────────────────────────────────────────────
        matches = vector_db.search(embedding, top_k=1)

        log.info("add: SIMILARITY_THRESHOLD=%.3f, best_match=%s",
                 SIMILARITY_THRESHOLD,
                 f"{matches[0][0]} @ {matches[0][1]:.4f}" if matches else "none")

        if matches and matches[0][1] >= SIMILARITY_THRESHOLD:
            person_id, similarity = matches[0]
            person = storage.get_person(person_id)

            if person:
                # Update running average embedding (weighted by image count)
                current_embedding = vector_db.get_embedding(person_id)
                if current_embedding is not None:
                    n = person['image_count']
                    new_embedding = (current_embedding * n + embedding) / (n + 1)
                    new_embedding = new_embedding / np.linalg.norm(new_embedding)
                    vector_db.update_embedding(person_id, new_embedding)
                    log.info("add: updated embedding for person %s (n=%d → %d)", person_id, n, n+1)

                image_id = storage.save_image(person_id, img_bytes, score)
                vector_db.save()

                log.info("add: added image %s to EXISTING person %s (sim=%.4f)",
                         image_id, person_id, similarity)
                return {
                    'success': True,
                    'action': 'added_to_existing',
                    'person_id': person_id,
                    'name': person.get('name'),
                    'similarity': float(f"{similarity:.4f}"),
                    'image_count': person['image_count'] + 1,
                    'image_id': image_id
                }

        # ── No match — create new person ─────────────────────────────────────────
        # Use a unique placeholder key (not shared 'temp') to avoid clobbering
        # when multiple /api/add requests arrive concurrently.
        placeholder_key = f"_new_{uuid.uuid4().hex}"
        faiss_index = vector_db.add_embedding(placeholder_key, embedding)
        person_id = storage.create_person(faiss_index)

        # Move placeholder → real person_id in mappings
        vector_db.person_to_id[person_id] = vector_db.person_to_id.pop(placeholder_key)
        vector_db.id_to_person[faiss_index] = person_id

        image_id = storage.save_image(person_id, img_bytes, score)
        vector_db.save()

        log.info("add: created NEW person %s (no match above threshold)", person_id)
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
        
        assert image is not None
        
        if is_cropped and image.shape[0] == 112 and image.shape[1] == 112:
            embedding = encoder.encode_cropped_face(image)
        else:
            embedding, _ = encoder.detect_and_encode(image)

        if embedding is None:
            error_msg = 'Failed to encode cropped face' if is_cropped else 'No face detected in image'
            log.info("search: %s", error_msg)
            return {'success': False, 'error': error_msg}

        matches = vector_db.search(embedding, top_k=1)

        log.info("search: best_match=%s  threshold=%.3f",
                 f"{matches[0][0]} @ {matches[0][1]:.4f}" if matches else "none",
                 SIMILARITY_THRESHOLD)

        if not matches or matches[0][1] < SIMILARITY_THRESHOLD:
            best_score = matches[0][1] if matches else 0.0
            log.info("search: NO MATCH (best_score=%.4f < threshold=%.3f)", best_score, SIMILARITY_THRESHOLD)
            return {
                'success': True,
                'match': False,
                'message': 'No matching person found',
                'best_score': float(f"{best_score:.4f}"),
                'threshold': SIMILARITY_THRESHOLD
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

        log.info("search: MATCH person=%s name=%s sim=%.4f images=%d",
                 person_id, person.get('name'), similarity, len(image_data))
        return {
            'success': True,
            'match': True,
            'person_id': person_id,
            'name': person.get('name'),
            'similarity': float(f"{similarity:.4f}"),
            'images': image_data
        }

    def get_person_by_id_or_name(self, pid: Optional[str], name: Optional[str]) -> Dict[str, Any]:
        """Retrieve person details by ID or Name."""
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
            'person_id': person['id'],
            'name': person.get('name'),
            'images': image_data
        }

    def assign_name(self, person_id: str, name: str) -> Dict[str, Any]:
        """Assign a name to a person ID."""
        storage, _, _ = self._get_deps()

        person = storage.get_person(person_id)
        if not person:
            return {'success': False, 'error': 'Person not found'}

        success = storage.assign_name(person_id, name)
        if not success:
            return {'success': False, 'error': 'Name already taken or update failed'}

        return {'success': True, 'person_id': person_id, 'name': name}

    def delete_person(self, person_id: str) -> Dict[str, Any]:
        """Delete a person and all valid references."""
        storage, vector_db, _ = self._get_deps()

        person = storage.get_person(person_id)
        if not person:
            return {'success': False, 'error': 'Person not found'}

        vector_db.remove(person_id)
        vector_db.save()
        storage.delete_person(person_id)

        log.info("delete: person %s removed.", person_id)
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

        return {'success': True, 'images': image_list}

    def get_image_bytes(self, image_id: str) -> Optional[bytes]:
        """Get raw image bytes."""
        storage, _, _ = self._get_deps()
        return storage.get_image_bytes(image_id)
