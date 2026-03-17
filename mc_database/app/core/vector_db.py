"""
Vector Database Module
Handles FAISS index operations for fast similarity search.
"""
import os
import numpy as np
import faiss

from app.config import FAISS_INDEX_PATH, EMBEDDING_DIM


class VectorDB:
    """FAISS-based vector database for face embeddings."""
    
    def __init__(self):
        """Initialize or load the FAISS index."""
        self.dimension = EMBEDDING_DIM
        self.index_path = FAISS_INDEX_PATH
        
        # ID mapping: FAISS internal ID -> person_id
        self.id_to_person: dict[int, str] = {}
        self.person_to_id: dict[str, int] = {}
        self.next_id = 0
        
        # Load existing index or create new one
        if os.path.exists(self.index_path):
            self.load()
        else:
            # Use IndexFlatIP for cosine similarity (embeddings must be normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def add_embedding(self, person_id: str, embedding: np.ndarray) -> int:
        """
        Add or update embedding for a person.
        
        Args:
            person_id: Unique person identifier
            embedding: 512-dim normalized embedding
            
        Returns:
            Internal FAISS ID
        """
        embedding = np.array([embedding], dtype=np.float32)
        
        if person_id in self.person_to_id:
            # Update existing - remove old and add new
            old_id = self.person_to_id[person_id]
            # Note: FAISS IndexFlat doesn't support removal, so we rebuild
            # For production, use IndexIDMap or other removable index
            self._rebuild_without(old_id)
        
        # Add new embedding
        internal_id = self.next_id
        self.index.add(embedding)
        self.id_to_person[internal_id] = person_id
        self.person_to_id[person_id] = internal_id
        self.next_id += 1
        
        return internal_id
    
    def get_embedding(self, person_id: str) -> np.ndarray | None:
        """
        Retrieve the current embedding for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Numpy array of embedding or None if not found
        """
        if person_id not in self.person_to_id:
            return None
            
        internal_id = self.person_to_id[person_id]
        try:
            # Reconstruct is supported by IndexFlat
            return self.index.reconstruct(internal_id)
        except Exception as e:
            print(f"Error retrieving embedding: {e}")
            return None
            
    def update_embedding(self, person_id: str, embedding: np.ndarray) -> bool:
        """
        Update the embedding for an existing person.
        
        Args:
            person_id: Person identifier
            embedding: New 512-dim normalized embedding
            
        Returns:
            True if successful
        """
        if person_id not in self.person_to_id:
            return False
            
        self.add_embedding(person_id, embedding)
        return True
    
    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        """
        Search for similar faces.
        
        Args:
            query_embedding: 512-dim normalized query embedding
            top_k: Number of results to return
            
        Returns:
            List of (person_id, similarity_score) tuples, sorted by similarity
        """
        if self.index.ntotal == 0:
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        
        # Search for more candidates to handle soft-deleted vectors
        # If top matches are filtering out (deleted), we need subsequent matches
        search_k = min(max(top_k * 20, 20), self.index.ntotal)
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_to_person:
                results.append((self.id_to_person[idx], float(score)))
                if len(results) >= top_k:
                    break
        
        return results
    
    def remove(self, person_id: str) -> bool:
        """
        Remove a person from the index.
        
        Args:
            person_id: Person to remove
            
        Returns:
            True if removed, False if not found
        """
        if person_id not in self.person_to_id:
            return False
        
        internal_id = self.person_to_id[person_id]
        self._rebuild_without(internal_id)
        return True
    
    def _rebuild_without(self, exclude_id: int):
        """Rebuild index excluding a specific ID (for updates/deletions)."""
        # This is inefficient for large indices - production should use IndexIDMap
        if exclude_id in self.id_to_person:
            del self.id_to_person[exclude_id]
        
        person_id_to_remove = None
        for pid, iid in self.person_to_id.items():
            if iid == exclude_id:
                person_id_to_remove = pid
                break
        
        if person_id_to_remove:
            del self.person_to_id[person_id_to_remove]
    
    def save(self):
        """Save the index and mappings to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        # Save mappings
        mapping_path = self.index_path + '.mappings.npy'
        np.save(mapping_path, {
            'id_to_person': self.id_to_person,
            'person_to_id': self.person_to_id,
            'next_id': self.next_id
        }, allow_pickle=True)
    
    def load(self):
        """Load the index and mappings from disk."""
        self.index = faiss.read_index(self.index_path)
        
        mapping_path = self.index_path + '.mappings.npy'
        if os.path.exists(mapping_path):
            mappings = np.load(mapping_path, allow_pickle=True).item()
            self.id_to_person = mappings.get('id_to_person', {})
            self.person_to_id = mappings.get('person_to_id', {})
            self.next_id = mappings.get('next_id', 0)
    
    @property
    def count(self) -> int:
        """Number of embeddings in the index."""
        return self.index.ntotal
        
    def reset(self):
        """Clear the index and all mappings."""
        import shutil
        
        # Reset memory
        self.id_to_person = {}
        self.person_to_id = {}
        self.next_id = 0
        
        # Reset FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Clear files
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except OSError:
                pass
                
        mapping_path = self.index_path + '.mappings.npy'
        if os.path.exists(mapping_path):
            try:
                os.remove(mapping_path)
            except OSError:
                pass


# Global instance
_vector_db: VectorDB | None = None


def get_vector_db() -> VectorDB:
    """Get or create the global VectorDB instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db
