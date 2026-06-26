#app/utils/memory.py

from cachetools import TTLCache
from threading import Lock

memory_cache = TTLCache(maxsize=10000, ttl=3600)
lock = Lock()


class MemoryStore:
  """ In memory conversational history with 1-hour expiry."""
  def get_history(self, session_id: str):
      """ Retrieve conversation history list of messages"""

      with lock:
          return memory_cache.get(session_id, []).copy()

  def save_history(self,session_id: str, history: list) :
      """ save/overwrite conversation history."""      
      with lock:
           memory_cache[session_id] = history.copy()

  def clear_history(self, session_id: str):
      """Manually clear a session. """  
      with lock:
           memory_cache.pop(session_id, None) 

memory_store = MemoryStore()               