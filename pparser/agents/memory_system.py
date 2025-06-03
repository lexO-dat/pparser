"""
Memory management system for agents to replace simple list-based memory.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class MemoryType(Enum):
    """Types of memory entries."""
    INTERACTION = "interaction"
    CONTEXT = "context" 
    RESULT = "result"
    ERROR = "error"
    METADATA = "metadata"


@dataclass
class MemoryEntry:
    """Individual memory entry."""
    id: str
    type: MemoryType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: str = ""
    tags: List[str] = field(default_factory=list)
    importance: int = 1  # 1-10 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'tags': self.tags,
            'importance': self.importance
        }


class AgentMemory:
    """Enhanced memory system for agents."""
    
    def __init__(self, agent_name: str, max_entries: int = 100):
        self.agent_name = agent_name
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self._entry_counter = 0
    
    def add_entry(self, entry_type: MemoryType, content: Dict[str, Any],
                  tags: List[str] = None, importance: int = 1) -> str:
        """Add new memory entry."""
        self._entry_counter += 1
        entry_id = f"{self.agent_name}_{entry_type.value}_{self._entry_counter}"
        
        entry = MemoryEntry(
            id=entry_id,
            type=entry_type,
            content=content,
            agent_name=self.agent_name,
            tags=tags or [],
            importance=importance
        )
        
        self.entries.append(entry)
        self._cleanup_old_entries()
        
        return entry_id
    
    def add_interaction(self, prompt: str, response: str, 
                       success: bool = True, metadata: Dict = None) -> str:
        """Add LLM interaction to memory."""
        content = {
            'prompt': prompt[:500],  # Truncate for storage
            'response': response[:1000],
            'success': success,
            'metadata': metadata or {}
        }
        
        importance = 3 if success else 5  # Failed interactions are more important
        return self.add_entry(MemoryType.INTERACTION, content, 
                             tags=['llm', 'interaction'], importance=importance)
    
    def add_context(self, context_type: str, data: Dict[str, Any]) -> str:
        """Add context information to memory."""
        content = {
            'context_type': context_type,
            'data': data
        }
        
        return self.add_entry(MemoryType.CONTEXT, content,
                             tags=['context', context_type], importance=2)
    
    def add_result(self, operation: str, result: Any, 
                   processing_time: float = None) -> str:
        """Add operation result to memory."""
        content = {
            'operation': operation,
            'result_type': type(result).__name__,
            'result_summary': str(result)[:200],
            'processing_time': processing_time,
            'success': True
        }
        
        return self.add_entry(MemoryType.RESULT, content,
                             tags=['result', operation], importance=3)
    
    def get_entries_by_type(self, entry_type: MemoryType) -> List[MemoryEntry]:
        """Get all entries of specific type."""
        return [entry for entry in self.entries if entry.type == entry_type]
    
    def get_entries_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        """Get entries that have any of the specified tags."""
        return [entry for entry in self.entries 
                if any(tag in entry.tags for tag in tags)]
    
    def get_recent_entries(self, count: int = 10) -> List[MemoryEntry]:
        """Get most recent entries."""
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:count]
    
    def get_important_entries(self, min_importance: int = 7) -> List[MemoryEntry]:
        """Get high importance entries."""
        return [entry for entry in self.entries if entry.importance >= min_importance]
    
    def search_entries(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search entries by content."""
        query_lower = query.lower()
        matches = []
        
        for entry in self.entries:
            content_str = json.dumps(entry.content).lower()
            if query_lower in content_str or query_lower in ' '.join(entry.tags):
                matches.append(entry)
        
        # Sort by importance and recency
        matches.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return matches[:limit]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory contents."""
        type_counts = {}
        tag_counts = {}
        total_entries = len(self.entries)
        
        for entry in self.entries:
            type_counts[entry.type.value] = type_counts.get(entry.type.value, 0) + 1
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            'total_entries': total_entries,
            'type_distribution': type_counts,
            'tag_distribution': tag_counts,
            'oldest_entry': min(self.entries, key=lambda x: x.timestamp).timestamp.isoformat() if self.entries else None,
            'newest_entry': max(self.entries, key=lambda x: x.timestamp).timestamp.isoformat() if self.entries else None
        }
    
    def _cleanup_old_entries(self):
        """Remove old entries when memory is full."""
        if len(self.entries) <= self.max_entries:
            return
        
        # Sort by importance (desc) and timestamp (desc)
        sorted_entries = sorted(self.entries, 
                               key=lambda x: (x.importance, x.timestamp), 
                               reverse=True)
        
        # Keep most important and recent entries
        self.entries = sorted_entries[:self.max_entries]
    
    def clear_memory(self, keep_important: bool = True):
        """Clear memory, optionally keeping important entries."""
        if keep_important:
            self.entries = self.get_important_entries(min_importance=8)
        else:
            self.entries = []
        
        self._entry_counter = len(self.entries)
    
    def export_memory(self) -> Dict[str, Any]:
        """Export memory for persistence."""
        return {
            'agent_name': self.agent_name,
            'max_entries': self.max_entries,
            'entry_counter': self._entry_counter,
            'entries': [entry.to_dict() for entry in self.entries]
        }


class MemoryManager:
    """Manager for multiple agent memories."""
    
    def __init__(self):
        self.agent_memories: Dict[str, AgentMemory] = {}
    
    def get_memory(self, agent_name: str, max_entries: int = 100) -> AgentMemory:
        """Get or create memory for agent."""
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = AgentMemory(agent_name, max_entries)
        
        return self.agent_memories[agent_name]
    
    def get_cross_agent_insights(self) -> Dict[str, Any]:
        """Get insights across all agent memories."""
        total_entries = sum(len(memory.entries) for memory in self.agent_memories.values())
        
        all_tags = {}
        all_types = {}
        
        for memory in self.agent_memories.values():
            for entry in memory.entries:
                all_types[entry.type.value] = all_types.get(entry.type.value, 0) + 1
                for tag in entry.tags:
                    all_tags[tag] = all_tags.get(tag, 0) + 1
        
        return {
            'total_agents': len(self.agent_memories),
            'total_entries': total_entries,
            'global_type_distribution': all_types,
            'global_tag_distribution': all_tags,
            'agent_summaries': {name: memory.get_memory_summary() 
                               for name, memory in self.agent_memories.items()}
        }
