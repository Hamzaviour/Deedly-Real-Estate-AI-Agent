"""Rebuild the vector index. Run: python -m realestate_agent.ingest"""
from .rag import KnowledgeBase

if __name__ == "__main__":
    print(KnowledgeBase().ingest())
