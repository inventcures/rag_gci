from fastapi import Depends

_rag_pipeline = None
_safety_manager = None
_memory_manager = None
_medication_manager = None
_memory_agents_store = None
_memory_agents_query = None
_meta_evaluator = None


def init_dependencies(rag_pipeline, safety_manager, memory_manager, medication_manager):
    global _rag_pipeline, _safety_manager, _memory_manager, _medication_manager
    _rag_pipeline = rag_pipeline
    _safety_manager = safety_manager
    _memory_manager = memory_manager
    _medication_manager = medication_manager


def init_memory_agents(store, query_agent):
    global _memory_agents_store, _memory_agents_query
    _memory_agents_store = store
    _memory_agents_query = query_agent


def init_meta_evaluator(evaluator):
    global _meta_evaluator
    _meta_evaluator = evaluator


def get_rag_pipeline():
    return _rag_pipeline


def get_safety_manager():
    return _safety_manager


def get_memory_manager():
    return _memory_manager


def get_medication_manager():
    return _medication_manager


def get_memory_agents_store():
    return _memory_agents_store


def get_memory_agents_query():
    return _memory_agents_query


def get_meta_evaluator():
    return _meta_evaluator
