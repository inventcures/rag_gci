from fastapi import Depends

_rag_pipeline = None
_safety_manager = None
_memory_manager = None
_medication_manager = None


def init_dependencies(rag_pipeline, safety_manager, memory_manager, medication_manager):
    global _rag_pipeline, _safety_manager, _memory_manager, _medication_manager
    _rag_pipeline = rag_pipeline
    _safety_manager = safety_manager
    _memory_manager = memory_manager
    _medication_manager = medication_manager


def get_rag_pipeline():
    return _rag_pipeline


def get_safety_manager():
    return _safety_manager


def get_memory_manager():
    return _memory_manager


def get_medication_manager():
    return _medication_manager
