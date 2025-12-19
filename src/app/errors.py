from __future__ import annotations


class AppError(Exception):
    # Base class for domain errors (intended, meaningful failures).
    pass


class OutOfScopeQuestion(AppError):
    # Raised when user question is not related to the questionnaire.
    pass


class MappingLowConfidence(AppError):
    # Raised when column mapping confidence is too low to proceed safely.
    pass


class UnsafeCodeError(AppError):
    # Raised when generated code violates safety policies (imports, filesystem, network, etc.).
    pass


class CodeExecutionError(AppError):
    # Raised when code execution fails in sandbox.
    pass


class DataNotSufficient(AppError):
    # Raised when data is insufficient for a reliable analysis.
    pass


class ImporterError(AppError):
    # Raised for importer-related failures (schema mismatch, unreadable file, etc.).
    pass
