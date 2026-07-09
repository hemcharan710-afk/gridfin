"""GridFin — grid-based analysis of financial documents.

A question is broken into a matrix of documents (rows) by sub-questions (columns).
Each cell is answered on its own, scoped to a single document, and any figure that
needs computing goes through a deterministic engine rather than the model.
"""

__version__ = "0.1.0"
