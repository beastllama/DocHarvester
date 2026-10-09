"""Safe Cypher node labels.

Neo4j does not allow query parameters for labels, so any label that reaches a Cypher
string must come from this allowlist. Anything else becomes the generic 'Entity' label.
"""
import re

# Every entity type defined by the lens configs and the logistics templates
KNOWN_ENTITY_LABELS = frozenset({
    "Document", "Section", "Concept",
    "BusinessRule", "Process", "Decision", "Procedure", "Checklist", "Policy",
    "Product", "Market", "Strategy",
    "Equipment", "Route", "Facility", "Entity", "Topic", "Reference",
    "Carrier", "Shipment",
})

_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def safe_entity_label(raw) -> str:
    """Return a label that is safe to put in Cypher.

    Only names in KNOWN_ENTITY_LABELS pass. Anything else, including values from user
    input or LLM output, becomes 'Entity'.
    """
    candidate = str(raw or "").strip()
    if candidate in KNOWN_ENTITY_LABELS and _SAFE_LABEL.match(candidate):
        return candidate
    return "Entity"
