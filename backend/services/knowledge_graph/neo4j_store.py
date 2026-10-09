"""Knowledge-graph storage. The only module that talks to Neo4j.

Schema. Every node carries project_id, so every read is scoped by that property.
Neo4j Community has one database, so project isolation is enforced here, in each query.

    (:Document {id, project_id, title, source_type, updated_at})
    (:<Label> {name, project_id, ...properties})     Label from labels.safe_entity_label
    (:Document)-[:MENTIONS]->(:<Label>)

Entities are merged on (name, project_id). The same name in two projects is two nodes.
"""
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.services.knowledge_graph.labels import safe_entity_label

_driver = None
_driver_lock = threading.Lock()


def _get_driver():
    """One shared driver per process. Created on first use."""
    global _driver
    with _driver_lock:
        if _driver is None:
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        return _driver


def close_driver() -> None:
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None


def _clean_props(props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Neo4j properties must be scalars. Store anything else as JSON text.

    name and project_id are identity keys, so they are never overwritten here.
    """
    clean: Dict[str, Any] = {}
    for key, value in (props or {}).items():
        if key in ("name", "project_id") or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, default=str)
    return clean


def write_document_entities(doc_id: str, title: str, source_type: str,
                            project_id: int, entities: List[Dict]) -> int:
    """Write one document and its entities for one project. Returns the entity count."""
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    with _get_driver().session() as session:
        session.run(
            """
            MERGE (d:Document {id: $doc_id})
            SET d.project_id = $project_id, d.title = $title,
                d.source_type = $source_type, d.updated_at = $now
            """,
            doc_id=doc_id, project_id=project_id, title=title,
            source_type=source_type, now=now,
        )
        for entity in entities:
            name = str(entity.get("name") or "Unknown")
            # Label is interpolated only after the allowlist check
            label = safe_entity_label(entity.get("type"))
            session.run(
                f"""
                MATCH (d:Document {{id: $doc_id}})
                MERGE (e:{label} {{name: $name, project_id: $project_id}})
                SET e += $props
                MERGE (d)-[:MENTIONS]->(e)
                """,
                doc_id=doc_id, name=name, project_id=project_id,
                props=_clean_props(entity.get("properties")),
            )
            written += 1
    return written


def project_status(project_id: int) -> Dict[str, int]:
    """Document, entity and relationship counts for one project only."""
    with _get_driver().session() as session:
        docs = session.run(
            "MATCH (d:Document {project_id: $pid}) RETURN count(d) AS n",
            pid=project_id,
        ).single()["n"]
        entities = session.run(
            """
            MATCH (:Document {project_id: $pid})-[:MENTIONS]->(e)
            WHERE e.project_id = $pid
            RETURN count(DISTINCT e) AS n
            """,
            pid=project_id,
        ).single()["n"]
        relationships = session.run(
            "MATCH (:Document {project_id: $pid})-[r:MENTIONS]->() RETURN count(r) AS n",
            pid=project_id,
        ).single()["n"]
    return {
        "project_documents": docs,
        "total_entities": entities,
        "total_relationships": relationships,
    }


def project_stats(project_id: int) -> Dict[str, Any]:
    """Counts plus entities by label and the last document update, for one project."""
    counts = project_status(project_id)
    with _get_driver().session() as session:
        by_type = session.run(
            """
            MATCH (:Document {project_id: $pid})-[:MENTIONS]->(e)
            WHERE e.project_id = $pid
            UNWIND labels(e) AS label
            RETURN label, count(DISTINCT e) AS count
            ORDER BY count DESC
            """,
            pid=project_id,
        )
        entities_by_type = {record["label"]: record["count"] for record in by_type}
        last = session.run(
            "MATCH (d:Document {project_id: $pid}) RETURN max(d.updated_at) AS last",
            pid=project_id,
        ).single()["last"]
    return {
        "total_entities": counts["total_entities"],
        "total_relationships": counts["total_relationships"],
        "entities_by_type": entities_by_type,
        "last_updated": last,
    }


def project_graph(project_id: int, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """Documents and entities for one project, as nodes and edges."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []
    with _get_driver().session() as session:
        rows = session.run(
            """
            MATCH (d:Document {project_id: $pid})-[:MENTIONS]->(e)
            WHERE e.project_id = $pid
            RETURN d.id AS doc_id, d.title AS doc_title, labels(e) AS types, e.name AS name
            LIMIT $limit
            """,
            pid=project_id, limit=limit,
        )
        for row in rows:
            doc_key = f"doc:{row['doc_id']}"
            nodes.setdefault(doc_key, {"id": doc_key, "label": row["doc_title"], "kind": "document"})
            types = row["types"] or ["Entity"]
            entity_key = f"entity:{types[0]}:{row['name']}"
            nodes.setdefault(entity_key, {"id": entity_key, "label": row["name"],
                                          "kind": "entity", "type": types[0]})
            edges.append({"source": doc_key, "target": entity_key, "type": "MENTIONS"})
    return {"nodes": list(nodes.values()), "edges": edges}


def search_entities(project_id: int, query: str = "", entity_types: Optional[str] = None,
                    limit: int = 50) -> List[Dict[str, Any]]:
    """Entities in one project whose name or properties contain the query text."""
    cypher = """
    MATCH (d:Document {project_id: $pid})-[:MENTIONS]->(e)
    WHERE e.project_id = $pid
    """
    params: Dict[str, Any] = {"pid": project_id, "limit": limit}
    if query:
        cypher += " AND (e.name CONTAINS $query OR ANY(prop IN keys(e) WHERE toString(e[prop]) CONTAINS $query))"
        params["query"] = query
    if entity_types:
        # Labels cannot be parameters. Each one passes the allowlist first.
        conditions = " OR ".join(f"e:{safe_entity_label(t)}" for t in entity_types.split(","))
        cypher += f" AND ({conditions})"
    cypher += """
    RETURN e.name AS name, labels(e) AS types, properties(e) AS properties,
           count(DISTINCT d) AS document_count
    ORDER BY document_count DESC, name
    LIMIT $limit
    """
    with _get_driver().session() as session:
        rows = session.run(cypher, params)
        return [
            {
                "name": row["name"],
                "type": row["types"][0] if row["types"] else "Entity",
                "types": row["types"],
                "properties": {k: v for k, v in dict(row["properties"]).items()
                               if k not in ("project_id",)},
                "document_count": row["document_count"],
                "confidence": 1.0,
                "source": "knowledge_graph",
            }
            for row in rows
        ]


def related_entities(project_id: int, name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Entities that appear in the same documents as the named entity, within one project.

    The data has no entity-to-entity edges yet, so 'related' means co-mentioned.
    """
    with _get_driver().session() as session:
        rows = session.run(
            """
            MATCH (d:Document {project_id: $pid})-[:MENTIONS]->(e {name: $name, project_id: $pid})
            MATCH (d)-[:MENTIONS]->(other {project_id: $pid})
            WHERE other <> e
            RETURN other.name AS name, labels(other) AS types,
                   count(DISTINCT d) AS shared_documents
            ORDER BY shared_documents DESC, name
            LIMIT $limit
            """,
            pid=project_id, name=name, limit=limit,
        )
        return [
            {
                "name": row["name"],
                "type": row["types"][0] if row["types"] else "Entity",
                "shared_documents": row["shared_documents"],
            }
            for row in rows
        ]


def delete_project_graph(project_id: int) -> None:
    """Remove one project's documents, then any entities no document still mentions."""
    with _get_driver().session() as session:
        session.run("MATCH (d:Document {project_id: $pid}) DETACH DELETE d", pid=project_id)
        session.run(
            """
            MATCH (e)
            WHERE e.project_id = $pid AND NOT ()-[:MENTIONS]->(e)
            DETACH DELETE e
            """,
            pid=project_id,
        )
