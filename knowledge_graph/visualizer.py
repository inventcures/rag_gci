"""
Graph Visualizer for Palliative Care Knowledge Graph

Generates visualization data for graph rendering in the admin UI.
Supports Cytoscape.js format for interactive graph visualization.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NodeColor(Enum):
    """Color scheme for different node types."""
    SYMPTOM = "#FF6B6B"       # Red
    MEDICATION = "#4ECDC4"    # Teal
    CONDITION = "#45B7D1"     # Blue
    TREATMENT = "#96CEB4"     # Green
    SIDE_EFFECT = "#FFEAA7"   # Yellow
    BODY_PART = "#DDA0DD"     # Plum
    DEFAULT = "#95A5A6"       # Gray


class EdgeColor(Enum):
    """Color scheme for different relationship types."""
    TREATS = "#2ECC71"        # Green
    CAUSES = "#E74C3C"        # Red
    SIDE_EFFECT_OF = "#F39C12"  # Orange
    MANAGES = "#3498DB"       # Blue
    ALLEVIATES = "#9B59B6"    # Purple
    AFFECTS = "#1ABC9C"       # Turquoise
    DEFAULT = "#7F8C8D"       # Gray


@dataclass
class VisualizationData:
    """Container for graph visualization data."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    layout: str = "cose"
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cytoscape(self) -> Dict[str, Any]:
        """Convert to Cytoscape.js format."""
        return {
            "elements": {
                "nodes": self.nodes,
                "edges": self.edges
            },
            "layout": {"name": self.layout},
            "style": self.style
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "layout": self.layout,
            "metadata": self.metadata
        }


class GraphVisualizer:
    """
    Generates visualization data from Neo4j query results.

    Features:
    - Cytoscape.js format output
    - Color coding by node/edge type
    - Interactive graph layouts
    - Subgraph extraction

    Usage:
        visualizer = GraphVisualizer()

        # From query results
        viz_data = visualizer.from_query_results(results)

        # For Cytoscape.js
        cytoscape_data = viz_data.to_cytoscape()
    """

    def __init__(self):
        """Initialize visualizer with default styles."""
        self.node_colors = {t.name: t.value for t in NodeColor}
        self.edge_colors = {t.name: t.value for t in EdgeColor}

        # Default Cytoscape style
        self.default_style = [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "data(color)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "12px",
                    "width": "60px",
                    "height": "60px",
                    "border-width": "2px",
                    "border-color": "#333"
                }
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "line-color": "data(color)",
                    "target-arrow-color": "data(color)",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "font-size": "10px",
                    "text-rotation": "autorotate"
                }
            },
            {
                "selector": ":selected",
                "style": {
                    "border-width": "4px",
                    "border-color": "#000"
                }
            }
        ]

    def from_query_results(
        self,
        results: List[Dict[str, Any]],
        node_key: str = "name",
        node_type_key: str = "type"
    ) -> VisualizationData:
        """
        Create visualization from query results.

        Args:
            results: Neo4j query results
            node_key: Key for node label
            node_type_key: Key for node type

        Returns:
            VisualizationData object
        """
        nodes = []
        edges = []
        seen_nodes = set()
        node_counter = 0
        edge_counter = 0

        for row in results:
            # Extract node information
            for key, value in row.items():
                if isinstance(value, dict) and node_key in value:
                    # This is a node
                    node_name = value.get(node_key, f"Node_{node_counter}")
                    node_type = value.get(node_type_key, "DEFAULT")

                    if node_name not in seen_nodes:
                        seen_nodes.add(node_name)
                        nodes.append(self._create_node(
                            id=f"n{node_counter}",
                            label=node_name,
                            node_type=node_type,
                            properties=value
                        ))
                        node_counter += 1

                elif isinstance(value, str):
                    # Could be a simple node reference
                    if value not in seen_nodes and key in ["medication", "symptom", "condition", "treatment"]:
                        seen_nodes.add(value)
                        node_type = key.upper() if key else "DEFAULT"
                        nodes.append(self._create_node(
                            id=f"n{node_counter}",
                            label=value,
                            node_type=node_type
                        ))
                        node_counter += 1

            # Try to extract relationships from row
            edge = self._extract_edge_from_row(row, seen_nodes)
            if edge:
                edge["data"]["id"] = f"e{edge_counter}"
                edges.append(edge)
                edge_counter += 1

        return VisualizationData(
            nodes=nodes,
            edges=edges,
            layout="cose",
            style=self.default_style,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )

    def from_entities_and_relationships(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> VisualizationData:
        """
        Create visualization from extracted entities and relationships.

        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries

        Returns:
            VisualizationData object
        """
        nodes = []
        edges = []
        node_ids = {}

        # Create nodes
        for i, entity in enumerate(entities):
            node_id = f"n{i}"
            node_name = entity.get("name", f"Entity_{i}")
            node_type = entity.get("type", "DEFAULT")

            node_ids[node_name] = node_id

            nodes.append(self._create_node(
                id=node_id,
                label=node_name,
                node_type=node_type,
                properties=entity.get("properties", {})
            ))

        # Create edges
        for i, rel in enumerate(relationships):
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")

            source_id = node_ids.get(source_name)
            target_id = node_ids.get(target_name)

            if source_id and target_id:
                edges.append(self._create_edge(
                    id=f"e{i}",
                    source=source_id,
                    target=target_id,
                    label=rel.get("relationship", "RELATED"),
                    rel_type=rel.get("relationship", "DEFAULT")
                ))

        return VisualizationData(
            nodes=nodes,
            edges=edges,
            layout="cose",
            style=self.default_style,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )

    def _create_node(
        self,
        id: str,
        label: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Cytoscape node."""
        color = self.node_colors.get(
            node_type.upper(),
            NodeColor.DEFAULT.value
        )

        return {
            "data": {
                "id": id,
                "label": label,
                "type": node_type,
                "color": color,
                **(properties or {})
            }
        }

    def _create_edge(
        self,
        id: str,
        source: str,
        target: str,
        label: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Cytoscape edge."""
        color = self.edge_colors.get(
            rel_type.upper(),
            EdgeColor.DEFAULT.value
        )

        return {
            "data": {
                "id": id,
                "source": source,
                "target": target,
                "label": label,
                "color": color,
                **(properties or {})
            }
        }

    def _extract_edge_from_row(
        self,
        row: Dict[str, Any],
        seen_nodes: set
    ) -> Optional[Dict[str, Any]]:
        """Try to extract edge from a query result row."""
        # Look for common relationship patterns
        source = None
        target = None
        rel_type = "RELATED"

        # Check for medication -> symptom pattern
        if "medication" in row and "symptom" in row:
            source = row["medication"]
            target = row["symptom"]
            rel_type = row.get("relationship", "TREATS")

        # Check for condition -> symptom pattern
        elif "condition" in row and "symptom" in row:
            source = row["condition"]
            target = row["symptom"]
            rel_type = row.get("relationship", "CAUSES")

        # Check for side_effect -> medication pattern
        elif "side_effect" in row and "medication" in row:
            source = row["side_effect"]
            target = row["medication"]
            rel_type = "SIDE_EFFECT_OF"

        if source and target and source in seen_nodes and target in seen_nodes:
            # Find node IDs (simplified - would need proper mapping in production)
            return self._create_edge(
                id="",  # Will be set by caller
                source=source,
                target=target,
                label=rel_type,
                rel_type=rel_type
            )

        return None

    def create_subgraph(
        self,
        center_node: str,
        results: List[Dict[str, Any]],
        max_nodes: int = 50
    ) -> VisualizationData:
        """
        Create a subgraph visualization centered on a node.

        Args:
            center_node: Name of the center node
            results: Query results containing neighbors
            max_nodes: Maximum nodes to include

        Returns:
            VisualizationData for subgraph
        """
        nodes = []
        edges = []
        seen = set()

        # Add center node
        nodes.append(self._create_node(
            id="center",
            label=center_node,
            node_type="CENTER",
            properties={"is_center": True}
        ))
        seen.add(center_node)

        # Add neighbors
        for i, row in enumerate(results[:max_nodes - 1]):
            neighbor_name = row.get("name", row.get("m", {}).get("name", f"Neighbor_{i}"))
            rel_type = row.get("rel_type", "RELATED")

            if neighbor_name not in seen:
                seen.add(neighbor_name)
                node_type = row.get("type", "DEFAULT")

                nodes.append(self._create_node(
                    id=f"n{i}",
                    label=neighbor_name,
                    node_type=node_type
                ))

                edges.append(self._create_edge(
                    id=f"e{i}",
                    source="center",
                    target=f"n{i}",
                    label=rel_type,
                    rel_type=rel_type
                ))

        return VisualizationData(
            nodes=nodes,
            edges=edges,
            layout="concentric",  # Radial layout from center
            style=self.default_style,
            metadata={
                "center_node": center_node,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )

    def generate_html(self, viz_data: VisualizationData) -> str:
        """
        Generate standalone HTML with embedded Cytoscape visualization.

        Args:
            viz_data: Visualization data

        Returns:
            HTML string
        """
        import json

        cytoscape_data = viz_data.to_cytoscape()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        #cy {{
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
        }}
        .legend {{
            padding: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <h2>Knowledge Graph Visualization</h2>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: {NodeColor.SYMPTOM.value}"></div>
            <span>Symptom</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {NodeColor.MEDICATION.value}"></div>
            <span>Medication</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {NodeColor.CONDITION.value}"></div>
            <span>Condition</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {NodeColor.TREATMENT.value}"></div>
            <span>Treatment</span>
        </div>
    </div>
    <div id="cy"></div>
    <script>
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {json.dumps(cytoscape_data["elements"])},
            layout: {json.dumps(cytoscape_data["layout"])},
            style: {json.dumps(cytoscape_data["style"])}
        }});
    </script>
</body>
</html>
"""
        return html
