import networkx as nx
import json
import re
import threading
from typing import List, Dict, Any
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class LocalGraphRAG:
    def __init__(self, model, tokenizer, lock: threading.Lock = None):
        self.graph = nx.DiGraph()
        self.model = model
        self.tokenizer = tokenizer
        self.lock = lock or threading.Lock()
        self.entity_pattern = re.compile(r'\[\s*\{.*?\}\s*\]', re.DOTALL)

    def extract_knowledge(self, text_chunk: str):
        """
        Extracts entities and relationships from a text chunk using the LLM.
        Adds them to the graph.
        """
        prompt = f"""
        [INSTRUCTION]
        Identify the top entities (People, Organizations, Concepts) and their relationships in the text below.
        Output STRICTLY as a JSON list of triples. Do not add any other text.
        Format: [{{"source": "Entity A", "target": "Entity B", "relation": "connected_to"}}]
        
        Text:
        {text_chunk[:2000]} 
        [/INSTRUCTION]
        """
        
        messages = [{"role": "user", "content": prompt}]
        prompt_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate with low temp for structured output
        # Protect with lock to avoid Metal concurrency crashes
        with self.lock:
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt_input, 
                max_tokens=512, 
                sampler=make_sampler(0.1),
                verbose=False
            )
        
        self._parse_and_add_triples(response)

    def _parse_and_add_triples(self, response: str):
        """
        Parses the LLM response and updates the graph.
        """
        try:
            # Extract JSON list from response
            match = self.entity_pattern.search(response)
            if match:
                json_str = match.group(0)
                triples = json.loads(json_str)
                
                for triple in triples:
                    source = triple.get("source")
                    target = triple.get("target")
                    relation = triple.get("relation")
                    
                    if source and target and relation:
                        self.graph.add_edge(source, target, relation=relation)
                        logger.debug(f"Graph Edge Added: {source} --[{relation}]--> {target}")
            else:
                logger.debug("No JSON found in extraction response.")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from extraction: {response[:100]}...")
        except Exception as e:
            logger.error(f"Graph update failed: {e}")

    def query_subgraph(self, query: str, depth: int = 1) -> str:
        """
        Retrieves a subgraph related to the query keywords.
        Returns a formatted string context.
        """
        # 1. Extract Keywords (Simple heuristic or LLM)
        # For speed, let's try to match existing nodes in the query
        # This is a simple "Entity Linking"
        
        found_nodes = []
        lower_query = query.lower()
        
        for node in self.graph.nodes():
            if str(node).lower() in lower_query:
                found_nodes.append(node)
        
        if not found_nodes:
            return "No related knowledge graph connections found."
            
        logger.debug(f"Graph Query found nodes: {found_nodes}")
        
        # 2. Traverse
        subgraph_edges = set()
        for node in found_nodes:
            # Get edges within depth
            # Outgoing
            if depth >= 1:
                for neighbor in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    relation = edge_data.get("relation", "related_to")
                    subgraph_edges.add(f"{node} --({relation})--> {neighbor}")
                    
                    # Depth 2 (simple implementation)
                    if depth >= 2:
                        for next_neighbor in self.graph.successors(neighbor):
                            next_edge = self.graph.get_edge_data(neighbor, next_neighbor)
                            next_rel = next_edge.get("relation", "related_to")
                            subgraph_edges.add(f"{neighbor} --({next_rel})--> {next_neighbor}")

            # Incoming (optional, but good for context)
            if depth >= 1:
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph.get_edge_data(predecessor, node)
                    relation = edge_data.get("relation", "related_to")
                    subgraph_edges.add(f"{predecessor} --({relation})--> {node}")

        if not subgraph_edges:
            return "No relevant connections found in knowledge graph."

        return "Knowledge Graph Connections:\n" + "\n".join(subgraph_edges)

    def visualize(self, output_file="graph.html"):
        """
        Generates an HTML visualization of the graph using PyVis.
        """
        try:
            from pyvis.network import Network
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
            
            # Convert nx graph to pyvis
            net.from_nx(self.graph)
            
            # Save
            net.save_graph(output_file)
            return output_file
        except ImportError:
            logger.warning("PyVis not installed. Skipping visualization.")
            return None
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None
