# ============================================================================
# storage.py - Database and graph persistence
# ============================================================================
from sqlmodel import Session, create_engine, select, SQLModel
from typing import Optional
import networkx as nx
import json

from schemas import Entity, Timeline, SystemPrompt

class GraphStore:
    """Unified storage for entities, timelines, and graphs"""
    
    def __init__(self, db_url: str = "sqlite:///timepoint.db"):
        self.engine = create_engine(db_url)
        SQLModel.metadata.create_all(self.engine)
    
    def save_entity(self, entity: Entity) -> Entity:
        with Session(self.engine) as session:
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        with Session(self.engine) as session:
            statement = select(Entity).where(Entity.entity_id == entity_id)
            return session.exec(statement).first()
    
    def save_graph(self, graph: nx.Graph, timepoint_id: str):
        """Serialize NetworkX graph to database"""
        graph_dict = nx.to_dict_of_dicts(graph)
        with Session(self.engine) as session:
            timeline = session.exec(
                select(Timeline).where(Timeline.timepoint_id == timepoint_id)
            ).first()
            if timeline:
                timeline.graph_data = json.dumps(graph_dict)
                session.add(timeline)
                session.commit()
    
    def load_graph(self, timepoint_id: str) -> Optional[nx.Graph]:
        """Deserialize NetworkX graph from database"""
        with Session(self.engine) as session:
            timeline = session.exec(
                select(Timeline).where(Timeline.timepoint_id == timepoint_id)
            ).first()
            if timeline and timeline.graph_data:
                graph_dict = json.loads(timeline.graph_data)
                return nx.from_dict_of_dicts(graph_dict)
        return None
    
    def get_prompt(self, name: str) -> Optional[SystemPrompt]:
        with Session(self.engine) as session:
            return session.exec(
                select(SystemPrompt).where(SystemPrompt.name == name)
            ).first()
