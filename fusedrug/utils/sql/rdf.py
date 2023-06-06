from rdflib import Graph, Literal
import os


def SQLfromTTL(db_dir: str, db_name: str, ttl_filepath: str) -> None:
    uri = Literal("sqlite:///" + os.path.join(db_dir, db_name + ".sqlite3"))
    graph = Graph("SQLAlchemy")
    graph.open(uri, create=True)
    tmpgraph = Graph()
    tmpgraph.parse(ttl_filepath, format="ttl")
    graph += tmpgraph
    graph.close()
