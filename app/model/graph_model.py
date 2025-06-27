from neo4j import GraphDatabase

# driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "password"))
driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password"))


def create_relationship(tx, head, rel, tail):
    tx.run(
        """
        MERGE (a:Entity {name: $head})
        MERGE (b:Entity {name: $tail})
        MERGE (a)-[r:%s]->(b)
        """ % rel.upper(),
        head=head,
        tail=tail
    )

def init_graph(edge_list: list[list[str]]):
    with driver.session() as session:
        for relation in edge_list:
            if len(relation) >= 3:
                head = relation[0]
                rel = relation[1]
                tail = relation[2]
                session.write_transaction(create_relationship, head, rel, tail)
