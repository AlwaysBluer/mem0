import hashlib
import logging
import time

from mem0.memory.utils import format_entities

try:
    from gremlin_python.driver import client, serializer
except ImportError:
    raise ImportError("gremlin_python is not installed. Please install it using `pip install gremlinpython`")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = client.Client(
            url=config.graph_store.config.url,
            traversal_source=config.graph_store.config.traversal_source,
            username=config.graph_store.config.username,
            password=config.graph_store.config.password,
            message_serializer=config.graph_store.config.message_serializer
            if config.graph_store.config.message_serializer else serializer.GraphBinarySerializersV1(),
        )

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)

        self.node_label = "__Entity__" if self.config.graph_store.config.base_label else ""

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        return search_results

    def delete_all(self, filters):
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")

        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        query = f"g.V().hasLabel('{self.node_label}').has('user_id', '{user_id}')"
        if agent_id:
            query += f".has('agent_id', '{agent_id}')"
        query += ".sideEffect(__.bothE().drop()).drop()"
        self.graph.submit(query)

    def get_all(self, filters, limit=100):
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        query = (
            f"g.V().hasLabel(node_label)"
            f".has('user_id', user_id)"
        )
        if agent_id:
            query += f".has('agent_id', agent_id)"
        query += (
            f".outE()"
            f".where(inV().has('user_id', user_id)"
        )
        if agent_id:
            query += f".has('agent_id', agent_id)"
        query += (
            f")"
            f".project('source', 'relationship', 'target')"
            f".by(outV().values('name'))"
            f".by(label)"
            f".by(inV().values('name'))"
            f".limit(limit)"
        )
        params = {
            'node_label': self.node_label,
            'user_id': user_id,
            'limit': limit
        }
        if agent_id:
            params['agent_id'] = agent_id

        results = self.graph.submit(query, params).all().result()
        final_results = [
            {
                "source": result["source"],
                "relationship": result["relationship"],
                "target": result["target"]
            }
            for result in results
        ]
        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. "
                               f"If user message contains self reference such as 'I', 'me', 'my' etc. "
                               f"then use {filters['user_id']} as the source entity. Extract all the entities from"
                               f" the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """
        Establish relations among the extracted nodes.
        return List[Dict] stored `source` `destination` `relationship`
        """
        # Compose user identification string for prompt
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace(
                "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities["tool_calls"]:
            entities = extracted_entities["tool_calls"][0]["arguments"]["entities"]

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        base_query = "g.V().hasLabel(nodeLabel).has('user_id', user_id)"
        if filters.get("agent_id"):
            base_query += ".where(__.has('agent_id', agent_id))"

        outgoing_query = (
            f"{base_query}"
            ".as('source')"
            ".outE().as('relation')"
            ".inV().has('user_id', user_id)"
        )
        if filters.get("agent_id"):
            outgoing_query += ".where(__.has('agent_id', agent_id))"
        outgoing_query += (
            ".as('destination')"
            ".select('source', 'relation', 'destination')"
            ".by(project('id', 'name').by(id).by(values('name')))"
            ".by(project('label', 'id').by(label).by(id))"
            ".by(project('id', 'name').by(id).by(values('name')))"
        )

        incoming_query = (
            f"{base_query}"
            ".as('destination')"
            ".inE().as('relation')"
            ".outV().has('user_id', user_id)"
        )
        if filters.get("agent_id"):
            incoming_query += ".where(__.has('agent_id', agent_id))"
        incoming_query += (
            ".as('source')"
            ".select('source', 'relation', 'destination')"
            ".by(project('id', 'name').by(id).by(values('name')))"
            ".by(project('label', 'id').by(label).by(id))"
            ".by(project('id', 'name').by(id).by(values('name')))"
        )

        outgoing_query += ".dedup().limit(limit)"
        incoming_query += ".dedup().limit(limit)"

        query_params = {
            'nodeLabel': self.node_label,
            'user_id': filters['user_id'],
            'limit': limit
        }

        if 'agent_id' in filters:
            query_params['agent_id'] = filters['agent_id']

        outgoing_results = self.graph.submit(outgoing_query, query_params).all().result()
        incoming_results = self.graph.submit(incoming_query, query_params).all().result()

        all_results = []
        all_results.extend(outgoing_results)
        all_results.extend(incoming_results)

        unique_results = set()
        for r in all_results:
            source = r['source']
            relation = r['relation']
            destination = r['destination']

            result_tuple = (
                source.get('name', [None]),
                source.get('id', [None]),
                relation.get('label', [None]),
                relation.get('id', [None]),
                destination.get('name', [None]),
                destination.get('id', [None])
            )

            if result_tuple not in unique_results:
                unique_results.add(result_tuple)
                if len(unique_results) >= limit:
                    break

        result_relations = [
            {
                'source': r[0],
                'source_id': r[1],
                'relationship': r[2],
                'relation_id': r[3],
                'destination': r[4],
                'destination_id': r[5]
            }
            for r in unique_results
        ]

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            params = {
                'node_label': self.node_label,
                'source_name': source,
                'dest_name': destination,
                'user_id': user_id,
                'relationship': relationship
            }
            if agent_id:
                params['agent_id'] = agent_id

            check_relation_query = (
                f"g.V().hasLabel(node_label)"
                f".has('name', source_name)"
                f".has('user_id', user_id)"
            )
            if agent_id:
                check_relation_query += f".has('agent_id', agent_id)"
            check_relation_query += (
                f".outE(relationship)"
                f".where(inV().has('name', dest_name)"
                f".has('user_id', user_id)"
            )
            if agent_id:
                check_relation_query += f".has('agent_id', agent_id)"
            check_relation_query += (
                f")"
                f".project('source', 'target', 'relationship')"
                f".by(outV().values('name'))"
                f".by(inV().values('name'))"
                f".by(label())"
            )

            result = self.graph.submit(check_relation_query, params).all().result()

            # only when you find it that you can delete the edge and vertex.
            if result:
                results.extend(result)
                delete_edge_query = (
                    f"g.V().hasLabel(node_label)"
                    f".has('name', source_name)"
                    f".has('user_id', user_id)"
                )
                if agent_id:
                    delete_edge_query += f".has('agent_id', agent_id)"
                delete_edge_query += (
                    f".outE(relationship)"
                    f".where(inV().has('name', dest_name)"
                    f".has('user_id', user_id)"
                )
                if agent_id:
                    delete_edge_query += f".has('agent_id', agent_id)"
                delete_edge_query += f").drop()"
                self.graph.submit(delete_edge_query, params)

                source_query = (
                    f"g.V().has('name', source_name).has('user_id', user_id)"
                    f".choose(__.not(bothE()),"
                    f"__.drop(),"
                    f"__.property('mentions', __.values('mentions').math('_ - 1')))"
                )

                dest_query = (
                    f"g.V().has('name', dest_name).has('user_id', user_id)"
                    f".choose(__.not(bothE()),"
                    f"__.drop(),"
                    f"__.property('mentions', __.values('mentions').math('_ - 1')))"
                )

                self.graph.submit_async(source_query).result()
                self.graph.submit_async(dest_query).result()
        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        if filters is None or "user_id" not in filters:
            raise ValueError("filters can't be None and it must contain 'user_id'")
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []

        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]
            # types
            source_type = entity_type_map.get(source, "__User__")
            source_type = self.node_label if self.node_label else source_type
            destination_type = entity_type_map.get(destination, "__User__")
            destination_type = self.node_label if self.node_label else destination_type

            source_id = f"{source}_{user_id}"
            hash_source_id = hashlib.md5(source_id.encode()).hexdigest()
            source_id = f"{hash_source_id}_{source_id}"

            dest_id = f"{destination}_{user_id}"
            hash_dest_id = hashlib.md5(dest_id.encode()).hexdigest()
            dest_id = f"{hash_dest_id}_{dest_id}"

            query = f"""
            g.V('{source_id}').fold()
            .coalesce(
                unfold(),
                addV('{source_type}')
                    .property(id, '{source_id}')
                    .property('name', '{source}')
                    .property('user_id', '{user_id}')
                    .property('created', {self._current_timestamp()})
                    {f".property('agent_id', '{agent_id}')" if agent_id else ''}
            )
            .property('mentions', coalesce(values('mentions'), constant(0)).math('_ + 1'))
            .store('a')
            .V('{dest_id}').fold()
            .coalesce(
                unfold(),
                addV('{destination_type}')
                    .property(id, '{dest_id}')
                    .property('name', '{destination}')
                    .property('user_id', '{user_id}')
                    .property('created', {self._current_timestamp()})
                    {f".property('agent_id', '{agent_id}')" if agent_id else ''}
            )
            .property('mentions', coalesce(values('mentions'), constant(0)).math('_ + 1'))
            .as('b')
            .select('a').unfold()
            .coalesce(
                __.outE('{relationship}').where(inV().as('b'))
                    .property('mentions', coalesce(values('mentions'), constant(0)).math('_ + 1')),
                addE('{relationship}')
                    .to('b')
                    .property('mentions', 1)
                    .property('created', {self._current_timestamp()})
            )
            .outV() 
            .project('source', 'destination')
            .by('name')
            .by(select('b').values('name'))
            """

            # Execute the query
            result = self.graph.submit_async(query).result().all().result()

            if result:
                results.append({
                    "source": result[0]['source'],
                    "relationship": relationship,
                    "target": result[0]['destination']
                })

        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        raise NotImplementedError("embedding searching is not supported in Gremlin.")

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        raise NotImplementedError("embedding searching is not supported in Gremlin.")
    def _current_timestamp(self):
        return int(time.time())

    def drop_all_entities(self):
        self.graph.submit('g.E().drop()').all().result()
        self.graph.submit('g.V().drop()').all().result()

