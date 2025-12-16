# =============================================================================
# Agent module for RAG Assistant
# =============================================================================
# This file creates an AI agent that can DECIDE when to search the database.
# Instead of always retrieving passages, the LLM chooses when retrieval helps.
# =============================================================================

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from backend.database import RAGDatabase

class RAGAgent:
    def __init__(self, db: RAGDatabase, model_name: str, max_iter: int):
        self.db = db
        self.model_name = model_name
        self.max_iter = max_iter
        self.last_sources = []  # We'll store retrieved passages here for the UI

    def create_tool(self):
        # ---------------------------------------------------------------------
        # The @tool decorator transforms this function into something the
        # LLM can call. The docstring is CRUCIAL—it's what the LLM reads
        # to decide whether and how to use this tool.
        # ---------------------------------------------------------------------
        @tool("Query RAG Database")
        def query_rag_db(query: str) -> str:
            """Search the vector database containing customized texts.
            
            Args:
                query: Search query about topic.
                
            Returns:
                Relevant passages from the database
            """
            try:
                results = self.db.query(query)
                
                if results:
                    # Store sources for UI display
                    self.last_sources.extend(results)
                    
                    # Format passages for the LLM to read
                    passages = [row["text"] for row in results]
                    return "\n\n---\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(passages)])
                else:
                    return "No relevant passages found."
                    
            except Exception as e:
                return f"Error querying database: {str(e)}"
        
        return query_rag_db

    # TO DO: Update the ask() function
    def ask(self, question: str) -> dict:
        """
        Ask a question to the agent.
        
        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        # Reset sources for this query
        self.last_sources = []
        
        # TO DO: Create the LLM instance
        llm = LLM(model = self.model_name)

        # TO DO: Call the database tool (e.g. the function above)
        query_tool = self.create_tool()
        
        # UPDATE THESE (AGENTS) FOR FINAL PROJECT
        agent = Agent(
            role='Video Game Content Assistant',
            goal='Answer questions about video games using the provided database of reviews, facts, and forum discussions.',
            backstory='You are an expert assistant with access to a curated database of video game reviews, factual information, and community discussions.',
            tools=[query_tool],
            llm=llm,
            verbose=False, # Shows what the agent is doing
            allow_delegation=False, # Does not create sub-agents
            max_iter=self.max_iter # Limits tool calls
        )
        
        # TO DO: Create the task
        task = Task(
            description=f"""
        You are answering a user's question.

        Question: {question}

        You may use the tool "Query RAG Database" if it helps.
        After using the tool (if you use it), you MUST write a final answer to the user.

        Rules:
        - Always output a final answer in plain English.
        - Do NOT stop after tool usage.
        - If the tool returns "No relevant passages found.", still answer using general knowledge
        and briefly note that nothing relevant was found in the database.
        """,
            agent=agent,
            expected_output="A final answer to the user (not tool logs)."
        )
        
        # TO DO: Create the Crew and run it
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
            max_rpm=20
        )
        
        result = crew.kickoff()
        
        # Extract final text robustly across crewai versions
        if isinstance(result, str):
            answer_text = result
        elif hasattr(result, "raw") and isinstance(result.raw, str):
            answer_text = result.raw
        elif hasattr(result, "output") and isinstance(result.output, str):
            answer_text = result.output
        else:
            answer_text = str(result)
        
        # Returns the answer and sources
        return {
            "answer": answer_text.strip(),
            "sources": self.last_sources.copy()
        }
