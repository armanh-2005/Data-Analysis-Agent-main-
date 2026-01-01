from langgraph.graph import StateGraph, END
from src.workflows.state import WorkflowState
from src.app.config import Settings

# Import your agents
from src.agents.router_mapper_agent import RouterMapperAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.code_writer_agent import CodeWriterAgent
from src.agents.code_reviewer_agent import CodeReviewerAgent
from src.agents.quality_review_agent import QualityReviewAgent
from src.agents.report_writer_agent import ReportWriterAgent

def build_graph(settings: Settings):
    # Initialize Agents
    router_agent = RouterMapperAgent(settings.router_model, settings.db_path)
    planner_agent = PlannerAgent(settings.planner_model)
    coder_agent = CodeWriterAgent(settings.code_writer_model)
    reviewer_agent = CodeReviewerAgent(settings.code_reviewer_model)
    # ... init others ...

    workflow = StateGraph(WorkflowState)

    # Define Nodes
    workflow.add_node("router", router_agent.run)
    workflow.add_node("planner", planner_agent.run)
    workflow.add_node("coder", coder_agent.run)
    workflow.add_node("reviewer", reviewer_agent.run)
    
    # Define Logic (Edges)
    workflow.set_entry_point("router")
    
    # Conditional logic for Router
    def route_decision(state):
        if not state.is_related:
            return END
        return "planner"

    workflow.add_conditional_edges("router", route_decision)
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "reviewer")

    # Conditional logic for Reviewer (Loop back if unsafe)
    def review_decision(state):
        if not state.code_review.get("approved"):
            return "coder" # Loop back!
        return END # Or move to execution/report

    workflow.add_conditional_edges("reviewer", review_decision)
    
    return workflow.compile()