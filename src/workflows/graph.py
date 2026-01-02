from langgraph.graph import StateGraph, END
from src.workflows.state import WorkflowState
from src.app.config import Settings

# Import Agents
from src.agents.router_mapper_agent import RouterMapperAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.code_writer_agent import CodeWriterAgent
from src.agents.report_writer_agent import ReportWriterAgent

# فرض می‌کنیم تابع اجرا را در ماژولی جداگانه (مثلا src.execution.utils) دارید
# اگر ندارید، می‌توانید همان تابع execute_generated_code که در ui2.py نوشتیم را به یک فایل مشترک ببرید و ایمپورت کنید
from src.execution.executor import execute_generated_code 

def build_graph(settings: Settings):
    # --- 1. Initialize Agents ---
    router_agent = RouterMapperAgent(settings.router_model, settings.db_path)
    planner_agent = PlannerAgent(settings.planner_model)
    
    # ایجنت نویسنده در دو حالت مختلف
    analyst_agent = CodeWriterAgent(settings.code_writer_model, mode="analysis")
    visualizer_agent = CodeWriterAgent(settings.code_writer_model, mode="visualization")
    
    reporter_agent = ReportWriterAgent(settings.report_writer_model)

    workflow = StateGraph(WorkflowState)

    # --- 2. Define Logic Nodes (Functions to wrap execution) ---
    
    def run_analysis_execution(state: WorkflowState):
        """نود اجرای کد آنالیز"""
        result = execute_generated_code(
            state.analysis_code, 
            settings.db_path, 
            settings.artifacts_dir, 
            state.questionnaire_id
        )
        # آپدیت استیت با نتیجه اجرا
        return {
            "execution": result, 
            "analysis_output": result["output"],
            # شمارنده تلاش را می‌توان در State مدیریت کرد (اختیاری)
        }

    def run_viz_execution(state: WorkflowState):
        """نود اجرای کد نمودار"""
        result = execute_generated_code(
            state.viz_code, 
            settings.db_path, 
            settings.artifacts_dir, 
            state.questionnaire_id
        )
        return {
            "execution": result, 
            "viz_artifacts": result["artifacts"]
        }

    # --- 3. Add Nodes to Graph ---
    workflow.add_node("router", router_agent.run)
    workflow.add_node("planner", planner_agent.run)
    
    # فاز ۱: آنالیز
    workflow.add_node("analyst", analyst_agent.run)
    workflow.add_node("execute_analysis", run_analysis_execution)
    
    # فاز ۲: نمودار
    workflow.add_node("visualizer", visualizer_agent.run)
    workflow.add_node("execute_viz", run_viz_execution)
    
    # فاز ۳: گزارش
    workflow.add_node("reporter", reporter_agent.run)

    # --- 4. Define Edges & Flow ---
    workflow.set_entry_point("router")

    # شرط روتر (مرتبط یا نامرتبط)
    def route_decision(state):
        if not state.is_related:
            return END
        return "planner"

    workflow.add_conditional_edges("router", route_decision)
    workflow.add_edge("planner", "analyst")
    workflow.add_edge("analyst", "execute_analysis")

    # 

    # شرط حلقه اصلاح آنالیز (Analyst Feedback Loop)
    def analysis_check(state):
        execution = state.execution
        # اگر اجرا موفق بود -> برو مرحله بعد (Visualizer)
        if execution.get("success"):
            return "visualizer"
        # اگر خطا داشت -> برگرد به Analyst (تا با دیدن خطا کد را اصلاح کند)
        # نکته: برای جلوگیری از حلقه بی‌نهایت، باید یک مکانیسم max_retries در State داشته باشید
        # در اینجا ساده‌سازی شده است:
        return "analyst"

    workflow.add_conditional_edges("execute_analysis", analysis_check)

    workflow.add_edge("visualizer", "execute_viz")

    # شرط حلقه اصلاح نمودار (Visualizer Feedback Loop)
    def viz_check(state):
        execution = state.execution
        if execution.get("success"):
            return "reporter"
        return "visualizer" # برگرد و اصلاح کن

    workflow.add_conditional_edges("execute_viz", viz_check)

    workflow.add_edge("reporter", END)
    
    return workflow.compile()