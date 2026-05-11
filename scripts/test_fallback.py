from app.services.blog_pipeline.agent_section_planner import _fallback_plan
p = _fallback_plan("narrative", [], 2600)
for s in p:
    print(f"{s['role']:18s} -> {s['heading']}")
