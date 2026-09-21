# Middleware-Based Workflow Mapping

This prototype models a small deterministic state graph:

```mermaid
flowchart TD
    A["agent:react-orchestrator"] --> B["tool:langgraph-observability-workflow"]
    B --> C["node:plan"]
    C --> D["node:map_workflow"]
    D --> E["subagent:workflow-mapper"]
    E --> F["subagent-node:workflow-mapper.extract_state_graph"]
    F --> G["tool:extract_state_graph"]
    D --> H["node:assess_relay_fit"]
    H --> I["subagent:trace-validator"]
    I --> J["subagent-node:trace-validator.define_trace_checks"]
    J --> K["tool:define_trace_checks"]
    H --> L["node:recommend_next_steps"]
```

The important validation rule is hierarchy, not only event presence. The expected
shape is:

1. Custom Python React-loop orchestrator opens the root agent scope.
2. The LangGraph workflow runs as a tool scope under that root.
3. Each deterministic graph node opens a `node:*` Relay scope.
4. Each subagent is a compiled LangGraph subgraph invoked inside the parent node.
5. Each subagent graph opens a `subagent-node:*` Relay scope for its deterministic node.
6. Tools are nested under the subagent graph node that owns the call.
7. The emitted ATOF events contain parent UUIDs that prove the trace is not flat.
