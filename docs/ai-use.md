# AI use

LLMs are allowed during development and training preparation. Declare actual tools,
providers and model names where known, your access tier, and their role:

```toml
[ai_usage]
access_tier = "none_or_free"
tools = []
notes = ""
```

Supported tiers are `none_or_free`, `basic_subscription`, and
`professional_subscription`. Never submit credentials or private chat histories.
Access tiers are descriptive; they do not create percentile-based grading groups.

You must understand and maintain your submission. The course may ask you to explain
code or make small changes without an LLM. Final inference must use only the
submitted files and reference environment, with no external model service.
