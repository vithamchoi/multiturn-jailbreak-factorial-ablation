# Third-party code and data, not redistributed here

The experiments depend on the following. We link them rather than bundle them,
so that their own licences and versions govern.

| Component | Where | Used for |
|---|---|---|
| Groq API | groq.com | the inference endpoint for target and judge calls |
| `llama-3.1-8b-instant` | Meta, served by Groq | the target model under attack |
| `llama-3.3-70b-versatile` | Meta, served by Groq | judge 1 |
| `qwen/qwen3-32b` | Alibaba, served by Groq | judge 2 |

Record the version or commit of each before re-running; the scripts do not pin them.
