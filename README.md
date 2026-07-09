# academic-stats-advisor · MCP server

An **MCP (Model Context Protocol) server** that lets an AI assistant — ChatGPT, Claude, Claude Code, Cursor, or any MCP client — **directly call** a statistician's decision logic instead of guessing. Point it at a study design and it returns the correct test, its assumptions, the SPSS menu path and R code, an APA reporting template, and an a-priori power analysis.

Built by [Gan Lin](https://github.com/ganlin770). Dependency-light (only `mcp`), so it runs with a single command and deploys anywhere. (Repo `academic-stats-mcp`; server/package name `academic-stats-advisor`.)

> ⚠️ Decision-support for people who already know some statistics. It does not run your data or replace a statistician — always verify assumptions against your own dataset.

## Tools

| Tool | What the AI can call it for |
|------|------------------------------|
| `recommend_test` | "What statistical test should I use?" → test + why + assumptions + SPSS path + R code + APA template |
| `check_assumptions` | Assumptions of a given test, how to check each, and what to do if violated |
| `interpret_result` | Turn a p-value / effect size into a correct, APA-style conclusion (guards the classic mistakes) |
| `plan_sample_size` | A-priori power analysis — required *n* for two means, paired means, two proportions, or a correlation |
| `normality_guide` | How to decide **and report** normality the right way (the #1 thing students get wrong) |
| `list_supported_tests` | Everything the advisor knows, with SPSS menu paths |

Covers the full classic tree: one-sample / independent / paired **t-tests**, **Welch**, **Mann–Whitney**, **Wilcoxon**, one-way / **Welch** / **repeated-measures ANOVA**, **Kruskal–Wallis**, **Friedman**, **Pearson/Spearman**, **chi-square / Fisher / McNemar / goodness-of-fit**, and **Poisson/NB** for counts.

## Run it (zero setup)

The server file carries its own dependencies (PEP 723), so [`uv`](https://docs.astral.sh/uv/) needs nothing installed:

```bash
uv run server.py            # stdio  (for Claude Desktop / Claude Code / Cursor)
MCP_HTTP=1 uv run server.py # HTTP    (remote endpoint at http://localhost:8000/mcp)
```

## Use it in Claude Code / Claude Desktop (local, stdio)

Add to your MCP config (Claude Desktop: `claude_desktop_config.json`; Claude Code: `claude mcp add`):

```json
{
  "mcpServers": {
    "academic-stats-advisor": {
      "command": "uv",
      "args": ["run", "/absolute/path/to/academic-stats-mcp/server.py"]
    }
  }
}
```

Claude Code one-liner:

```bash
claude mcp add academic-stats-advisor -- uv run /absolute/path/to/academic-stats-mcp/server.py
```

Then just ask: *"My outcome is a continuous score, two independent groups, the data are skewed — what test, and how do I report it?"* — the model calls `recommend_test` and answers with the real decision logic.

## Use it in ChatGPT / Claude.ai (remote, HTTP)

Deploy the HTTP transport, then add the resulting `https://…/mcp` URL as a **custom connector**.

- **Render (free):** this repo includes `render.yaml` + `Dockerfile` → New ▸ Blueprint → pick the repo. Endpoint: `https://<service>.onrender.com/mcp`.
- **Docker anywhere:** `docker build -t stats-mcp . && docker run -p 8000:8000 stats-mcp` → `http://<host>:8000/mcp`.
- **Any Python host:** `python server.py --http`, and set `PUBLIC_HOST=<your-domain>` so the Host check allows it (without it, DNS-rebinding protection is off so it still works behind any proxy).

## License

MIT — see [LICENSE](LICENSE).
