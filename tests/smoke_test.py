"""Smoke test: unit-checks the stats logic AND drives the server over the real
MCP protocol (stdio) exactly like an AI client would."""
import asyncio, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server as S

def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    assert cond, name

print("== unit: decision tree ==")
check("2 indep, non-normal -> Mann-Whitney", S._recommend("continuous","independent",2,"non_normal","unknown")[0]=="mann_whitney")
check("2 indep, normal, unequal var -> Welch", S._recommend("continuous","independent",2,"normal","unequal")[0]=="welch_t")
check("2 indep, normal, equal var -> Student", S._recommend("continuous","independent",2,"normal","equal")[0]=="independent_t")
check("3 indep, normal -> one-way ANOVA", S._recommend("continuous","independent",3,"normal","equal")[0]=="one_way_anova")
check("3 indep, non-normal -> Kruskal-Wallis", S._recommend("continuous","independent",3,"non_normal","unknown")[0]=="kruskal_wallis")
check("paired, non-normal -> Wilcoxon signed", S._recommend("continuous","paired",2,"non_normal","unknown")[0]=="wilcoxon_signed")
check("paired, normal -> paired t", S._recommend("continuous","paired",2,"normal","unknown")[0]=="paired_t")
check("correlation, non-normal -> Spearman", S._recommend("continuous","correlation",2,"non_normal","unknown")[0]=="spearman")
check("association nominal -> chi-square", S._recommend("nominal","association",2,"unknown","unknown")[0]=="chi_square_independence")
check("count outcome -> count regression", S._recommend("count","independent",2,"unknown","unknown")[0]=="count_regression")

print("== unit: inverse normal & power ==")
check("_norm_ppf(0.975)≈1.95996", abs(S._norm_ppf(0.975)-1.959964)<1e-3)
check("_norm_ppf(0.80)≈0.84162", abs(S._norm_ppf(0.80)-0.841621)<1e-3)
ss = S.plan_sample_size("two_means", 0.5)         # d=0.5, α.05, power.8
check("d=0.5 two-means -> ~63/group (GPower 64)", ss["n_per_group"] in range(62,66))
ssr = S.plan_sample_size("correlation", 0.3)
check("r=0.3 -> ~84 total (GPower 84)", ssr["total_n"] in range(82,88))

print("== unit: p-value interpreter guards ==")
check("non-sig does not claim null true", "does NOT prove" in S.interpret_result(0.20)["interpretation"])
check("sig rejects null", S.interpret_result(0.001)["significant"] is True)

print("\n== integration: drive server over MCP stdio (as an AI client) ==")
async def mcp_roundtrip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable,
                                   args=[os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"server.py")])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            names = [t.name for t in tools.tools]
            check("server advertises 6 tools", len(names) == 6)
            check("recommend_test is callable", "recommend_test" in names)
            res = await sess.call_tool("recommend_test",
                {"outcome_type":"continuous","design":"independent","n_groups":2,"normality":"non_normal"})
            text = res.content[0].text if res.content else ""
            check("call returns Mann-Whitney", "Mann" in text)
            res2 = await sess.call_tool("plan_sample_size", {"comparison":"two_means","effect_size":0.5})
            check("power tool returns n_per_group", "n_per_group" in (res2.content[0].text if res2.content else ""))
            print("\n  advertised tools:", ", ".join(names))
            print("  sample call → recommend_test(continuous, independent, 2 groups, non-normal):")
            print("   ", (text[:220] + "…") if len(text) > 220 else text)

asyncio.run(mcp_roundtrip())
print("\nALL PASS ✅")
