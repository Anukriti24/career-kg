"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from career_kg import graph_store
from career_kg.recommender import Recommender

st.set_page_config(page_title="Career recommender")


@st.cache_resource(show_spinner="Connecting to the database …")
def get_recommender():
    return Recommender(graph_store.connect())


def dot(graph):
    fills = {"occupation": "#dbe4f0", "matched": "#cfe8d5", "inferred": "#f5e2b8", "missing": "#eeeeee", "user": "#dbe4f0"}
    lines = ["digraph G { rankdir=LR; node [style=filled, shape=box, fontname=Helvetica, fontsize=11]; edge [fontsize=9];"]
    for n in graph["nodes"]:
        lines.append(f'"{n["id"]}" [label="{n["label"].replace(chr(34), chr(39))}", fillcolor="{fills[n["kind"]]}"];')
    for e in graph["edges"]:
        style = "style=dashed" if e["kind"] == "infers" else ""
        lines.append(f'"{e["from_"]}" -> "{e["to"]}" [label="{e["label"]}", {style}];')
    return "\n".join(lines) + "}"


try:
    rec = get_recommender()
except Exception as exc:
    st.error(f"Cannot reach the database: {exc}")
    st.stop()

st.title("Career recommender")
st.write("Pick the skills you have and see which occupations fit. Data: O*NET 30.0, stored as a knowledge graph in Neo4j.")

chosen = st.multiselect("Your skills", [c["name"] for c in rec._catalog], accept_new_options=True,
                        placeholder="Type to search, e.g. Python, SQL, Tableau", key="skills")

with st.expander("Options"):
    k = st.slider("Number of results", 3, 20, 8)
    zmin, zmax = st.select_slider("Job zone (1 = little preparation, 5 = extensive)", [1, 2, 3, 4, 5], value=(1, 5))
    inference = st.checkbox("Use graph inference (similar tools count partly)", value=True)

if not chosen:
    st.stop()

res = rec.recommend(chosen, k=k, zone_min=zmin, zone_max=zmax, inference=inference)
for r in res["resolved"]:
    if r["typed_as"]:
        st.caption(f"\"{r['typed_as']}\" was understood as {r['name']}.")
for u in res["unknown"]:
    hint = f" Did you mean: {', '.join(u['suggestions'])}?" if u["suggestions"] else ""
    st.warning(f"Unknown skill \"{u['input']}\".{hint}")
if not res["results"]:
    st.info("No occupations found for these skills.")
    st.stop()

st.subheader("Recommended occupations")
st.dataframe([{"Occupation": r["title"], "Match": f"{r['coverage']:.0%}", "You have": len(r["matched"]),
               "Partly covered": len(r["inferred"]), "Missing": len(r["missing"])} for r in res["results"]],
             hide_index=True, width="stretch")
st.caption("Match = share of the occupation's weighted skill requirements that you cover. Rare skills weigh more than common ones.")

by_title = {f"{r['title']} ({r['code']})": r for r in res["results"]}
r = by_title[st.selectbox("Details for", list(by_title))]
occ = rec.occupation(r["code"])
st.write(occ["description"])

names = lambda items: ", ".join(m["name"] for m in items) or "none"
st.markdown(f"**Skills you have:** {names(r['matched'])}")
if r["inferred"]:
    st.markdown("**Partly covered through similar skills:**")
    for m in r["inferred"][:10]:
        st.markdown(f"- {m['name']}: {m['why']}")
st.markdown(f"**Skills you are missing:** {names(r['missing'][:12])}")
if occ["predicted"]:
    st.markdown("**Skills the model expects but O*NET does not list:** " + ", ".join(p["skill"] for p in occ["predicted"][:6]))
if occ["moves"]:
    st.markdown("**Possible career moves:** " + ", ".join(m["title"] for m in occ["moves"][:6]))

with st.expander("Show graph"):
    st.graphviz_chart(dot(rec.explain_graph(r["code"], chosen, inference)), width="stretch")

if res["learn"]:
    st.subheader("Worth learning next")
    st.write(", ".join(x["name"] for x in res["learn"]))
