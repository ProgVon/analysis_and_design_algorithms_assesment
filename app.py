import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from matchers import ALGORITHMS, benchmark, detect

SOURCE = """Photosynthesis is the process by which green plants convert light energy into chemical energy. \
It takes place mainly in the chloroplasts of leaf cells. During this process plants absorb carbon dioxide \
from the air and water from the soil. The energy of sunlight is used to split water molecules and release oxygen. \
The glucose produced is stored as starch or used immediately for growth."""

SUSPECT = """Plants make their own food in a remarkable way. Photosynthesis is the process by which green plants \
convert light energy into chemical energy. Most of it happens in the leaves. During this process plants absorb \
carbon dioxide from the air and water from the soil. The glucose produced is stored as starch for later use."""

st.set_page_config(page_title="Plagiarism Detection", layout="wide")
st.title("Plagiarism Detection with String Matching")
st.caption("Naive · Rabin-Karp · Knuth-Morris-Pratt")

detect_tab, bench_tab = st.tabs(["Detect plagiarism", "Benchmark algorithms"])


def doc_input(col, label, default, key):
    up = col.file_uploader(f"{label} (.txt)", type="txt", key=f"up_{key}")
    text = up.read().decode("utf-8", errors="replace") if up else default
    return col.text_area(label, text, height=220, key=f"ta_{key}_{up.name if up else ''}")


with detect_tab:
    c1, c2 = st.columns(2)
    source = doc_input(c1, "Source document", SOURCE, "src")
    suspect = doc_input(c2, "Suspect document", SUSPECT, "sus")
    n = st.slider("Phrase length (words per pattern)", 2, 12, 5)

    if not source.strip() or not suspect.strip():
        st.warning("Enter text in both documents.")
        st.stop()

    results = {name: detect(source, suspect, n, name) for name in ALGORITHMS}
    r = results["KMP"]
    phrases = {m["phrase"] for m in r["matches"]}
    assert all({m["phrase"] for m in x["matches"]} == phrases for x in results.values())

    m1, m2, m3 = st.columns(3)
    m1.metric("Similarity", f"{r['similarity']:.1f}%")
    m2.metric("Matched phrases", len(r["matches"]))
    m3.metric("Suspect words", len(r["words"]))

    st.subheader("Copied text (highlighted)")
    # words are [a-z0-9'] only, so this HTML cannot be injected into
    html = " ".join(f"<mark>{w}</mark>" if c else w for w, c in zip(r["words"], r["covered"]))
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Algorithm comparison on these documents")
    cmp = pd.DataFrame([{"Algorithm": k, "Time (ms)": v["time"] * 1000, "Matches": len(v["matches"]),
                         "Similarity %": round(v["similarity"], 1)} for k, v in results.items()])
    a, b = st.columns([1, 1])
    a.dataframe(cmp, hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.bar(cmp["Algorithm"], cmp["Time (ms)"], color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("Time (ms)")
    b.pyplot(fig)

    if r["matches"]:
        st.subheader("Matched phrases")
        st.dataframe(pd.DataFrame([{"Phrase": m["phrase"], "Suspect sentence": m["suspect_sentence"] + 1,
                                    "Source sentence(s)": ", ".join(str(s + 1) for s in m["source_sentences"])}
                                   for m in r["matches"]]), hide_index=True, use_container_width=True)

        st.subheader("Sentence match network")
        st.caption("Left: suspect sentences (S). Right: source sentences (O). Edge width = shared phrases.")
        G = nx.Graph()
        sus_nodes = [f"S{i + 1}" for i in range(len(r["suspect_sentences"]))]
        src_nodes = [f"O{i + 1}" for i in range(len(r["source_sentences"]))]
        G.add_nodes_from(sus_nodes)
        G.add_nodes_from(src_nodes)
        for m in r["matches"]:
            for s in m["source_sentences"]:
                u, v = f"S{m['suspect_sentence'] + 1}", f"O{s + 1}"
                G.add_edge(u, v, weight=G.get_edge_data(u, v, {"weight": 0})["weight"] + 1)
        pos = nx.bipartite_layout(G, sus_nodes)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(G))))
        nx.draw_networkx(G, pos, ax=ax, node_size=700, font_color="white",
                         node_color=["#d62728" if x in sus_nodes else "#1f77b4" for x in G],
                         width=[0.6 * d["weight"] for _, _, d in G.edges(data=True)])
        ax.axis("off")
        st.pyplot(fig)
        with st.expander("Sentence index"):
            for i, s in enumerate(r["suspect_sentences"]):
                st.write(f"**S{i + 1}:** {s}")
            for i, s in enumerate(r["source_sentences"]):
                st.write(f"**O{i + 1}:** {s}")
    else:
        st.success("No copied phrases found at this phrase length.")


with bench_tab:
    st.write("Times each algorithm on generated text. **Random** is English-like text; "
             "**Worst case** (`aaa…a` searched for `aa…ab`) is where Naive degrades to O(n·m).")
    kind = st.radio("Text type", ["random", "worst"], horizontal=True,
                    format_func=lambda k: "Random words" if k == "random" else "Worst case")
    c1, c2 = st.columns(2)
    sizes = c1.multiselect("Document sizes (chars)", [1000, 5000, 10000, 20000, 50000, 100000],
                           [1000, 5000, 10000, 20000])
    lengths = c2.multiselect("Pattern lengths (chars)", [5, 10, 20, 50, 100, 200], [5, 20, 50, 100])

    if st.button("Run benchmark", type="primary") and sizes and lengths:
        with st.spinner("Benchmarking..."):
            df = pd.DataFrame(benchmark(sorted(sizes), sorted(lengths), kind))
        st.session_state["bench"] = df

    df = st.session_state.get("bench")
    if df is not None:
        g1, g2 = st.columns(2)
        for col, x, fixed in ((g1, "Pattern length", "Document size"), (g2, "Document size", "Pattern length")):
            f = df[df[fixed] == df[fixed].max()]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for name, grp in f.groupby("Algorithm"):
                ax.plot(grp[x], grp["Time (ms)"], marker="o", label=name)
            ax.set(xlabel=x, ylabel="Time (ms)", title=f"Time vs {x.lower()} ({fixed.lower()} = {f[fixed].max()})")
            ax.legend()
            ax.grid(alpha=0.3)
            col.pyplot(fig)
        st.dataframe(df.pivot_table(index=["Document size", "Pattern length"], columns="Algorithm",
                                    values="Time (ms)").round(3), use_container_width=True)
