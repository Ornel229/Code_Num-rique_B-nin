"""Application Streamlit — Assistant « Lois numériques du Bénin ».

Lancer :  streamlit run app.py
Prérequis : dossier index/ produit par le notebook (section 8) + variable d'environnement GEMINI_API_KEY
            (ou secret Streamlit `GEMINI_API_KEY`).
"""
import os
from pathlib import Path

import streamlit as st

import rag_core as rc

st.set_page_config(page_title="Lois numériques du Bénin", page_icon="⚖️")

MAX_QUESTIONS = 15  # garde-fou par session : protège le quota de l'API pendant une démo publique
EXAMPLES = [
    "Quelqu'un a publié ma photo sur Facebook sans mon accord. Que dit la loi ?",
    "Que risque-t-on pour une injure ou une diffamation sur les réseaux sociaux ?",
    "Comment demander la suppression de mes données personnelles ?",
    "Quelle est la valeur juridique d'une signature électronique ?",
]


def _get_api_key():
    if os.getenv("GEMINI_API_KEY"):
        return True
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="Chargement de l'index et des modèles (la première fois : 1 à 2 minutes)…")
def load_engine():
    if not (Path("index") / "chunks.json").exists():
        return None
    rc.load_config("index")                 # mode, re-ranker et seuil calibrés lors de l'évaluation
    idx = rc.Index.load("index")
    rc.get_embedder()                       # préchargement : évite une première requête très lente
    if rc.CONFIG["DEFAULT_MODE"] == "hybrid_rerank":
        rc.get_cross_encoder()
    return idx


def render_assistant(msg, i):
    """Affiche une réponse : texte, reformulations, sources, mesures et retour utilisateur."""
    st.markdown(msg["content"])
    if msg.get("error"):
        st.warning("Un problème technique est survenu : " + msg["error"][:200])
    if msg.get("variants"):
        with st.expander("Reformulations utilisées pour la recherche"):
            for v in msg["variants"]:
                st.write("•", v)
    if msg.get("sources"):
        with st.expander(f"Sources citées ({len(msg['sources'])}) — à vérifier dans le texte de loi"):
            for s in msg["sources"]:
                score = f" — pertinence {s['score']:.2f}" if s.get("score") is not None else ""
                st.markdown(f"**{s['label']}**{score}")
                st.text(s["text"][:1200])
    st.caption(msg.get("caption", ""))
    if hasattr(st, "feedback"):
        def _log_feedback(i=i):
            rc.log_usage({"type": "feedback", "msg": i, "thumb": st.session_state.get(f"fb{i}")})
        st.feedback("thumbs", key=f"fb{i}", on_change=_log_feedback)


def main():
    st.title("⚖️ Assistant : lois numériques du Bénin")
    st.caption("Posez une question en langage courant. Les réponses s'appuient **uniquement** sur les textes de loi "
               "indexés et citent les articles. **Information générale : ce n'est pas un avis juridique.**")

    if not _get_api_key():
        st.error("Clé API absente. Définissez la variable d'environnement `GEMINI_API_KEY` "
                 "(ou le secret Streamlit du même nom), puis relancez.")
        st.stop()
    idx = load_engine()
    if idx is None:
        st.error("Dossier `index/` introuvable. Exécutez d'abord le notebook (section 8 : indexation) "
                 "et placez le dossier `index/` à côté de `app.py`.")
        st.stop()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("n_questions", 0)

    with st.sidebar:
        st.header("Paramètres")
        multiquery = st.toggle("Reformuler la question (multi-query)", value=True,
                               help="Un appel LLM supplémentaire qui rapproche votre question du vocabulaire juridique.")
        rc.CONFIG["K_FINAL"] = st.slider("Nombre d'extraits fournis au modèle", 2, 8, rc.CONFIG["K_FINAL"])
        st.markdown("**Exemples**")
        for e in EXAMPLES:
            if st.button(e, use_container_width=True):
                st.session_state["pending"] = e
        if st.button("Effacer la conversation"):
            st.session_state["messages"] = []
            st.rerun()
        with st.expander("À propos"):
            st.markdown(
                "- **Corpus** : Code du numérique (loi 2017-20) et Code pénal (2018). "
                "La loi 2020-35, qui modifie le Code du numérique, n'y figure pas.\n"
                "- Chaque question est traitée **indépendamment** (pas de mémoire de la conversation).\n"
                "- Si aucun extrait n'est assez pertinent, l'assistant **refuse de répondre** plutôt que d'inventer.\n"
                "- Ne saisissez pas de données personnelles sensibles : la question est envoyée à l'API du modèle.\n"
                f"- Modèle : `{rc.CONFIG['LLM_MODEL']}` (secours : {', '.join(rc.CONFIG['LLM_FALLBACKS'])}).")
        st.caption(f"Questions posées dans cette session : {st.session_state['n_questions']} / {MAX_QUESTIONS}")

    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg, i)

    limit_reached = st.session_state["n_questions"] >= MAX_QUESTIONS
    if limit_reached:
        st.info("Limite de démonstration atteinte pour cette session (elle protège le quota de l'API).")
    question = st.chat_input("Votre question…", disabled=limit_reached) or st.session_state.pop("pending", None)

    if question and not limit_reached:
        st.session_state["n_questions"] += 1
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les textes de loi…"):
                res = rc.answer(idx, question, multiquery=multiquery)
            gens = [c for c in res["llm_calls"] if c["kind"] == "generation"]
            parts = [f"Latence : {res['latency_s']:.1f} s"]
            if gens:
                parts.append(f"modèle : {gens[-1].get('model', '?')}" + (" (cache)" if gens[-1].get("cached") else ""))
            if res["refused"]:
                parts.append("aucun extrait assez pertinent → refus")
            msg = {"role": "assistant", "content": res["answer"], "sources": res["sources"],
                   "variants": res["variants"], "error": res["error"], "caption": " • ".join(parts)}
            st.session_state["messages"].append(msg)
            render_assistant(msg, len(st.session_state["messages"]) - 1)
        rc.log_usage({"type": "answer", "latency_s": round(res["latency_s"], 2), "refused": res["refused"],
                      "n_sources": len(res["sources"]), "error": bool(res["error"]),
                      "top": res["sources"][0]["label"] if res["sources"] else None,
                      "multiquery": multiquery})


main()
