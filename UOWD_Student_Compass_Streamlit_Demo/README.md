# UOWD Student Compass — Streamlit live demo

This is a small cloud-deployable demo using the expanded **89-passage** UOWD corpus, the same TF-IDF/MiniLM retrieval code, and the recorded 100-question benchmark results. It is an **independent academic prototype**, not an official UOWD service. Facts are manually curated, not refreshed automatically.

## Deploy without installing Python locally

1. Extract this ZIP on your computer.
2. Open https://github.com/new and create a new repository called `uowd-student-compass-demo` (public is easiest).
3. Open the new repository and choose **Add file → Upload files**. Upload the contents of the extracted folder, keeping `src/`, `data/`, and `.streamlit/` folders intact. Commit the files.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app → Yup, I have an app**. Select your repository and `main` branch. Set **Main file path** to `app.py`.
6. In **Advanced settings**, select Python **3.12** (or leave the default if it is 3.12), then deploy.
7. Open your generated `*.streamlit.app` URL and try **MiniLM** and **TF-IDF**. Model weights are downloaded on first MiniLM use.

## Recommended 60–90-second demonstration

- MiniLM: `Is there a free UOWD bus from Dubai Internet City Metro?` — show retrieved source URL.
- MiniLM: `What scholarships are available to Master's students?` — show top results.
- MiniLM: `What is my university GPA?` — show the targeted privacy refusal.
- Sidebar: Show benchmark results: MiniLM vs TF-IDF, 82% vs 61% Precision@1.

## If deployment fails

- Open Community Cloud **Manage app → Logs** to see the build/runtime error.
- If MiniLM cannot load, switch to the TF-IDF model. The fallback does not use PyTorch.
- Keep the executed Colab notebook open as a reliable presentation backup.

## Notebook comparison and limitations

The sidebar scores were computed previously in the executed expanded evaluation notebook on 100 manually authored questions and 89 passages. They are **not live-computed** by this app. This demo does not authenticate to UOWD, fetch student GPA, use a generative LLM, or automatically crawl the site. Similarity is not factual correctness.

## Project structure

- `app.py` — Streamlit interface and targeted GPA safeguard
- `src/retrieval.py` — same baseline and MiniLM retrieval logic as the evaluated notebook
- `data/uowd_sources.jsonl` — 89 manually curated UOWD facts
- `requirements.txt` — cloud dependencies, CPU-only PyTorch
- `.streamlit/config.toml` — theme
