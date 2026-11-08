---
name: refresh-guide
description: Re-capture the app screenshots in docs/screenshots/ and rebuild the presenter guide PDF (docs/Bangkok_Flood_Simulator_Presenter_Guide.pdf). Use after UI changes or before a presentation.
disable-model-invocation: true
---
Rebuild the presenter guide from the live app. The scripts need Playwright and Google Chrome, which are already
installed locally.

1. Start the app in the background on the port the script expects:
   `streamlit run app.py --server.port 8599 --server.headless true`. Wait until `http://localhost:8599` answers.
2. `python scripts/capture_screenshots.py` writes `docs/screenshots/*.png`.
3. Stop the Streamlit server.
4. `python scripts/build_guide_pdf.py` renders `docs/presenter_guide.html` to the PDF.
5. `git status --short docs/` shows what changed. List the screenshots that changed. Open (Read) 2–3 of the changed
   PNGs and check they show the app properly, not an error page or a half-loaded map.
6. If `docs/presenter_guide.html` mentions numbers that the new screenshots contradict (flooded area, people
   affected, pass counts), list them for the user. Don't rewrite the guide text without asking.

Don't commit. Report what changed and leave committing to the user.
