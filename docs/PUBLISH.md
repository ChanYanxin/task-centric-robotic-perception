# Publish as a **new, standalone** GitHub repository

This local project is GitHub-ready, but a new GitHub repository must be created
in the GitHub UI if the connected automation lacks repository-creation scope.
Creating one costs nothing.

1. Visit `https://github.com/new` while logged into your account.
2. Name it `task-centric-robotic-perception`; select **Public** only if you intend to show it to recruiters.
3. **Do not** auto-create a README, licence or .gitignore (they are here already).
4. In a terminal from this project root, run:

```bash
git init -b main
git add .
git commit -m "Initial event-triggered edge perception prototype"
git remote add origin https://github.com/YOUR_USERNAME/task-centric-robotic-perception.git
git push -u origin main
```

Replace `YOUR_USERNAME`; GitHub may ask for browser sign-in, a configured
credential manager, SSH authentication, or a personal access token. **Never
paste a GitHub token in chat or commit it into the repo.** If this directory
already includes a `.git` history from the source ZIP/bundle, skip `git init`
and reuse `main`.

## Screening-ready demo order

- GitHub README animation (trigger/ROI/STOP).
- `docs/assets/traffic_comparison.png` (synthetic comparison with caveats).
- `src/tc_perception/sensing.py` and `pipeline.py` (core work).
- `tests/` and GitHub Actions (automated testing).
- `docs/EXPERIMENTS.md` (limitations and reproducibility).

## Research integrity

Do not claim this has been deployed to Jetson, Raspberry Pi, physical robots,
6G networks, real event cameras, or a real industrial dataset. Those have not
been validated. Likewise, the YOLO plugin exists in code but was not tested
in the provided offline environment.
