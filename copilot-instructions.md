# Workspace Instructions

## Deployment Workflow

Whenever you create or modify files in this workspace:

1. Run the narrowest relevant validation for the change.
2. Check `git status --short` and review the diff.
3. If there are changes, run:

   ```bash
   git add -A
   git commit -m "update model"
   git push origin main
   ```

4. Confirm that the push completed successfully and report the commit hash.

Do not create an empty commit when there are no changes. Do not skip the push after a successful commit unless authentication, network access, merge conflicts, or another concrete Git error blocks it; report the exact blocker instead.

## Streamlit Deployment

The Streamlit entry point is `kospi_model.py`. After pushing, verify that Streamlit Cloud is configured for the `main` branch and redeploy or reboot the app. Use the deployment commit displayed in the app to confirm that Cloud is running the pushed revision.
