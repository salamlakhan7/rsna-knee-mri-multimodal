# KneeScope: Streamlit demo for the RSNA knee MRI project

Pages: Home, Technology, Results, Live Explore (login required), My History, Contact, Log in / Sign up, Account, Admin.

## Run locally (SQLite)
```
pip install -r requirements.txt
streamlit run app.py
```
A `kneeapp.db` SQLite file is created automatically. Put `resnet18_attn.pt` in `models/` to enable the MRI tab.

## Production (Neon PostgreSQL)
1. Create a Neon project and copy the connection string.
2. Set it as the `DATABASE_URL` secret (see `.streamlit/secrets.toml.example`) or as an environment variable.
3. Tables are created on first start. Deploy on Streamlit Community Cloud (or Hugging Face Spaces if memory is short).
4. Make yourself admin after signing up: `DATABASE_URL=... python manage.py promote you@example.com`

## Security notes
PBKDF2-SHA256 password hashes (600k iterations, per-user salt), same error for unknown email and wrong password, lockout for 10 minutes after 5 failed logins, zip-slip and size checks on uploads, no uploaded files stored. No email verification and no persistent login across browser refresh.

## Tests
`python -m pytest tests`
