# KneeScope: Streamlit demo

Pages: Home, Technology, Results, Live Explore (login required), My History, Contact, Log in / Sign up, Account, Admin.

## Run locally (SQLite)
```
pip install -r requirements.txt
streamlit run app.py
```
Put `resnet18_attn.pt` in `models/` to enable the MRI tab.

## Production
- **Frontend and app server:** Streamlit Community Cloud, main file `demo/app.py`, Python 3.12.
- **Database:** Neon PostgreSQL. Set the secret `DATABASE_URL` (see `.streamlit/secrets.toml.example`). Tables are created on first start.
- **Admin:** after signing up, run `DATABASE_URL=... python manage.py promote you@example.com`.

## Security notes
PBKDF2-SHA256 password hashes with per-user salt, identical error for unknown email and wrong password, 10-minute lockout after 5 failed logins,
zip-slip and size checks on uploads, uploaded files are not stored. No email verification; sessions end on browser refresh.

## Tests
`python -m pytest tests`
