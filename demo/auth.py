import hashlib, hmac, os, re
from datetime import datetime, timezone
import db

ITER = 600_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def hash_password(pw, iterations=ITER):
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"

def verify_password(pw, stored):
    try:
        algo, it, salt, dk = stored.split("$")
        new = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it))
        return algo == "pbkdf2_sha256" and hmac.compare_digest(new, bytes.fromhex(dk))
    except Exception:
        return False

_DUMMY = hash_password("timing-equaliser")

def validate_signup(name, email, pw, pw2):
    errs = []
    if len(name.strip()) < 2: errs.append("Enter your name.")
    if not EMAIL_RE.match(email.strip()): errs.append("Enter a valid email address.")
    if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        errs.append("Password must be at least 8 characters and contain a letter and a digit.")
    if pw != pw2: errs.append("Passwords do not match.")
    return errs

def register(name, email, pw, pw2):
    errs = validate_signup(name, email, pw, pw2)
    if errs: return None, errs
    try:
        return db.create_user(name, email, hash_password(pw)), []
    except ValueError:
        return None, ["An account with this email already exists."]

def authenticate(email, pw):
    row = db.get_user_row(email)
    if row is None:
        verify_password(pw, _DUMMY)                      # same cost whether or not the account exists
        return None, "Invalid email or password."
    lock = row["locked_until"]
    if lock and lock > datetime.now(timezone.utc):
        mins = int((lock - datetime.now(timezone.utc)).total_seconds() // 60) + 1
        return None, f"Too many failed attempts. Try again in about {mins} minute(s)."
    if verify_password(pw, row["password_hash"]):
        db.record_login_success(row["id"])
        return {k: row[k] for k in ("id", "name", "email", "role")}, ""
    db.record_login_failure(row["id"])
    return None, "Invalid email or password."
