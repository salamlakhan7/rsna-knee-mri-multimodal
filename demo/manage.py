"""Admin helper. Usage: python manage.py promote user@example.com | demote user@example.com | users"""
import sys, db
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd in ("promote", "demote") and len(sys.argv) == 3:
    print("done" if db.set_role(sys.argv[2], "admin" if cmd == "promote" else "user") else "no such user")
elif cmd == "users":
    for u in db.all_users(): print(u["id"], u["email"], u["role"], u["n_tests"], "tests")
else:
    print(__doc__)
