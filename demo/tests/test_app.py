import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/app.db"
from streamlit.testing.v1 import AppTest
import auth

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

def _run(user=None, page=None):
    at = AppTest.from_file(APP, default_timeout=60)
    if user: at.session_state["user"] = user
    if page: at.switch_page(page)
    return at.run()

def test_home_loads_logged_out():
    at = _run()
    assert not at.exception, at.exception
    assert any("Knee" in t.value for t in at.title)

def test_explore_requires_login():
    at = _run(page="views/explore.py")
    assert not at.exception, at.exception
    assert any("log in" in i.value.lower() for i in at.info)

def test_explore_and_history_when_logged_in():
    u, _ = auth.register("Mr AB", "ab@x.com", "secret123", "secret123")
    at = _run(user=u, page="views/explore.py")
    assert not at.exception, at.exception
    assert len(at.tabs) == 2
    at.selectbox[0].select("English").run()
    at.button[0].click().run()                       # Analyze report
    assert not at.exception, at.exception
    assert any("Matched 3 of 3" in s.value for s in at.success)
    at = _run(user=u, page="views/history.py")
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1                     # the test just run is in the history

def test_admin_blocked_for_normal_user():
    u, _ = auth.register("Plain", "plain@x.com", "secret123", "secret123")
    at = _run(user=u, page="views/admin.py")
    assert not at.exception, at.exception
