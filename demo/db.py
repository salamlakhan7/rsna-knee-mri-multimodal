"""Database layer. SQLite in development, Neon PostgreSQL in production (set DATABASE_URL)."""
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional
from sqlalchemy import create_engine, String, Integer, DateTime, ForeignKey, JSON, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.exc import IntegrityError

def _now(): return datetime.now(timezone.utc)
def aware(dt):
    return None if dt is None else (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))

class Base(DeclarativeBase): pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

class Run(Base):
    __tablename__ = "test_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    test_type: Mapped[str] = mapped_column(String(20))
    input_info: Mapped[dict] = mapped_column(JSON, default=dict)
    results: Mapped[dict] = mapped_column(JSON, default=dict)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

def get_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
        except Exception:
            url = None
    url = url or "sqlite:///kneeapp.db"
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg2://" + url[len(prefix):]
    return url

@lru_cache(maxsize=4)
def get_engine(url=None):
    url = url or get_url()
    kw = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
    else:
        kw["pool_recycle"] = 300          # Neon suspends idle connections
    eng = create_engine(url, **kw)
    Base.metadata.create_all(eng)
    return eng

def _u(u): return dict(id=u.id, name=u.name, email=u.email, role=u.role)

def create_user(name, email, password_hash, role="user"):
    with Session(get_engine()) as s:
        u = User(name=name.strip(), email=email.strip().lower(), password_hash=password_hash, role=role)
        s.add(u)
        try:
            s.commit()
        except IntegrityError:
            s.rollback(); raise ValueError("email already registered")
        s.refresh(u); return _u(u)

def get_user_row(email):
    with Session(get_engine()) as s:
        u = s.scalar(select(User).where(User.email == email.strip().lower()))
        if not u: return None
        return dict(_u(u), password_hash=u.password_hash, locked_until=aware(u.locked_until))

def record_login_success(user_id):
    with Session(get_engine()) as s:
        u = s.get(User, user_id); u.failed_attempts = 0; u.locked_until = None; u.last_login = _now(); s.commit()

def record_login_failure(user_id, max_attempts=5, lock_minutes=10):
    with Session(get_engine()) as s:
        u = s.get(User, user_id); u.failed_attempts = (u.failed_attempts or 0) + 1
        if u.failed_attempts >= max_attempts:
            u.locked_until = _now() + timedelta(minutes=lock_minutes); u.failed_attempts = 0
        s.commit()

def set_role(email, role):
    with Session(get_engine()) as s:
        u = s.scalar(select(User).where(User.email == email.strip().lower()))
        if not u: return False
        u.role = role; s.commit(); return True

def log_run(user_id, test_type, input_info, results, duration_ms=0):
    with Session(get_engine()) as s:
        r = Run(user_id=user_id, test_type=test_type, input_info=input_info, results=results, duration_ms=int(duration_ms))
        s.add(r); s.commit(); return r.id

def user_runs(user_id, limit=300):
    with Session(get_engine()) as s:
        rows = s.scalars(select(Run).where(Run.user_id == user_id).order_by(Run.created_at.desc()).limit(limit)).all()
        return [dict(id=r.id, created_at=aware(r.created_at), test_type=r.test_type, input_info=r.input_info,
                     results=r.results, duration_ms=r.duration_ms) for r in rows]

def all_users():
    with Session(get_engine()) as s:
        q = (select(User, func.count(Run.id), func.max(Run.created_at))
             .outerjoin(Run, Run.user_id == User.id).group_by(User.id).order_by(User.created_at.desc()))
        return [dict(_u(u), created_at=aware(u.created_at), last_login=aware(u.last_login),
                     n_tests=n, last_test=aware(last)) for u, n, last in s.execute(q).all()]
