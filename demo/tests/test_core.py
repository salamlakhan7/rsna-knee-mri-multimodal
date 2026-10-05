import os, sys, io, tempfile, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_dir = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_dir}/test.db"
import numpy as np, pytest, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import auth, db, config, mri_pipeline as mri
from report_labeler import label_report

def test_password_hash_roundtrip():
    h = auth.hash_password("abc12345")
    assert auth.verify_password("abc12345", h) and not auth.verify_password("abc12346", h)
    assert h != auth.hash_password("abc12345")                      # salted
    assert not auth.verify_password("x", "garbage")

def test_signup_validation():
    assert auth.validate_signup("Ab", "ab@x.com", "abcdefg1", "abcdefg1") == []
    assert len(auth.validate_signup("A", "bad", "short", "other")) == 4
    assert auth.validate_signup("Ab", "ab@x.com", "onlyletters", "onlyletters")      # needs a digit

def test_register_and_login_flow():
    u, e = auth.register("Mr AB", "AB@Example.com", "secret123", "secret123")
    assert u and u["email"] == "ab@example.com" and u["role"] == "user" and not e
    dup, e = auth.register("Other", "ab@example.com", "secret123", "secret123")
    assert dup is None and "already exists" in e[0]
    ok, err = auth.authenticate("ab@example.com", "secret123"); assert ok and ok["id"] == u["id"]
    bad, m1 = auth.authenticate("ab@example.com", "wrongpass1")
    none, m2 = auth.authenticate("nobody@example.com", "wrongpass1")
    assert bad is None and none is None and m1 == m2              # no account enumeration

def test_lockout_after_failures():
    auth.register("Lock Me", "lock@example.com", "secret123", "secret123")
    for _ in range(5): auth.authenticate("lock@example.com", "nope12345")
    user, msg = auth.authenticate("lock@example.com", "secret123")
    assert user is None and "Too many failed attempts" in msg

def test_history_is_per_user_and_ordered():
    a, _ = auth.register("User A", "a@example.com", "secret123", "secret123")
    b, _ = auth.register("User B", "b@example.com", "secret123", "secret123")
    db.log_run(a["id"], "report", {"language": "en", "n_chars": 10}, {"detected": ["ACL"]}, 120)
    db.log_run(a["id"], "mri", {"n_series_used": 3}, {"scores": {"ACL": 0.4}}, 900)
    assert [r["test_type"] for r in db.user_runs(a["id"])] == ["mri", "report"]       # newest first
    assert db.user_runs(b["id"]) == []
    row = next(x for x in db.all_users() if x["email"] == "a@example.com")
    assert row["n_tests"] == 2 and row["last_test"] is not None

def test_role_promotion():
    auth.register("Boss", "boss@example.com", "secret123", "secret123")
    assert db.set_role("boss@example.com", "admin") and not db.set_role("none@example.com", "admin")
    u, _ = auth.authenticate("boss@example.com", "secret123"); assert u["role"] == "admin"

@pytest.mark.parametrize("name", list(config.SAMPLES))
def test_sample_reports_match_expected_findings(name):
    s = config.SAMPLES[name]; out = label_report(s["text"])
    assert sorted(l for l, v in out.items() if v) == sorted(s["findings"])

def test_labeler_negation():
    assert not any(label_report("ACL is intact. No fracture is seen. No joint effusion.").values())

# ------------------------------------------------------------- MRI pipeline on synthetic DICOM
def _write_series(root, uid, plane_iop, desc, n, size=128):
    d = os.path.join(root, uid); os.makedirs(d)
    rng = np.random.RandomState(0)
    for i in range(n):
        fm = FileMetaDataset(); fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage; fm.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.Rows = ds.Columns = size; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
        ds.PixelRepresentation = 0; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SeriesInstanceUID = uid; ds.SeriesDescription = desc; ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = plane_iop; ds.ImagePositionPatient = [float(i) * 3, 0.0, 0.0]
        ds.PixelData = (100 + 10 * i + rng.randint(0, 40, (size, size))).astype(np.uint16).tobytes()
        pydicom.dcmwrite(os.path.join(d, f"{generate_uid()}.dcm"), ds, enforce_file_format=True)

def _make_zip():
    src = tempfile.mkdtemp()
    SAG, COR, AX = [0, 1, 0, 0, 0, -1], [1, 0, 0, 0, 0, -1], [1, 0, 0, 0, 1, 0]
    _write_series(src, "s_sag_t2", SAG, "SAG T2 FS", 24); _write_series(src, "s_sag_t1", SAG, "SAG T1", 30)
    _write_series(src, "s_cor_pd", COR, "COR PD FS", 20); _write_series(src, "s_ax_pd", AX, "AX PD FATSAT", 18)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for dp, _, fs in os.walk(src):
            for f in fs: z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), src))
    buf.seek(0); return buf

def test_mri_zip_pipeline():
    dest = tempfile.mkdtemp(); mri.extract_zip(_make_zip(), dest)
    series = mri.scan_series(dest)
    assert {e["plane"] for e in series.values()} == {"Sagittal", "Coronal", "Axial"}
    picks = mri.pick_series(series)
    assert picks[0] == "s_sag_t2"                     # fluid-sensitive sagittal wins over the longer T1 series
    arr, used = mri.build_study(series, picks)
    assert arr.shape == (3, 16, 192, 192) and arr.dtype == np.uint8 and arr.max() > 0 and len(used) == 3

def test_zip_slip_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("../evil.txt", "x")
    buf.seek(0)
    with pytest.raises(ValueError): mri.extract_zip(buf, tempfile.mkdtemp())

def test_postgres_url_conversion(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db?sslmode=require")
    assert db.get_url().startswith("postgresql+psycopg2://u:p@host/db")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host/db")
    assert db.get_url().startswith("postgresql+psycopg2://")
