"""Challenge public boundaries and receipts rather than restating implementation."""
import importlib.util, json, shutil, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import verify_companion as v
from contracts import unit01, unit03, unit06, unit10

class NumericalGuards(unittest.TestCase):
    def test_integer_cannot_drift(self):
        with self.assertRaises(ValueError):unit01.compare_rows([{'n':'10'}],[{'n':'10.000000000001'}],'counts')
    def test_ids_and_order_are_exact(self):
        with self.assertRaises(ValueError):unit01.compare_rows([{'row_id':'A'},{'row_id':'B'}],[{'row_id':'B'},{'row_id':'A'}],'ids')
    def test_nonfinite_prediction_fails(self):
        with self.assertRaises(ValueError):unit03.compare([{'prediction':'1.2'}],[{'prediction':'NaN'}],'bad')
    def test_solver_must_converge(self):
        with self.assertRaises(ValueError):unit06.compare([{'kkt':'0'}],[{'kkt':'0.01'}],'kkt')
    def test_undefined_auc_is_not_zero(self):
        with self.assertRaises(ValueError):unit10.compare([{'auc':''}],[{'auc':'0'}],'undefined')
    def test_secret_scan_key(self):
        value=b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----'
        import re
        self.assertTrue(any(re.search(p,value) for p in v.SECRET_PATTERNS))
    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):v.safe(Path(t),'../outside.csv')
    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'real.csv').write_text('x');(root/'alias.csv').symlink_to(root/'real.csv')
            with self.assertRaises(ValueError):v.safe(root,'alias.csv')
    def test_missing_receipt_is_not_success(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(v.receipt_valid(Path(t)/'missing.json','x',{},Path(t)))
    def test_public_inventory_is_self_contained(self):
        inventory=v.boundary(quiet=True)
        self.assertGreater(len(inventory),250)
        self.assertFalse(any('manuscript' in Path(n).parts for n in inventory))

if __name__=='__main__':unittest.main()
