"""Real subprocess bootstrap regression tests; help/invalid budget only, no science."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]


class RecoveryCliTests(unittest.TestCase):
    def run_entry(self, *arguments):
        # A clean isolated interpreter intentionally ignores injected test sys.path,
        # PYTHONPATH and the unrelated working directory. No live launcher flags.
        environment = {k: v for k, v in os.environ.items() if not k.upper().startswith(('PYTHON','PREFECT_'))}
        with tempfile.TemporaryDirectory(prefix='C1_E32_cli_check_') as folder:
            return subprocess.run([sys.executable,'-I','-B',str(PACKAGE/'scripts/run_c1e32_recovery.py'),*arguments],
                cwd=folder,env=environment,capture_output=True,text=True,timeout=45)

    def test_isolated_help_from_unrelated_directory(self):
        result = self.run_entry('--help')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('--confirm-new-fits',result.stdout)
        self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_invalid_budget_reaches_guard_without_scientific_launch(self):
        result = self.run_entry('--confirm-new-fits','0','--acknowledge-prior-partial-fits','97')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Explicit 18286 new fits',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)

    @unittest.skipUnless(os.name=='nt','Windows PowerShell wrapper check')
    def test_actual_powershell_wrapper_check_only(self):
        powershell = Path(os.environ['WINDIR'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
        with tempfile.TemporaryDirectory(prefix='C1_E32_ps_check_') as folder:
            result = subprocess.run([str(powershell),'-NoProfile','-ExecutionPolicy','Bypass',
                '-File',str(PACKAGE/'scripts/Resume-C1E32DatasetAnalysis-PC1.ps1'),'-CheckOnly'],
                cwd=folder,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('ENTRY POINT CHECK PASS',result.stdout)


if __name__=='__main__': unittest.main()
