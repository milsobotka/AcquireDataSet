
import sys, time, multiprocessing as mp
from pathlib import Path
import importlib.util
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QSpinBox, QLineEdit, QCheckBox, QFileDialog, QMessageBox, QHBoxLayout
)


# loader 

def load_py_function(py_path: str, func_name: str):
    path = Path(py_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


record_u235c = load_py_function("VmbPy/Examples/main_u235c_GUI.py", "record_u235c")
record_u130vswir = load_py_function("VmbPy/Examples/main_u130vswir_GUI.py", "record_u130vswir")
record_boson = load_py_function("boson_camera/boson_GUI.py", "record_boson")
record_hl7 = load_py_function("hl7_parser/hl7_data_collector_GUI.py", "record_hl7")


# worker-process 

def worker_entry(name, func, result_q: mp.Queue, *args):
    try:
        func(*args)
        result_q.put((name, True))
    except Exception as exc:
        print(f"[{name}] ERROR: {exc}")
        result_q.put((name, False))


# - GUI 
class MainWindow(QWidget):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("KIBM MSobotka")
        self.setMinimumSize(700, 580)

        # widgets 
        self.duration_spin = QSpinBox(minimum=1, maximum=600, value=120)
        self.com_edit = QLineEdit("COM7")
        self.port_edit = QLineEdit("8008")
        self.cb_u235c = QCheckBox("u235c", checked=True)
        self.cb_u130vswir = QCheckBox("u130vswir", checked=False)
        self.cb_boson = QCheckBox("Boson", checked=True)
        self.cb_hl7 = QCheckBox("Hl7", checked=True)
        self.icon_u235c = QLabel("-")
        self.icon_u130vswir = QLabel("-")
        self.icon_boson = QLabel("-")
        self.icon_hl7 = QLabel("-")
        self.output_btn = QPushButton("Wybierz docelowy folder")
        self.start_btn = QPushButton("Start Nagrywanie")
        self.stop_btn = QPushButton("Stop");
        self.stop_btn.setEnabled(False)
        self.timer_lbl = QLabel("00:00 uplynelo / 00:00 pozostalo", alignment=Qt.AlignCenter)
        self.status_lbl = QLabel("Gotowe", alignment=Qt.AlignCenter)

        # layout 
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Duration (s):"))
        row1.addWidget(self.duration_spin)
        row1.addStretch()
        row1.addWidget(QLabel("Boson COM:"))
        row1.addWidget(self.com_edit)
        row1.addStretch()
        row1.addWidget(QLabel("HL7 port:"))
        row1.addWidget(self.port_edit)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Enable recorders:"))
        row2.addWidget(self.cb_u235c)
        row2.addWidget(self.icon_u235c)
        row2.addWidget(self.cb_u130vswir)
        row2.addWidget(self.icon_u130vswir)
        row2.addWidget(self.cb_boson)
        row2.addWidget(self.icon_boson)
        row2.addWidget(self.cb_hl7)
        row2.addWidget(self.icon_hl7)
        row2.addStretch()
        lay = QVBoxLayout(self)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.output_btn)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        lay.addLayout(btn_row)
        lay.addWidget(self.timer_lbl);
        lay.addWidget(self.status_lbl)

        # signals 
        self.output_btn.clicked.connect(self.choose_folder)
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)

        # state 
        self.output_dir: Path | None = None
        self.processes: list[mp.Process] = []
        self.result_q: mp.Queue | None = None
        self.active = 0
        self.errors = False
        self.start_evt: mp.Event | None = None
        self.stop_evt: mp.Event | None = None
        self.rec_duration = 0
        self.start_time = 0.0

        # timers 
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._count_tick)
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._progress_tick)
        self.duration_timer = QTimer(self)
        self.duration_timer.setSingleShot(True)
        self.duration_timer.timeout.connect(self._on_duration_elapsed)
        self.result_timer = QTimer(self)
        self.result_timer.timeout.connect(self._poll_results)

    #  helpers 
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Wybierz docelowy folder")
        if d: self.output_dir = Path(d); self.status_lbl.setText(f"Output: {d}")

    @staticmethod
    def ts():
        import datetime;
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

    #  start 
    def start_recording(self):
        if not self.output_dir:
            QMessageBox.warning(self, "No folder", "Wybierz docelowy folder")
            return
        sel = [n for n, cb in (("u235c", self.cb_u235c), ("u130vswir", self.cb_u130vswir), ("boson", self.cb_boson), ("hl7", self.cb_hl7)) if cb.isChecked()]
        if not sel:
            QMessageBox.warning(self, "None", "Wybierze choc jedna kamere")
            return

        self.start_evt, self.stop_evt = mp.Event(), mp.Event()
        self.result_q = mp.Queue()
        self.processes.clear()
        self.active = 0
        self.errors = False
        self.icon_u235c.setText("-")
        self.icon_boson.setText("-")
        self.icon_hl7.setText("-")
        self.icon_u130vswir.setText("-")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("Przygotowanie")
        self.rec_duration = self.duration_spin.value()
        run_dir = self.output_dir / self.ts()

        if "u235c" in sel:
            p = mp.Process(target=worker_entry, args=(
                "u235c", record_u235c, self.result_q, self.start_evt, self.stop_evt, self.rec_duration,
                run_dir / "u235c"),
                           daemon=True)
            p.start()
            self.processes.append(p)
            self.active += 1

        if "u130vswir" in sel:
            p = mp.Process(target=worker_entry, args=(
                "u130vswir", record_u130vswir, self.result_q, self.start_evt, self.stop_evt, self.rec_duration,
                run_dir / "u130vswir"),
                           daemon=True)
            p.start()
            self.processes.append(p)
            self.active += 1

        if "boson" in sel:
            p = mp.Process(target=worker_entry, args=(
                "boson", record_boson, self.result_q, self.start_evt, self.stop_evt, self.rec_duration,
                run_dir / "boson",
                self.com_edit.text().strip()), daemon=True)
            p.start()
            self.processes.append(p)
            self.active += 1

        if "hl7" in sel:
            p = mp.Process(
                target=worker_entry,
                args=("hl7", record_hl7, self.result_q,
                      self.start_evt, self.stop_evt,
                      self.rec_duration,
                      run_dir / "hl7",
                      int(self.port_edit.text().strip())
                      ),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            self.active += 1

        self.countdown = 5
        self.status_lbl.setText(f"Rozpoczecie za {self.countdown} s")
        self.countdown_timer.start(1000)

    def _count_tick(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.status_lbl.setText(f"Rozpoczecie za {self.countdown} s")
        else:
            self.countdown_timer.stop()
            self.status_lbl.setText("Nagrywanie trwa")
            self.start_evt.set()
            self.start_time = time.time()
            self.progress_timer.start(1000)
            self.duration_timer.start(self.rec_duration * 1000)
            self.result_timer.start(500)

    #  stop 
    def stop_recording(self):
        if self.stop_evt: self.stop_evt.set()
        self._kill_all(force_error=True)
        self.status_lbl.setText("Zatrzymano - wymuszone przerwanie")

    def _kill_all(self, force_error=False):
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()
                force_error = True

        if force_error:
            self.errors = True

        self.result_timer.stop()
        self.progress_timer.stop()
        self.active = 0
        self._finish()

    # timers & results 
    def _progress_tick(self):
        elapsed = int(time.time() - self.start_time)
        remain = max(0, self.rec_duration - elapsed)
        self.timer_lbl.setText(f"{elapsed:02d}s uplynelo / {remain:02d}s pozostalo")

    def _on_duration_elapsed(self):
        self.status_lbl.setText("Nagrywanie zakonczone; zapisywanie plikow")
        self.progress_timer.stop()
        self.timer_lbl.setText(f"{self.rec_duration:02d}s uplynelo / 00s pozostalo")

    def _poll_results(self):
        # get from queue
        while self.result_q and not self.result_q.empty():
            name, ok = self.result_q.get()
            lab = {"u235c": self.icon_u235c,
                   "u130vswir": self.icon_u130vswir,
                   "boson": self.icon_boson,
                   "hl7": self.icon_hl7}.get(name.lower())
            if lab:
                lab.setText("V" if ok else "X")
                lab.setStyleSheet(f"color:{'green' if ok else 'red'}")
            self.active -= 1
            self.errors |= (not ok)

        #  check if process ended
        for p in list(self.processes):
            if not p.is_alive():
                # find label
                name = p._target.__name__.replace("record_", "")
                lab = {"u235c": self.icon_u235c,
                       "u130vswir": self.icon_u130vswir,
                       "boson": self.icon_boson,
                       "hl7": self.icon_hl7}.get(name)
                if lab and lab.text() == "-":
                    ok = (p.exitcode == 0)
                    lab.setText("V" if ok else "X")
                    lab.setStyleSheet(f"color:{'green' if ok else 'red'}")
                    self.active -= 1
                    self.errors |= (not ok)
                self.processes.remove(p)  

        # End
        if self.active <= 0:
            self.result_timer.stop()
            self._finish()

    def _finish(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_timer.stop()
        self.status_lbl.setText("Zakonczono z bledami" if self.errors else "Zakonczono pomyslnie")


#  entry 
if __name__ == "__main__":
    mp.set_start_method("spawn")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
