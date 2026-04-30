

Dieses Dokument bietet einen tiefen Einblick in die Logik der nnUNet-Segmentierungs-App und ist für Entwickler oder Forscher gedacht, die das System warten.


# Kernkonzepte 

Dieses Dokument bietet einen tiefen Einblick in die Logik der nnUNet-Segmentierungs-App und ist für Entwickler oder Forscher gedacht, die das System warten.

## 1. Datenaufnahme & Standardisierung
Die Pipeline beginnt mit rohen DICOM-Daten, die für ihre inkonsistenten Metadaten-Konventionen bekannt sind. Daher ist viel Regex-Handling erforderlich, um diese für Dateinamen zu nutzen. 
Wir haben uns für ein ausführliches Setup mit PatientID, PatientName, SeriesNumber, SeriesDescription und StudyDescription entschieden. Dies sollte alle Sonderfälle abdecken. Die Namen werden dadurch recht lang: 
```python
pid = getattr(dcm, "PatientID", "UnknownID")  
pname = getattr(dcm, "PatientName", "UnknownName")
sdesc = getattr(dcm, "SeriesDescription", "UnknownSeries")
snum = getattr(dcm, "SeriesNumber", 0)
stdesc = getattr(dcm, "StudyDescription", "unknownStudy")
# Es wäre schön, wenn die Radiologie ein einheitliches Format hätte, aber es kann alles sein. 
folder = sort_dir / f"{pid}_{pname}_Series{snum}@{stdesc}_{sdesc}"  
```
DICOM-Sortierung (DICOM_splitter): Anstatt Dateien zu verschieben, erstellt die App Hardlinks im Verzeichnis sortiert/. Dies ermöglicht es der App, Dateien in einer sauberen Hierarchie zu organisieren, selbst wenn die ursprüngliche Struktur extrem ungeordnet war, ohne den benötigten Speicherplatz zu verdoppeln.

Koordinatenstandard (RAS+): Die App erzwingt für alle NIfTI-Volumina die RAS+-Orientierung (Right, Anterior, Superior).

Warum? nnUNet und die meisten KI-Trainings-Pipelines benötigen konsistente Voxel-Orientierungen, um räumliche Beziehungen korrekt zu lernen. Zudem ist RAS der Standard für NIFTIs.

Logik: Wenn der Input als LPS (Left, Posterior, Superior) erkannt wird, erfolgt eine Reorientierung mittels nibabel.orientations, bevor die Speicherung im NIFTI/-Ordner erfolgt.

## 2. ROI-Verarbeitung & "Cutting"
Für große Volumina (z. B. Ganzkörper-CTs) bietet die App das Tool cut_volume, um den VRAM-Verbrauch zu senken und die Inferenz zu beschleunigen, da der Sliding-Window-Overlap kubisch ansteigt. Ein guter Schnitt reduziert die Rechenzeit erheblich. Es sollte genug Platz gelassen werden, damit das Modell ausreichend Kontext behält.

Koordinatensystem: Die GUI erlaubt Eingaben in RAS (nativ in der App und z. B. 3D Slicer) oder LPS (nativ in DICOM-Viewern wie MicroDicom).

Destruktiv vs. Nicht-Destruktiv: Wenn "Keep Originals" aktiviert ist, werden die zugeschnittenen Versionen in NIFTI_cut/ gespeichert, sodass der Standard-Ordner NIFTI/ für andere Aufgaben erhalten bleibt.

X-Achsen-Maskierung: Für bilaterale Aufgaben (z. B. Trennung von linkem und rechtem Femur) nutzt die App Maskierung. Anstatt die Datei physisch zu teilen, werden die Voxel der "unerwünschten" Seite genullt. Dies bewahrt die globale Koordinatenausrichtung, erleichtert die HU-Analyse und beschädigt die Original-NIFTI-Dateien nicht.

## 3. Die nnUNet Inferenz-Engine
Die App nutzt das nnUNetv2-Framework, den Goldstandard für medizinische Bildsegmentierung bei limitiertem VRAM.

Der Predictor: Der nnUNetPredictor wird mit spezifischen "Folds" initialisiert (meist 0 bis 4). Die finale Maske ist ein Ensemble-Durchschnitt dieser 5 Modelle, was "False Positive"-Segmentierungen signifikant reduziert.

Kaskaden-Architektur: Für komplexe Anatomie wie die Wirbelsäule (Dataset 217) wird ein mehrstufiger Ansatz verwendet:

Stufe 1: Sagt eine niedrig auflösende Maske voraus (label_lowres/), um die generelle Position zu finden (besserer globaler Kontext).

Stufe 2: Nutzt die Low-Res-Maske als "Anker", um die hochauflösende finale Segmentierung vorherzusagen.

Modell-Gewichte: Gewichte werden in dem durch die Umgebungsvariable nnUNet_results definierten Verzeichnis gespeichert.

Eine separate Datei nur für nnUNet und die Integration neuer Modelle ist ebenfalls vorhanden.

## 4. 3D-Oberflächenrekonstruktion (multi_stl.py)
Die Umwandlung einer Voxel-Maske in ein 3D-Mesh erfordert ein Gleichgewicht zwischen anatomischer Genauigkeit und Glätte der Oberfläche.

Konvertierung
Marching Cubes: Wir nutzen den skimage.measure.marching_cubes Algorithmus, um das initiale Dreiecksnetz zu erzeugen. Es gibt andere Algorithmen, aber Marching Cubes ist der Standard und nicht extrem rechenintensiv.
Am Ende benötigen wir die Affin-Matrix jedes Scans, um vom Voxel-Raum zurück in Weltkoordinaten zu gelangen:
```
Python
verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
verts = (affine @ verts.T).T[:, :3]
verts = convert_to_LPS(verts)
```
Glättung
Taubin-Glättung: Im Gegensatz zur Standard-Laplace-Glättung, die das Volumen "schrumpft", nutzt Taubin-Glättung einen Tiefpassfilter, um Treppeneffekte zu entfernen, ohne das zugrunde liegende Volumen zu verändern. Der Faktor verhält sich invertiert (niedrigerer Faktor = stärkere Glättung). Konsultieren Sie bei Bedarf die offizielle Dokumentation.

Mesh-Reparatur:
PyMeshFix: Füllt automatisch Löcher und stellt sicher, dass das Mesh "wasserdicht" (manifold) für den 3D-Druck ist. Es schließt auch offene Meshes, die entstehen, wenn das Segment bis zum Bildrand reicht.

Island Removal: Entfernt optional kleine schwebende Artefakte (versprengte Voxel), um nur die größte zusammenhängende Struktur zu behalten. Deaktivieren Sie dies bei mehrteiligen Strukturen wie Rippen.

## 5. Multiprocessing und -threading
Um die erhebliche Rechenlast der medizinischen Bildgebung zu bewältigen, ohne die Benutzeroberfläche einzufrieren, nutzt diese Anwendung ein hybrides Konformitätsmodell. Es wird zwischen I/O-gebundenen Aufgaben (Warten auf die Festplatte) und CPU-gebundenen Aufgaben (schwere mathematische Berechnungen) unterschieden.

Threading für UI-Flüssigkeit und I/O
Die GUI selbst läuft auf einem einzigen Haupt-Thread. Um ein Aufhängen des Fensters ("Keine Rückmeldung") zu verhindern, nutzt die App threading.Thread für Hintergrundoperationen, die Wartezeiten beinhalten:

UI-Reaktionsfähigkeit:
Die Hauptverarbeitungsschleife (process_data) wird in einem Daemon-Thread gestartet, sodass der Benutzer weiterhin mit dem Fenster interagieren kann, während die KI arbeitet.

Indikator-Scanning: Wenn Sie einen Eingabeordner auswählen, scannt ein Hintergrund-Thread die DICOM-Header, um das Menü "Scan Indicators" zu füllen, ohne die UI ins Stocken zu bringen.

DICOM-Verlinkung:
Während der DICOM_splitter-Phase wird ein ThreadPoolExecutor verwendet, um Hardlinks zu erstellen. Da das Verlinken eine I/O-Operation ist, können mehrere Threads diese Anfragen gleichzeitig an das Betriebssystem stellen.

Multiprocessing für rechenintensive Aufgaben
Der Global Interpreter Lock (GIL) von Python verhindert, dass mehrere Threads gleichzeitig CPU-lastige Mathematik ausführen. Um dies zu umgehen, nutzt die App Multiprocessing (speziell ProcessPoolExecutor), um nahezu jeden Kern Ihrer Workstation zu nutzen:

Paralleles DICOM zu NIfTI: Die Konvertierung mehrerer Serien erfolgt parallel, wobei jeder Prozess einen anderen Scan verarbeitet.

Batched STL-Generierung: Die Oberflächenrekonstruktion ist der CPU-intensivste Teil der Pipeline. Der ParallelSTLProcessor unterteilt die Liste der Labelmaps in Batches. Jeder Batch wird auf max_workers verteilt (standardmäßig CPU-Anzahl minus 4, um das System bedienbar zu halten). Jeder Prozess führt unabhängig die Algorithmen Marching Cubes, Taubin-Glättung und PyMeshFix-Reparatur aus.

HU-Analyse: Die Berechnung von Statistiken (Mittelwert, Schiefe, Kurtosis) für Millionen von Voxeln wird über die Probanden hinweg parallelisiert, um die Erstellung des finalen Excel-Berichts zu beschleunigen.

Inter-Prozess-Kommunikation (IPC)
Da Hintergrundprozesse nicht direkt mit der GUI "sprechen" können, nutzt die App eine queue.Queue:

Worker legen ProgressEvent-Objekte in die Queue.

Die GUI "pollt" diese Queue alle 100ms mit root.after(), um den Fortschrittsbalken und Statustext sicher auf dem Haupt-Thread zu aktualisieren.

## 6. Datenintegrität & Validierung (Pydantic)
Auch wenn Sie den Kerncode vielleicht nicht bearbeiten müssen, verwendet die Anwendung Pydantic (eine Datenvalidierungsbibliothek), um sicherzustellen, dass die in der GUI gewählten Einstellungen sinnvoll sind, bevor das Skript startet.

Fail-Fast-Logik: Wenn Sie versehentlich eine negative Zahl für einen Crop-Bereich oder ein ungültiges E-Mail-Format eingeben, fängt Pydantic dies sofort ab und verhindert den Start des Skripts. Dies spart Stunden an Verarbeitungszeit, die sonst für einen fehlerhaften Durchlauf verschwendet würden.

Rückverfolgbarkeit: Dies ist der Motor hinter der Datei run_parameter.json. Es wandelt den komplexen Zustand der GUI in ein sauberes, strukturiertes und reproduzierbares Datenformat um.

Typsicherheit: Es stellt sicher, dass Variablen den korrekten Typ behalten (z. B. dass eine Koordinate immer ein Integer und kein String ist).

🔗 Externe Dokumentation & Bibliotheken
Um die Kerngeometrie oder Inferenzlogik zu ändern, beziehen Sie sich auf diese Bibliotheken:

nnUNetv2 Offizielle Docs: Handhabung von Training, Inferenz und Umgebungsvariablen.
https://github.com/MIC-DKFZ/nnUNet

PyMeshFix Docs: Tiefer Einblick in die repair() Algorithmen.
https://github.com/pyvista/pymeshfix
https://docs.pyvista.org/index.html

Nibabel: Dokumentation zur Handhabung von NIfTI-Headern und Affin-Matrizen.
https://nipy.org/nibabel/

Pydantic: Typsicherheit
https://pydantic.dev/docs/validation/latest/get-started/

MLflow: Tracking-Server für Experimente.
Führen Sie den Server mit tmux in WSL aus. Nutzen Sie eventuell Cloudflare Quicktunnels für mobilen Zugriff. Quicktunnels scheinen ein Ratelimit zu haben; nach einem Neustart kann es zu Fehlern beim Aufbau eines neuen Tunnels kommen (bekanntes Problem seit 2023).
https://mlflow.org/

Streamlit: Für das Dashboard. Macht die UI super einfach, muss als eigener Prozess ausgeführt werden.
https://streamlit.io/

## 7. Datei-Rückverfolgbarkeit
Jeder Durchlauf generiert eine decoder.json und eine run_parameter.json.

decoder.json: Kritisch für die klinische Forschung. Sie ordnet die nnUNet-Dateinamen (z. B. CASE_001.nii.gz) wieder den ursprünglichen, bei der Sortierung generierten Namen zu.

run_parameter.json: Speichert die exakte GUI-Konfiguration, sodass jedes Ergebnis später perfekt repliziert werden kann.
