# RoadGuard AI website deployment

The existing Streamlit application remains at `app.py`. The independent website is in `frontend/`; its FastAPI service is in `backend/`. The API calls the existing report lifecycle and `road_ai.py` routines and uses `models/yolo11n.onnx` for photo inference. It does not provide public status mutation; status tracking shows the recorded lifecycle history.

## Local run and test

Use Python 3.13, matching the current `.venv/pyvenv.cfg`. The existing environment is Python 3.13.15 and its requirements specify ONNX Runtime `>=1.20,<2`. Do not replace the existing `.venv` or root `requirements.txt`.

In PowerShell, from the project root:

```powershell
py -3.13 -m venv .venv-website
.\.venv-website\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell window:

```powershell
cd frontend
python -m http.server 5173
```

Open `http://127.0.0.1:5173`. API documentation is at `http://127.0.0.1:8000/docs`. The API initializes/updates the existing `data/road_reports.sqlite` database using the original additive migrations. To test API behavior and the existing suite:

```powershell
python -m unittest discover -s tests -v
```

API tests use a temporary database. Existing tests should also be run before release. The browser loads Leaflet, Google Fonts, and OpenStreetMap tiles from their public hosts, so the map and fonts need internet access.

Optional local path settings:

```powershell
$env:ROADGUARD_DATABASE = "C:\path\to\persistent\road_reports.sqlite"
$env:ROADGUARD_UPLOAD_DIR = "C:\path\to\persistent\uploads"
$env:ROADGUARD_ANNOTATED_DIR = "C:\path\to\persistent\annotated"
$env:ROADGUARD_MODEL = "C:\path\to\yolo11n.onnx"
$env:CORS_ORIGINS = "http://127.0.0.1:5173"
```

## Deploy the API on Render

1. Ensure the project is in a Git repository accessible to Render. `models/*.onnx` and `data/*.sqlite` are intentionally Git-ignored. A fresh Render build therefore cannot load the existing model or SQLite data automatically. Verify you have permission to host the model. To include the model in the repository when its license permits, force-add `models/yolo11n.onnx`; alternatively provide a private HTTPS download URL as the Render secret environment variable `MODEL_DOWNLOAD_URL` and use this build command:

   ```sh
   mkdir -p models && curl -fL "$MODEL_DOWNLOAD_URL" -o models/yolo11n.onnx && echo "A93A8E1735AAF6CD502D13BE54684B6335EB8CCA1EEC270D053E43550531E1F9  models/yolo11n.onnx" | sha256sum -c - && pip install -r backend/requirements.txt
   ```

   The SHA-256 above is for the current local ONNX file. Do not put a signed URL or credentials in Git. The SQLite database remains on your local machine unless you separately migrate/copy it; copying a local database can expose submitted report data.
2. Create a Render **Web Service** for the repository. Set the root directory to the repository root. Use Python 3.13. With the model build command above, set start command to `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` and set `ROADGUARD_MODEL=/opt/render/project/src/models/yolo11n.onnx`. If you force-add the model and use the simple build command, the same model path applies. Set the build command to `pip install -r backend/requirements.txt` when the model is committed.
3. Add a Render persistent disk mounted at `/var/data` if reports, photos, and annotations must survive redeploys. Set:

   ```text
   ROADGUARD_DATABASE=/var/data/road_reports.sqlite
   ROADGUARD_UPLOAD_DIR=/var/data/uploads
   ROADGUARD_ANNOTATED_DIR=/var/data/annotated
   CORS_ORIGINS=https://YOUR-VERCEL-DOMAIN.vercel.app
   ```

   The model path remains `/opt/render/project/src/models/yolo11n.onnx` for either model provisioning method above. The service starts even when the model is absent, but uploads will report AI analysis as unavailable and `/api/health` will show `model_file: false`. Upload directories and the SQLite database share the persistent disk.
4. Deploy and verify `https://YOUR-RENDER-SERVICE.onrender.com/api/health` and `/api/districts`. Confirm `model_file` is true before presenting AI photo analysis as available. Render free instances may sleep; the first request after sleep can take longer.

`ROADGUARD_DATABASE` should point to a new/empty database for a clean hosted instance, or to a deliberately provisioned copy. On startup the existing project migration runs. Keep backups of any database containing reports before deploying schema updates.

## Deploy the frontend on Vercel

1. Import the same Git repository into Vercel. Set the project **Root Directory** to `frontend`. No build command is needed; it is a static HTML/CSS/JavaScript site. `frontend/vercel.json` supplies clean static routing.
2. In `frontend/config.js`, replace the local default with the verified Render API origin, for example:

   ```js
   window.ROADGUARD_API_URL = "https://YOUR-RENDER-SERVICE.onrender.com";
   ```

   Commit/redeploy this configuration. The frontend is a static site and does not substitute for the Python API.
3. Set the Render `CORS_ORIGINS` value to the exact Vercel production origin, and include any Vercel preview origin only when you intend to allow it. Redeploy the API after changing CORS.
4. Open the Vercel URL and verify the API badge says **Connected**, district boundaries load, empty reports show the empty state, and the API health endpoint is reachable. Test a real report and photo only against a database where you intend to retain that submission.

The deployment is not considered verified until both public URLs have been opened and checked. No public deployment is performed by this project setup.

## Known connection limits

- The current district file has 34 valid boundaries for 38 registered districts. Four districts remain listed in selectors and coverage diagnostics, but have no map polygon in the source GeoJSON.
- The website displays real data already in the configured SQLite database. It will show an honest empty state when there are no reports. A fresh hosted database does not inherit local reports.
- Status tracking is read-only. Status transitions still obey the original app's lifecycle, repair evidence, and verification requirements; this public frontend does not expose unauthenticated status changes.
- There is no user authentication or private report access in the source project. Report details and photos returned by this API are public. Add an authentication and authorization design before using it for restricted data.
- Hosted AI requires the actual `yolo11n.onnx` artifact to be provisioned. If the artifact is unavailable or invalid, no AI result is fabricated; the API reports the analysis failure.
