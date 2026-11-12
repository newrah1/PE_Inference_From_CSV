import requests
import urllib3
import threading

import tkinter as tk
import pandas as pd

from tkinter import filedialog, messagebox, ttk


# Set PROMETHEUS_BASE_URL when the deployment is not using the default gateway.
BASE_URL = "https://magneto.ds.maxar.com:3001"
CORE_URL = f"{BASE_URL}/api/core"
INFERENCE_URL = f"{BASE_URL}/api/inference"
OASIS_URL = f"{BASE_URL}/api/oasis"

# M2M configuration
CLIENT_ID = "m2m-eb68d436-e3d1-4471-b650-5d988a755d90"
CLIENT_SECRET = "1x2J5dTYa2LCY3BUP3wqo_-nGCH271WAc5mFnEcluvv5jYdbGna_rucRxEOUyyTf"

# Inference defaults
IMAGEID_CSV = r"D:\PythonProject\ImageID_CSV\imageids.csv"
MODEL_ID = 15
PROVIDER = "ispy"
NMS = 20
CONFIDENCE = 51

# Keep TLS verification enabled in normal environments. Only disable it for
# trusted development environments with a self-signed certificate.
VERIFY_TLS = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Poll OASIS indexing job every five seconds and stops after three minutes.
INDEX_POLL_SECONDS = 5
INDEX_TIMEOUT_SECONDS = 180



def fetch_models(session, token):
    # Build the payload
    payload = {}

    # Get the response
    response = session.get(
        f"{CORE_URL}/model",
        json=payload,
        headers=auth_headers(token),
        verify=VERIFY_TLS,
    )

    response.raise_for_status()
    body = response_body(response)
    models = []

    for model in body.get("content", []):
        models.append({
            "id": model.get("id"),
            "name": model.get("name"),
            "version": model.get("version"),
            "description": model.get("description"),
        })

    return sorted(
        models,
        key=lambda model: int(model["id"]) if str(model.get("id", "")).isdigit() else 0,
    )


def get_models(session, token):
    models = fetch_models(session, token)
    df = pd.DataFrame(models)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.sort_values(by="id")

    print(df.to_string(
        index=False,
        justify="left",
        formatters={
            "description": lambda x: "" if pd.isna(x) else str(x)
        }
    ))

    return models


def get_token(session):
    # Exchange the M2M client credentials for a JWT.
    response = session.post(
        f"{CORE_URL}/auth/m2m/token",
        json={
            "client_id": CLIENT_ID,
            "secret": CLIENT_SECRET,
        },
        verify=VERIFY_TLS,
    )
    response.raise_for_status()
    return response.json()["jwt"]


def get_imageid_df(csv_file):
    # pandas treats the first CSV line as the header by default.
    df = pd.read_csv(csv_file)
    return df


def auth_headers(token):
    # All API requests after authentication use the JWT as a Bearer token.
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def response_body(response):
    # Some endpoints return JSON while others may return an empty response.
    try:
        return response.json()
    except ValueError:
        return response.text


def response_message(body):
    # Extract an error message from the response formats used by the APIs.
    if not isinstance(body, dict):
        return str(body)

    details = body.get("details")
    if isinstance(details, dict) and details.get("message"):
        return str(details["message"])

    return str(body.get("message") or body.get("detail") or body.get("error") or "")


def get_oasis_image_reference(session, token, cat_id):
    # Resolve the CAT ID through the documented OASIS catalog endpoint.
    response = session.get(
        f"{OASIS_URL}/catalog/search/catid/{cat_id}",
        headers=auth_headers(token),
        verify=VERIFY_TLS,
    )

    if response.status_code == 404:
        raise LookupError(
            f"OASIS catalog does not contain CAT ID {cat_id!r}. "
            "Verify the CAT ID, provider/account visibility, and that the image "
            "has been ingested before submitting inference."
        )

    response.raise_for_status()
    image_reference = response.json()
    if not image_reference or not image_reference.get("dcImageId"):
        raise RuntimeError(f"OASIS returned an invalid image reference: {response.text}")

    return image_reference


def submit_inference_job(session, token, payload):
    # Submit the job request to the inference API.
    response = session.post(
        f"{INFERENCE_URL}/job",
        json=payload,
        headers=auth_headers(token),
        verify=VERIFY_TLS,
    )

    if response.status_code in {200, 201, 204}:
        return response_body(response)

    body = response_body(response)
    raise requests.HTTPError(
        f"Inference job request failed ({response.status_code}): "
        f"{response_message(body)}",
        response=response,
    )


def create_inference_job(session,token,cat_id,model_id,provider="ispy",nms=None,confidence=None,):
    # Build the payload
    payload = {
        "catId": cat_id,
        "modelId": str(model_id),
        "type": "InferenceJobRequest",
    }

    # Optional parameters are omitted unless the caller provides them.
    inference_parameters = {}
    if nms is not None:
        inference_parameters["nms"] = str(nms)
    if confidence is not None:
        inference_parameters["confidence"] = str(confidence)
    if inference_parameters:
        payload["inferenceParameters"] = inference_parameters

    # Try to create the inference job using the original CAT ID.
    response = session.post(
        f"{INFERENCE_URL}/job",
        json=payload,
        headers=auth_headers(token),
        verify=VERIFY_TLS,
    )

    if response.status_code in {200, 201, 204}:
        return response_body(response)

    body = response_body(response)
    message = response_message(body)

    # If inference cannot find the image by CAT ID, resolve it through the
    # documented OASIS catalog lookup and retry with the full image reference.
    if response.status_code in {400, 500} and "Image not found with ID" in message:
        payload.pop("catId", None)
        payload["imageReference"] = get_oasis_image_reference(session, token, cat_id)
        return submit_inference_job(session, token, payload)

    raise requests.HTTPError(
        f"Inference job request failed ({response.status_code}): {message}",
        response=response,
    )


class InferenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PE Inference From CSV")
        self.geometry("1050x720")
        self.minsize(900, 600)

        self.csv_path = tk.StringVar(value=IMAGEID_CSV)
        self.model_id = tk.StringVar(value=str(MODEL_ID))
        self.provider = tk.StringVar(value=PROVIDER)
        self.nms = tk.StringVar(value=str(NMS))
        self.confidence = tk.StringVar(value=str(CONFIDENCE))

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="CSV").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(controls, textvariable=self.csv_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text="Browse", command=self.browse_csv).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(controls, text="Model ID").grid(row=1, column=0, sticky="w", pady=(8, 0), padx=(0, 6))
        ttk.Entry(controls, textvariable=self.model_id, width=12).grid(row=1, column=1, sticky="w", pady=(8, 0))

        params = ttk.Frame(controls)
        params.grid(row=1, column=1, sticky="e", pady=(8, 0))
        ttk.Label(params, text="Provider").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(params, textvariable=self.provider, width=12).grid(row=0, column=1, padx=(0, 14))
        ttk.Label(params, text="NMS").grid(row=0, column=2, padx=(0, 6))
        ttk.Entry(params, textvariable=self.nms, width=8).grid(row=0, column=3, padx=(0, 14))
        ttk.Label(params, text="Confidence").grid(row=0, column=4, padx=(0, 6))
        ttk.Entry(params, textvariable=self.confidence, width=8).grid(row=0, column=5)

        buttons = ttk.Frame(controls)
        buttons.grid(row=1, column=2, sticky="e", pady=(8, 0))
        self.load_models_button = ttk.Button(buttons, text="Load Models", command=self.load_models)
        self.load_models_button.grid(row=0, column=0, padx=(0, 6))
        self.run_button = ttk.Button(buttons, text="Run Inference", command=self.run_inference)
        self.run_button.grid(row=0, column=1, padx=(0, 6))
        ttk.Button(buttons, text="Quit", command=self.destroy).grid(row=0, column=2)

        content = ttk.PanedWindow(self, orient=tk.VERTICAL)
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        model_frame = ttk.LabelFrame(content, text="Models", padding=8)
        model_frame.rowconfigure(0, weight=1)
        model_frame.columnconfigure(0, weight=1)

        columns = ("id", "name", "version", "description")
        self.model_table = ttk.Treeview(model_frame, columns=columns, show="headings", height=12)
        self.model_table.heading("id", text="ID")
        self.model_table.heading("name", text="Name")
        self.model_table.heading("version", text="Version")
        self.model_table.heading("description", text="Description")
        self.model_table.column("id", width=70, anchor="w", stretch=False)
        self.model_table.column("name", width=230, anchor="w")
        self.model_table.column("version", width=120, anchor="w", stretch=False)
        self.model_table.column("description", width=560, anchor="w")
        self.model_table.grid(row=0, column=0, sticky="nsew")
        self.model_table.bind("<<TreeviewSelect>>", self.select_model)

        model_scroll = ttk.Scrollbar(model_frame, orient=tk.VERTICAL, command=self.model_table.yview)
        model_scroll.grid(row=0, column=1, sticky="ns")
        self.model_table.configure(yscrollcommand=model_scroll.set)

        log_frame = ttk.LabelFrame(content, text="Log", padding=8)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        content.add(model_frame, weight=2)
        content.add(log_frame, weight=1)

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select Image ID CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path.set(path)

    def select_model(self, _event):
        selected = self.model_table.selection()
        if not selected:
            return
        values = self.model_table.item(selected[0], "values")
        if values:
            self.model_id.set(values[0])

    def load_models(self):
        self.load_models_button.configure(state="disabled")
        self.run_in_thread(self._load_models_worker, self._models_loaded)

    def _load_models_worker(self):
        with requests.Session() as session:
            token = get_token(session)
            return fetch_models(session, token)

    def _models_loaded(self, models):
        for row in self.model_table.get_children():
            self.model_table.delete(row)

        for model in models:
            self.model_table.insert(
                "",
                tk.END,
                values=(
                    model.get("id") or "",
                    model.get("name") or "",
                    model.get("version") or "",
                    model.get("description") or "",
                ),
            )

        self.load_models_button.configure(state="normal")
        self.log(f"Loaded {len(models)} models.")

    def run_inference(self):
        self.run_button.configure(state="disabled")
        self.run_in_thread(self._run_inference_worker, self._inference_done)

    def _run_inference_worker(self):
        csv_path = self.csv_path.get().strip()
        model_id = self.model_id.get().strip()
        provider = self.provider.get().strip() or PROVIDER
        nms = self.optional_number(self.nms.get())
        confidence = self.optional_number(self.confidence.get())

        if not csv_path:
            raise ValueError("CSV path is required.")
        if not model_id:
            raise ValueError("Model ID is required.")

        imageid_df = get_imageid_df(csv_path)
        results = []

        with requests.Session() as session:
            token = get_token(session)
            for image_id in imageid_df["ImageID"].dropna():
                job = create_inference_job(
                    session=session,
                    token=token,
                    cat_id=image_id,
                    model_id=model_id,
                    provider=provider,
                    nms=nms,
                    confidence=confidence,
                )

                job_id = job.get("jobId") or job.get("id")
                if not job_id:
                    raise RuntimeError(f"Inference response did not include a job ID: {job}")
                results.append((image_id, job_id))

        return results

    def _inference_done(self, results):
        for image_id, job_id in results:
            self.log(f"{image_id}: Inference Request JOBID = {job_id}")
        self.log(f"Submitted {len(results)} inference jobs.")
        self.run_button.configure(state="normal")

    def run_in_thread(self, worker, on_success):
        def target():
            try:
                result = worker()
            except Exception as exc:
                self.after(0, self.show_error, exc)
                return
            self.after(0, on_success, result)

        threading.Thread(target=target, daemon=True).start()

    def show_error(self, exc):
        self.load_models_button.configure(state="normal")
        self.run_button.configure(state="normal")
        self.log(f"ERROR: {exc}")
        messagebox.showerror("Error", str(exc))

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    @staticmethod
    def optional_number(value):
        value = value.strip()
        if not value:
            return None
        return int(value)


if __name__ == "__main__":
    app = InferenceApp()
    app.mainloop()
# end