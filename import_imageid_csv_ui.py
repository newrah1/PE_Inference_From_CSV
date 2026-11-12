import json
import requests
import os
import time
import urllib3

import pandas as pd


# Set PROMETHEUS_BASE_URL when the deployment is not using the default gateway.
BASE_URL = "https://magneto.ds.maxar.com:3001"
CORE_URL = f"{BASE_URL}/api/core"
INFERENCE_URL = f"{BASE_URL}/api/inference"
OASIS_URL = f"{BASE_URL}/api/oasis"
MODEL_URL = f"{BASE_URL}/api/model"

CLIENT_ID = "m2m-eb68d436-e3d1-4471-b650-5d988a755d90"
CLIENT_SECRET = "1x2J5dTYa2LCY3BUP3wqo_-nGCH271WAc5mFnEcluvv5jYdbGna_rucRxEOUyyTf"

# Keep TLS verification enabled in normal environments. Only disable it for
# trusted development environments with a self-signed certificate.
VERIFY_TLS = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Poll OASIS indexing job every five seconds and stops after three minutes.
INDEX_POLL_SECONDS = 5
INDEX_TIMEOUT_SECONDS = 180

IMAGEID_CSV = r"D:\PythonProject\ImageID_CSV\imageids.csv"
MODEL_ID = 15
PROVIDER = "ispy"
NMS = 20
CONFIDENCE = 50

def get_models(token):
    # Build the payload
    payload = {}

    # Try to create the inference job using the original CAT ID.
    response = session.get(
        f"{MODEL_URL}/model",
        json=payload,
        headers=auth_headers(token),
        verify=VERIFY_TLS,
    )

    body = response_body(response)
    # key_list = list(body.keys())
    # print(key_list)
    models = []

    for model in body.get("content", []):
        models.append({
            "id": model.get("id"),
            "name": model.get("name"),
            "version": model.get("version"),
            "description": model.get("description"),
        })

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


if __name__ == "__main__":
    # Show models
    with requests.Session() as session:
        jwt = get_token(session)
        get_models(jwt)

    imageid_df = get_imageid_df(IMAGEID_CSV)
    for image_id in imageid_df['ImageID'].dropna():
        print(f"\nProcessing image -- {image_id}")

        # A Session reuses connections and keeps the example efficient for callers
        # that create more than one job.
        with requests.Session() as session:
            jwt = get_token(session)
            job = create_inference_job(
                session=session,
                token=jwt,
                cat_id=image_id,
                model_id=MODEL_ID,
                provider=PROVIDER,
                nms=NMS,
                confidence=CONFIDENCE,
            )

            job_id = job.get("jobId") or job.get("id")
            if not job_id:
                raise RuntimeError(f"Inference response did not include a job ID: {job}")
            print(json.dumps(f"Inference Request JOBID = {job_id}", indent=2))

