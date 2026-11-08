"""Calls the VisionGuard model API in the cloud instead of loading the model locally (Task 8)."""
import time
import requests


class RemoteModelError(Exception):
    """The cloud model API could not give an answer."""


class RemotePredictor:
    """Same interface as Predictor (predict + info), but the model runs on the Oracle VM."""
    INFO_TTL = 60   # seconds: cache /health so every page render doesn't call the cloud

    def __init__(self, base_url, api_key, timeout=30.0):
        if not base_url or not api_key:
            raise RuntimeError("INFERENCE_MODE=remote needs MODEL_API_URL and MODEL_API_KEY.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()        # reuses the connection between calls
        self._info, self._info_at = None, 0.0
        self.info()                               # check the API once at start-up

    def info(self):
        if self._info and time.monotonic() - self._info_at < self.INFO_TTL:
            return self._info
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            r.raise_for_status()
            model = r.json()["model"]
            self._info = {**model, "mode": "remote", "api_url": self.base_url, "reachable": True}
        except (requests.RequestException, ValueError, KeyError):
            last = self._info or {"name": "unknown", "version": "unknown", "threshold": None, "img_size": None}
            self._info = {**last, "mode": "remote", "api_url": self.base_url, "reachable": False}
        self._info_at = time.monotonic()
        return self._info

    def predict(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        try:
            r = self.session.post(f"{self.base_url}/predict",
                                  files={"image": ("upload.jpg", data)},
                                  headers={"X-API-Key": self.api_key},
                                  timeout=self.timeout)
        except requests.Timeout:
            raise RemoteModelError(f"The model API did not answer within {self.timeout:.0f} seconds.")
        except requests.RequestException:
            raise RemoteModelError("The model API could not be reached. It may be stopped, or the network is down.")
        if r.status_code != 200:
            try:
                msg = r.json().get("message", "")
            except ValueError:
                msg = r.text[:200]
            raise RemoteModelError(f"The model API answered {r.status_code}: {msg}")
        result = r.json()
        result["model_latency_ms"] = result.get("latency_ms")             # time inside the cloud
        result["latency_ms"] = (time.perf_counter() - t0) * 1000           # full round trip, as the user feels it
        return result