"""Isolated-environment and third-party license metadata screening."""
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import sys
from r07.core.artifacts import digest, sha256, write_json

REQUIRED = {"numpy": "2.3.5", "pandas": "2.3.3", "scipy": "1.18.1",
            "scikit-learn": "1.9.1", "prefect": "3.8.8",
            "fastapi": "0.141.1", "starlette": "1.3.1",
            "pydantic": "2.13.4", "pydantic-core": "2.46.4",
            "pydantic-settings": "2.14.2", "pydantic-extra-types": "2.11.1"}
OPEN_LICENSES = ("MIT", "BSD", "Apache", "Python Software Foundation", "PSF",
                 "ISC", "Mozilla Public License", "MPL-", "GNU General Public License",
                 "GNU Lesser General Public License", "GPL-", "LGPL-", "Artistic",
                 "Unlicense", "CC0-", "Zlib", "Public Domain", "HPND")
LICENSE_PHRASES = ("permission is hereby granted, free of charge",
                   "redistribution and use in source and binary forms",
                   "apache license", "python software foundation license",
                   "mozilla public license", "gnu lesser general public license",
                   "gnu general public license", "isc license", "artistic license")

def environment_inventory():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Pinned initial environment requires Python 3.12")
    versions = {name: metadata.version(name) for name in REQUIRED}
    if versions != REQUIRED:
        raise ValueError("Unexpected required versions: " + str(versions))
    rows, unresolved = [], []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        expression = dist.metadata.get("License-Expression", "")
        license_field = dist.metadata.get("License", "")
        classifiers = [c for c in dist.metadata.get_all("Classifier", []) if c.startswith("License ::")]
        notices = []
        for file in dist.files or []:
            normalized = str(file).replace("\\", "/")
            if ".dist-info/" in normalized and re.search(r"(license|copying|notice)", normalized, re.I):
                path = Path(dist.locate_file(file))
                if path.is_file():
                    notices.append({"file": normalized, "sha256": sha256(path)})
        fields = expression + " " + license_field + " " + " ".join(classifiers)
        recognized = any(re.search(r"(?<![A-Za-z])" + re.escape(token) + r"(?![A-Za-z])",
                                   fields, flags=re.I) for token in OPEN_LICENSES)
        if not recognized:
            for notice in notices:
                text = Path(dist.locate_file(notice["file"])).read_text(encoding="utf-8", errors="replace").lower()
                if any(phrase in text for phrase in LICENSE_PHRASES):
                    recognized = True
        if name.lower().replace("_", "-") == "c1-r07":
            recognized = True  # Local researcher's own code, not a third-party subscription.
        if not recognized:
            unresolved.append(name)
        rows.append(dict(name=name, version=dist.version, license_expression=expression,
                         license_metadata=license_field, classifiers=classifiers,
                         license_files=notices, recognized_open_license=recognized))
    return dict(python=platform.python_version(), platform=platform.platform(),
                required_versions=versions, distributions=rows,
                metadata_screen_only_not_legal_opinion=True,
                unresolved_license_metadata=unresolved,
                status="READY" if not unresolved else "LICENSE_METADATA_REVIEW_REQUIRED")

def require_environment(workspace):
    inventory = environment_inventory()
    home = Path(workspace) / "environment"
    home.mkdir(parents=True, exist_ok=True)
    identity = digest(inventory)
    path = home / ("inventory_" + identity[:16] + ".json")
    if not path.exists():
        write_json(path, inventory)
    if inventory["unresolved_license_metadata"]:
        raise ValueError("Open-license metadata needs review before execution: " +
                         ", ".join(inventory["unresolved_license_metadata"]) + "; " + str(path))
    # All installed versions matter for reproducible resume, not only direct dependencies.
    fingerprint = digest({"python": inventory["python"], "platform": inventory["platform"],
                          "packages": [(r["name"], r["version"]) for r in inventory["distributions"]]})
    return inventory, fingerprint

def configure_local_engine(workspace, root):
    # Applied before importing Prefect. Never reuse a signed-in Cloud profile/key.
    for name in list(os.environ):
        if name.startswith("PREFECT_"):
            os.environ.pop(name, None)
    os.chdir(root)  # Do not load another project's working-directory settings.
    os.environ["PREFECT_HOME"] = str((Path(workspace) / "coordinator_state").resolve())
    os.environ["PREFECT_PROFILES_PATH"] = str(Path(root) / "configs/machines/prefect_profiles.toml")
    os.environ["PREFECT_PROFILE"] = "r07_local"
    os.environ["PREFECT_SERVER_EPHEMERAL_ENABLED"] = "true"
    os.environ["PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS"] = "120"
    os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] = "false"
