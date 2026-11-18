
from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import folium

from streamlit_folium import st_folium


# ==========================================
# 1. CONFIGURATION
# ==========================================

API_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STUDENT_TEST_DIR = PROJECT_ROOT / "data" / "student_test"

st.set_page_config(
    page_title="PathSnap | Map Matching",
    page_icon="🗺️",
    layout="wide"
)


# ==========================================
# 2. APPLICATION HEADER
# ==========================================

st.title("🗺️ PathSnap")

st.subheader("GPS Map Matching and Route Reconstruction")

st.write(
    "Reconstruct realistic vehicle trajectories "
    "from noisy GPS observations."
)

st.divider()


# ==========================================
# 3. BACKEND CONNECTION
# ==========================================

try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    health.raise_for_status()

    st.success("Backend connected successfully")

except requests.RequestException:
    st.error("Backend unavailable. Start FastAPI first.")
    st.stop()


# ==========================================
# 4. MODEL PIPELINE STATUS
# ==========================================

st.header("Model Pipeline Status")

try:
    response = requests.get(
        f"{API_URL}/pipeline/status",
        timeout=5
    )

    response.raise_for_status()

    pipeline = response.json()

    components = pipeline["components"]

    if pipeline["pipeline_ready"]:
        st.success("All prediction components are ready.")
    else:
        st.warning(
            "Prediction pipeline integration is pending."
        )

    col1, col2, col3, col4 = st.columns(4)

    component_items = [
        ("Candidate Generator", "candidate_generator"),
        ("Feature Extractor", "feature_extractor"),
        ("ML Predictor", "model_predictor"),
        ("HMM Decoder", "decoder")
    ]

    for column, (label, key) in zip(
        [col1, col2, col3, col4],
        component_items
    ):

        ready = components.get(key, False)

        column.metric(
            label,
            "Ready" if ready else "Pending"
        )

except (requests.RequestException, ValueError, KeyError) as error:

    st.warning(
        f"Unable to retrieve pipeline status: {error}"
    )


# ==========================================
# 5. MODEL ARTIFACT STATUS
# ==========================================

with st.expander("Model Artifact Status"):

    try:
        response = requests.get(
            f"{API_URL}/artifacts/status",
            timeout=5
        )

        response.raise_for_status()

        artifact_data = response.json()

        artifacts = artifact_data["artifacts"]

        if artifact_data["artifacts_ready"]:
            st.success("All model artifacts are available.")
        else:
            st.warning(
                "Some model artifacts are not available yet."
            )

        for name, status in artifacts.items():

            if status == "available":
                st.success(f"{name}: Available")
            else:
                st.warning(f"{name}: {status}")

    except (requests.RequestException, ValueError, KeyError) as error:

        st.error(
            f"Unable to retrieve artifact status: {error}"
        )


st.divider()


# ==========================================
# 6. FETCH AVAILABLE TRIPS
# ==========================================

st.header("Trip Selection")

try:
    response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )

    response.raise_for_status()

    trips = response.json()["trips"]

except requests.RequestException as error:

    st.error(f"Unable to retrieve trips: {error}")
    st.stop()

if not trips:
    st.warning("No student-testing trips available.")
    st.stop()

st.write(
    f"Available student-testing trips: {len(trips)}"
)

selected_trip = st.selectbox(
    "Select a trip",
    trips
)


# ==========================================
# 7. LOAD GPS OBSERVATIONS
# ==========================================

st.divider()

st.header("GPS Trajectory Visualisation")

track_path = (
    STUDENT_TEST_DIR
    / selected_trip
    / f"{selected_trip}.track"
)

if not track_path.exists():

    st.error("GPS track file not found.")
    st.stop()

try:

    track = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=[
            "longitude",
            "latitude",
            "timestamp"
        ]
    )

    track = track.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    # Validate GPS coordinates
    valid = (
        track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
    )

    track = (
        track[valid]
        .reset_index(drop=True)
    )

    if track.empty:

        st.error("No valid GPS observations found.")
        st.stop()

    # Folium requires latitude before longitude
    coordinates = list(
        zip(
            track["latitude"],
            track["longitude"]
        )
    )

    center_lat = track["latitude"].mean()
    center_lon = track["longitude"].mean()


    # ======================================
    # 8. CREATE FOLIUM MAP
    # ======================================

    gps_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=14,
        control_scale=True
    )

    # Raw GPS trajectory
    folium.PolyLine(
        coordinates,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="Raw GPS trajectory"
    ).add_to(gps_map)

    # Starting GPS point
    folium.Marker(
        coordinates[0],
        popup="GPS Start",
        tooltip="Starting point",
        icon=folium.Icon(color="green")
    ).add_to(gps_map)

    # Ending GPS point
    folium.Marker(
        coordinates[-1],
        popup="GPS End",
        tooltip="Ending point",
        icon=folium.Icon(color="red")
    ).add_to(gps_map)

    folium.LayerControl().add_to(gps_map)


    # ======================================
    # 9. DISPLAY TRIP STATISTICS
    # ======================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "GPS Observations",
        len(track)
    )

    col2.metric(
        "Starting Timestamp",
        f"{track['timestamp'].iloc[0]:.0f} s"
    )

    col3.metric(
        "Ending Timestamp",
        f"{track['timestamp'].iloc[-1]:.0f} s"
    )


    # ======================================
    # 10. DISPLAY INTERACTIVE MAP
    # ======================================

    st_folium(
        gps_map,
        width=None,
        height=500,
        key=f"gps_map_{selected_trip}"
    )

    st.caption(
        "Blue line: Raw GPS trajectory. "
        "Green marker: Start. Red marker: End."
    )

except Exception as error:

    st.error(
        f"Unable to generate GPS map: {error}"
    )

    st.stop()


# ==========================================
# 11. ROUTE PREDICTION
# ==========================================

st.divider()

st.header("Route Prediction")

st.write(
    f"Selected Trip: **{selected_trip}**"
)

if st.button(
    "Predict Route",
    type="primary",
    use_container_width=False
):

    with st.spinner(
        "Processing GPS observations and requesting prediction..."
    ):

        try:

            prediction_response = requests.post(
                f"{API_URL}/predict/",
                json={
                    "trip_id": selected_trip
                },
                timeout=120
            )

            prediction_response.raise_for_status()

            result = prediction_response.json()

            st.success("Route prediction completed successfully!")

            st.subheader("Prediction Results")

            st.json(result)

            # Display prediction metrics when available
            if "metrics" in result:

                metrics = result["metrics"]

                col1, col2 = st.columns(2)

                col1.metric(
                    "GPS Observations",
                    metrics.get("gps_observations", "N/A")
                )

                col2.metric(
                    "Candidate Records",
                    metrics.get("candidate_records", "N/A")
                )

            if "inference_time" in result:

                st.metric(
                    "Inference Time",
                    f"{result['inference_time']} seconds"
                )

            st.info(
                "The prediction response is displayed above. "
                "The current Folium map still shows raw GPS data "
                "until predicted road geometry is integrated."
            )

        except requests.HTTPError as error:

            try:
                error_data = prediction_response.json()

                detail = error_data.get(
                    "detail",
                    str(error)
                )

            except ValueError:
                detail = str(error)

            st.warning(
                f"Prediction unavailable: {detail}"
            )

        except requests.Timeout:

            st.error(
                "Prediction request timed out."
            )

        except requests.RequestException as error:

            st.error(
                f"Unable to connect to prediction API: {error}"
            )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "PathSnap | GPS Map Matching and Route Reconstruction"
)
