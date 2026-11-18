
import logging
import time

import pandas as pd

from src.api.gps_loader import load_gps_track

logger = logging.getLogger("pathsnap.prediction_service")


class PredictionService:
    """
    Coordinates the PathSnap GPS map-matching pipeline.

    Components are injected so that the actual implementations
    from the other team members can be integrated without
    changing the overall service architecture.
    """

    REQUIRED_PROBABILITY_COLUMNS = {
        "trip_id",
        "obs_idx",
        "candidate_arc_id",
        "probability"
    }

    def __init__(
        self,
        candidate_generator=None,
        feature_extractor=None,
        model_predictor=None,
        decoder=None
    ):
        self.candidate_generator = candidate_generator
        self.feature_extractor = feature_extractor
        self.model_predictor = model_predictor
        self.decoder = decoder

    # ==========================================
    # 1. CHECK PIPELINE READINESS
    # ==========================================

    def check_readiness(self):

        components = {
            "candidate_generator": self.candidate_generator,
            "feature_extractor": self.feature_extractor,
            "model_predictor": self.model_predictor,
            "decoder": self.decoder
        }

        return {
            name: component is not None
            for name, component in components.items()
        }

    # ==========================================
    # 2. LOAD GPS AND START PREDICTION
    # ==========================================

    def predict_trip(self, trip_id):

        logger.info(
            "Loading GPS data for trip %s",
            trip_id
        )

        track_df = load_gps_track(trip_id)

        logger.info(
            "Loaded %d GPS observations for trip %s",
            len(track_df),
            trip_id
        )

        return self.predict(
            trip_id=trip_id,
            track_df=track_df
        )

    # ==========================================
    # 3. COMPLETE PREDICTION PIPELINE
    # ==========================================

    def predict(self, trip_id, track_df):

        # Check whether all required components exist
        readiness = self.check_readiness()

        missing = [
            name
            for name, available in readiness.items()
            if not available
        ]

        if missing:
            raise RuntimeError(
                "Prediction pipeline components unavailable: "
                + ", ".join(missing)
            )

        if track_df.empty:
            raise ValueError(
                "GPS track contains no observations"
            )

        start_time = time.perf_counter()

        logger.info(
            "Starting prediction for trip %s",
            trip_id
        )

        # --------------------------------------
        # Stage 1: Candidate generation
        # --------------------------------------

        candidate_df = self.candidate_generator(
            trip_id,
            track_df
        )

        if not isinstance(candidate_df, pd.DataFrame):
            raise TypeError(
                "Candidate generator must return a DataFrame"
            )

        if candidate_df.empty:
            raise ValueError(
                "No road candidates generated"
            )

        logger.info(
            "Generated %d candidate records",
            len(candidate_df)
        )

        # --------------------------------------
        # Stage 2: Feature extraction
        # --------------------------------------

        feature_df = self.feature_extractor(
            candidate_df
        )

        if not isinstance(feature_df, pd.DataFrame):
            raise TypeError(
                "Feature extractor must return a DataFrame"
            )

        # --------------------------------------
        # Stage 3: ML probability prediction
        # --------------------------------------

        probabilities = self.model_predictor(
            feature_df
        )

        if not isinstance(probabilities, pd.DataFrame):
            raise TypeError(
                "Model predictor must return a DataFrame"
            )

        missing_columns = (
            self.REQUIRED_PROBABILITY_COLUMNS
            - set(probabilities.columns)
        )

        if missing_columns:
            raise ValueError(
                "Prediction output missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        if probabilities.empty:
            raise ValueError(
                "Model returned no probabilities"
            )

        # Validate probability values
        if probabilities["probability"].isna().any():
            raise ValueError(
                "Model returned missing probabilities"
            )

        if not probabilities["probability"].between(0, 1).all():
            raise ValueError(
                "Invalid probability values"
            )

        # Validate probability normalization per observation
        probability_sums = (
            probabilities.groupby(
                ["trip_id", "obs_idx"]
            )["probability"].sum()
        )

        if not (
            (probability_sums - 1.0).abs() < 1e-5
        ).all():

            raise ValueError(
                "Candidate probabilities are not normalized "
                "for every GPS observation"
            )

        logger.info(
            "ML candidate probabilities validated for trip %s",
            trip_id
        )

        # --------------------------------------
        # Stage 4: HMM / Viterbi decoding
        # --------------------------------------

        predicted_route = self.decoder(
            probabilities
        )

        if predicted_route is None:
            raise ValueError(
                "HMM decoder returned no route"
            )

        # --------------------------------------
        # Stage 5: Calculate inference time
        # --------------------------------------

        inference_time = (
            time.perf_counter() - start_time
        )

        logger.info(
            "Prediction completed for trip %s in %.4f seconds",
            trip_id,
            inference_time
        )

        # --------------------------------------
        # Stage 6: Return prediction results
        # --------------------------------------

        return {
            "trip_id": trip_id,
            "predicted_route": predicted_route,
            "metrics": {
                "gps_observations": len(track_df),
                "candidate_records": len(candidate_df)
            },
            "inference_time": round(
                inference_time,
                4
            )
        }
