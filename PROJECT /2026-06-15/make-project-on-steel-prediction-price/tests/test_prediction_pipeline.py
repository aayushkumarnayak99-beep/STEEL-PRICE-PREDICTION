from data.generate_sample_data import build_dataset
from src.steel_price_prediction.train_model import FEATURES, TARGET, build_pipeline


def test_pipeline_trains_and_predicts_positive_price():
    df = build_dataset(rows=80, seed=7)
    pipeline = build_pipeline()
    pipeline.fit(df[FEATURES], df[TARGET])

    prediction = pipeline.predict(df[FEATURES].head(1))[0]

    assert prediction > 0
