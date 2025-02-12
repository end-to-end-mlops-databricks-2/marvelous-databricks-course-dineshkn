from nba_analysis.config import ProjectConfig
from nba_analysis.models.basic_model import BasicModel


def test_data_validation(model):
    """Basic data validation checks"""
    assert model.data is not None, "Data should be loaded"
    assert model.X is not None, "Features should be prepared"
    assert model.y is not None, "Target should be prepared"
    assert len(model.X) == len(model.y), "Features and target should have same length"
    assert not model.X.isna().any().any(), "Features should not contain NaN values"
    print("All data validations passed!")


def main():
    # Load config
    config = ProjectConfig(
        input_data="/Volumes/mlops_dev/dineshka/nba_data/all_seasons.csv",
        local_data="data/raw/all_seasons.csv",
        target="pts",
        experiment_name="/Shared/nba-points-basic",
        num_features=[
            "age",
            "player_height",
            "player_weight",
            "gp",
            "reb",
            "ast",
            "net_rating",
            "oreb_pct",
            "dreb_pct",
            "usg_pct",
            "ts_pct",
            "ast_pct",
        ],
        cat_features=[
            "team_abbreviation",
            "college",
            "country",
            "draft_year",
            "draft_round",
            "draft_number",
        ],
        parameters={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.01},
        experiment_name_basic="/Shared/nba-points-basic",
        experiment_name_custom="/Shared/nba-points-custom",
        experiment_name_fe="/Shared/nba-points-fe",
    )

    # Initialize model
    basic_model = BasicModel(config=config, tags={"git_sha": "test", "branch": "local"})

    try:
        # Load and prepare data
        basic_model.load_data()
        basic_model.prepare_features()
        print("Data loading and feature preparation successful!")

        # Run validations
        test_data_validation(basic_model)

        # Train model
        print("\nStarting model training...")
        basic_model.train()

        # Validate model exists and basic metrics
        assert basic_model.model is not None, "Model should exist after training"
        train_score = basic_model.model.score(basic_model.X, basic_model.y)
        print(f"Model R² score on training data: {train_score:.4f}")

    except Exception as e:
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    main()
