from SyntheticDataGenerator import SyntheticRareEventGenerator

def generate_and_save_csv(name, config, save_path="./synthetic_datasets"):
    gen = SyntheticRareEventGenerator(
        n_samples=5000,
        n_features=15,
        n_informative=5,
        n_redundant=3,
        class_sep=config["sep"],
        weights=config["weights"],
        flip_y=0.01,
        random_state=42
    )
    
    df = gen.generate()
    df = gen.add_noise(df, noise_level=config["noise"])
    df = gen.inject_drift(df, drift_strength=config["drift"])

    file_name = f"{name}_w{int(config['weights'][1]*100)}_n{int(config['noise']*100)}_d{int(config['drift']*100)}.csv"
    df.to_csv(f"{save_path}/{file_name}", index=False)
    print(f"✅ Saved {file_name}")