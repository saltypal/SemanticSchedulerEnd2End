from SemParam.latent_features import extract_latent_features, confirmed_novelty


def test_task_features_use_one_image_forward_and_do_not_guess_labels(tmp_path):
    path = tmp_path / "image.bin"
    path.write_bytes(b"fixture")
    class Backend:
        def __init__(self):
            self.calls = 0
        def encode_image(self, path):
            self.calls += 1
            return (1., 0.)
        def encode_texts(self, texts):
            return {text: (1., 0.) for text in texts}
    backend = Backend()
    features = extract_latent_features(path, task_id="task", task_text="Explicit task", backend=backend)
    assert features["context_relevance"] == 1
    assert backend.calls == 1
    assert features["task_importance"] is None
    assert features["priority_probabilities"] is None
    assert confirmed_novelty((1., 0.), ()) is None
    assert confirmed_novelty((1., 0.), ((1., 0.),)) == 0
