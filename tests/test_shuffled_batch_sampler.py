from data.samplers import ShuffledBatchSampler


def test_shuffled_batch_sampler_preserves_contiguous_batches() -> None:
    sampler = ShuffledBatchSampler(dataset_size=10, batch_size=4, seed=123)
    batches = list(iter(sampler))

    flattened = [idx for batch in batches for idx in batch]
    assert sorted(flattened) == list(range(10))
    for batch in batches:
        assert batch == list(range(batch[0], batch[0] + len(batch)))


def test_shuffled_batch_sampler_changes_order_across_epochs() -> None:
    sampler = ShuffledBatchSampler(dataset_size=16, batch_size=4, seed=7)
    epoch_one = list(iter(sampler))
    epoch_two = list(iter(sampler))

    assert epoch_one != epoch_two


def test_shuffled_batch_sampler_is_reproducible_for_same_seed() -> None:
    sampler_a = ShuffledBatchSampler(dataset_size=16, batch_size=4, seed=7)
    sampler_b = ShuffledBatchSampler(dataset_size=16, batch_size=4, seed=7)

    assert list(iter(sampler_a)) == list(iter(sampler_b))
