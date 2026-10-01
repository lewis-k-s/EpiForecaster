from data.samplers import ShuffledBatchSampler


def test_shuffled_batch_sampler_covers_indices_in_reproducible_contiguous_batches() -> None:
    sampler = ShuffledBatchSampler(dataset_size=18, batch_size=4, seed=7)
    first_epoch = list(sampler)
    second_epoch = list(sampler)
    peer = ShuffledBatchSampler(dataset_size=18, batch_size=4, seed=7)

    assert first_epoch == list(peer)
    assert second_epoch == list(peer)
    assert first_epoch != second_epoch
    for batches in (first_epoch, second_epoch):
        assert sorted(index for batch in batches for index in batch) == list(range(18))
        assert all(
            batch == list(range(batch[0], batch[0] + len(batch))) for batch in batches
        )
