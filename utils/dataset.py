from pathlib import Path
import os.path
import random
from collections import defaultdict
import math
import os
import hashlib
import json
import tarfile

import numpy as np
import torch
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from datasets.fingerprint import Hasher
from PIL import Image
import imageio
import multiprocess as mp
from tqdm import tqdm

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple


DEBUG = False
IMAGE_SIZE_ROUND_TO_MULTIPLE = 32
NUM_PROC = min(8, os.cpu_count())
CAPTIONS_JSON_FILE = 'captions.json'

UNCOND_FRACTION = 0.0


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


def shuffle_captions(captions: list[str], count: int = 0, delimiter: str = ', ', caption_prefix: str = '') -> list[str]:
    if count == 0:
        return [caption_prefix + c for c in captions]

    def shuffle_caption(caption: str, delimiter: str = ", ") -> str:
        split = caption.split(delimiter)
        random.shuffle(split)
        return delimiter.join(split)

    return [caption_prefix + shuffle_caption(caption, delimiter) for caption in captions for _ in range(count)]


def bucket_suffix(key):
    if len(key) == 2:
        return f'{key[0]:.3f}_{key[1]}'
    elif len(key) == 3:
        return f'{key[0]}x{key[1]}x{key[2]}'
    else:
        raise RuntimeError(f'Unexpected bucket: {key}')


def _map_and_cache(dataset, map_fn, cache_dir, cache_file_prefix='', new_fingerprint_args=None, regenerate_cache=False, caching_batch_size=1):
    # Do the fingerprinting ourselves, because otherwise map() does it by serializing the map function.
    # That goes poorly when the function is capturing huge models (slow, OOMs, etc).
    new_fingerprint_args = [] if new_fingerprint_args is None else new_fingerprint_args
    new_fingerprint_args.append(dataset._fingerprint)
    new_fingerprint = Hasher.hash(new_fingerprint_args)
    cache_file = cache_dir / f'{cache_file_prefix}{new_fingerprint}.arrow'
    cache_file = str(cache_file)
    dataset = dataset.map(
        map_fn,
        cache_file_name=cache_file,
        load_from_cache_file=(not regenerate_cache),
        writer_batch_size=100,
        new_fingerprint=new_fingerprint,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=caching_batch_size,
        num_proc=NUM_PROC,
        with_rank=True,
    )
    dataset.set_format('torch')
    return dataset


class TextEmbeddingDataset:
    def __init__(self, te_dataset):
        self.te_dataset = te_dataset
        self.image_spec_to_te_idx = defaultdict(list)
        for i, image_spec in enumerate(te_dataset['image_spec']):
            self.image_spec_to_te_idx[tuple(image_spec)].append(i)

    def get_text_embeddings(self, image_spec, caption_number):
        return self.te_dataset[self.image_spec_to_te_idx[image_spec][caption_number]]


def _cache_text_embeddings(metadata_dataset, map_fn, i, cache_dir, regenerate_cache, caching_batch_size):

    def flatten_captions(example):
        image_spec_out, caption_out, is_video_out = [], [], []
        for image_spec, captions, is_video in zip(example['image_spec'], example['caption'], example['is_video']):
            for caption in captions:
                image_spec_out.append(image_spec)
                caption_out.append(caption)
                is_video_out.append(is_video)
        return {'image_spec': image_spec_out, 'caption': caption_out, 'is_video': is_video_out}

    flattened_captions = metadata_dataset.map(flatten_captions, batched=True, keep_in_memory=True, remove_columns=metadata_dataset.column_names)
    te_dataset = _map_and_cache(
        flattened_captions,
        map_fn,
        cache_dir,
        cache_file_prefix=f'text_embeddings_{i}_',
        new_fingerprint_args=[i],
        regenerate_cache=regenerate_cache,
        caching_batch_size=caching_batch_size,
    )
    return TextEmbeddingDataset(te_dataset)


# The smallest unit of a dataset. Represents a single size bucket from a single folder of images
# and captions on disk. Not batched; returns individual items.
class SizeBucketDataset:
    def __init__(self, metadata_dataset, directory_config, size_bucket, model_name):
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.size_bucket = size_bucket
        self.model_name = model_name
        self.path = Path(self.directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name / f'cache_{bucket_suffix(size_bucket)}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_embedding_datasets = []
        self.uncond_text_embeddings = []
        self.num_repeats = self.directory_config['num_repeats']
        self.shuffle_skip = max(directory_config.get('cache_shuffle_num', 0), 1) # Should be provided in DirectoryDataset
        if self.num_repeats <= 0:
            raise ValueError(f'num_repeats must be >0, was {self.num_repeats}')

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.size_bucket}')
        self.latent_dataset = _map_and_cache(
            self.metadata_dataset,
            map_fn,
            self.cache_dir,
            cache_file_prefix='latents_',
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
        )
        iteration_order = []
        for example in self.metadata_dataset.select_columns(['image_spec', 'caption']):
            image_spec = example['image_spec']
            captions = example['caption']
            for i, caption in enumerate([captions[i:i + self.shuffle_skip] for i in range(0, len(captions), self.shuffle_skip)]):
                iteration_order.append((image_spec, caption, i))
        # Shuffle again, since one media file can produce multiple training examples. E.g. video, or maybe
        # in the future data augmentation. Don't need to shuffle text embeddings since those are looked
        # up by image file name.
        shuffle_with_seed(iteration_order, 42)
        self.iteration_order = iteration_order
        self.image_spec_to_latents_idx = {
            tuple(image_spec): i
            for i, image_spec in enumerate(self.latent_dataset['image_spec'])
        }


    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.size_bucket}')
        te_dataset = _cache_text_embeddings(self.metadata_dataset, map_fn, i, self.cache_dir, regenerate_cache, caching_batch_size)
        self.text_embedding_datasets.append(te_dataset)

    def add_text_embedding_dataset(self, te_dataset):
        self.text_embedding_datasets.append(te_dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.iteration_order)
        image_spec, captions, caption_number = self.iteration_order[idx]
        image_spec = tuple(image_spec)
        ret = self.latent_dataset[self.image_spec_to_latents_idx[image_spec]]
        if DEBUG:
            print(Path(image_spec[1]).stem)
        offset = random.randrange(self.shuffle_skip)
        caption_idx = (caption_number*self.shuffle_skip) + offset

        use_uncond = UNCOND_FRACTION > 0 and random.random() < UNCOND_FRACTION
        caption = '' if use_uncond else captions[offset]

        for ds, uncond_ds in zip(self.text_embedding_datasets, self.uncond_text_embeddings):
            emb_dict = uncond_ds[0] if use_uncond else ds.get_text_embeddings(image_spec, caption_idx)
            ret.update(emb_dict)
        ret['caption'] = caption
        return ret

    def __len__(self):
        return int(len(self.iteration_order) * self.num_repeats)


# Logical concatenation of multiple SizeBucketDataset, for the same size bucket. It returns items
# as batches.
class ConcatenatedBatchedDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.post_init_called = False

    def post_init(self, batch_size, batch_size_image):
        iteration_order = []
        size_bucket = self.datasets[0].size_bucket
        for i, ds in enumerate(self.datasets):
            assert ds.size_bucket == size_bucket
            iteration_order.extend([i]*len(ds))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.datasets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        self.batch_size = batch_size_image if size_bucket[-1] == 1 else batch_size
        self._make_divisible_by(self.batch_size)
        self.post_init_called = True

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order) // self.batch_size

    def __getitem__(self, idx):
        assert self.post_init_called
        start = idx * self.batch_size
        end = start + self.batch_size
        return [self.datasets[i][j] for i, j in self.iteration_order[start:end]]

    def _make_divisible_by(self, n):
        new_length = (len(self.iteration_order) // n) * n
        self.iteration_order = self.iteration_order[:new_length]
        if new_length == 0 and is_main_process():
            logger.warning(f"size bucket {self.datasets[0].size_bucket} is being completely dropped because it doesn't have enough images")


class ARBucketDataset:
    def __init__(self, ar_frames, resolutions, metadata_dataset, directory_config, model_name):
        self.ar_frames = ar_frames
        self.resolutions = resolutions
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.model_name = model_name
        self.size_buckets = []
        self.path = Path(directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name / f'ar_frames_{bucket_suffix(self.ar_frames)}'
        os.makedirs(self.cache_dir, exist_ok=True)

        for res in resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_nearest_multiple(w, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            h = round_to_nearest_multiple(h, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            size_bucket = (w, h, self.ar_frames[1])
            metadata_with_size_bucket = self.metadata_dataset.map(lambda example: {'size_bucket': size_bucket}, keep_in_memory=True, desc='Adding size bucket')
            self.size_buckets.append(
                SizeBucketDataset(metadata_with_size_bucket, directory_config, size_bucket, model_name)
            )

    def get_size_bucket_datasets(self):
        return self.size_buckets

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.ar_frames}')
        for ds in self.size_buckets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.ar_frames}')
        te_dataset = _cache_text_embeddings(self.metadata_dataset, map_fn, i, self.cache_dir, regenerate_cache, caching_batch_size)
        for size_bucket_dataset in self.size_buckets:
            size_bucket_dataset.add_text_embedding_dataset(te_dataset)


class DirectoryDataset:
    def __init__(self, directory_config, dataset_config, model_name, framerate=None, skip_dataset_validation=False):
        self._set_defaults(directory_config, dataset_config)
        self.directory_config = directory_config
        self.dataset_config = dataset_config
        if not skip_dataset_validation:
            self.validate()
        self.model_name = model_name
        self.framerate = framerate
        self.enable_ar_bucket = directory_config.get('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        # Configure directly from user-specified size buckets.
        self.size_buckets = directory_config.get('size_buckets', dataset_config.get('size_buckets', None))
        self.use_size_buckets = (self.size_buckets is not None)
        if self.use_size_buckets:
            # sort size bucket from longest frame length to shortest
            self.size_buckets.sort(key=lambda t: t[-1], reverse=True)
            self.size_buckets = np.array(self.size_buckets)
            self.size_bucket_datasets = []
        else:
            self.resolutions = self._process_user_provided_resolutions(
                directory_config.get('resolutions', dataset_config['resolutions'])
            )
            self.ar_bucket_datasets = []
        self.shuffle = directory_config.get('cache_shuffle_num', dataset_config.get('cache_shuffle_num', 0))
        self.directory_config['cache_shuffle_num'] = self.shuffle # Make accessible if it wasn't yet, for picking one out
        self.shuffle_delimiter = directory_config.get('cache_shuffle_delimiter', dataset_config.get('cache_shuffle_delimiter', ", "))
        self.path = Path(self.directory_config['path'])
        self.mask_path = Path(self.directory_config['mask_path']) if 'mask_path' in self.directory_config else None
        self.control_path = Path(self.directory_config['control_path']) if 'control_path' in self.directory_config else None
        # For testing. Default if a mask is missing.
        self.default_mask_file = Path(self.directory_config['default_mask_file']) if 'default_mask_file' in self.directory_config else None
        self.cache_dir = self.path / 'cache' / self.model_name
        self.grouping_keys_json_file = self.cache_dir / 'metadata/grouping_keys.json'

        if not self.path.exists() or not self.path.is_dir():
            raise RuntimeError(f'Invalid path: {self.path}')
        if self.mask_path is not None and (not self.mask_path.exists() or not self.mask_path.is_dir()):
            raise RuntimeError(f'Invalid mask_path: {self.mask_path}')
        if self.control_path is not None and (not self.control_path.exists() or not self.control_path.is_dir()):
            raise RuntimeError(f'Invalid control_path: {self.control_path}')
        if self.default_mask_file is not None and (not self.default_mask_file.exists() or not self.default_mask_file.is_file()):
            raise RuntimeError(f'Invalid default_mask_file: {self.default_mask_file}')

        if self.use_size_buckets:
            self.ars = np.array([w / h for w, h, _ in self.size_buckets])
        elif not self.enable_ar_bucket:
            self.ars = np.array([1.0])
        elif ars := self.directory_config.get('ar_buckets', self.dataset_config.get('ar_buckets', None)):
            self.ars = self._process_user_provided_ars(ars)
        else:
            min_ar = self.directory_config.get('min_ar', self.dataset_config['min_ar'])
            max_ar = self.directory_config.get('max_ar', self.dataset_config['max_ar'])
            num_ar_buckets = self.directory_config.get('num_ar_buckets', self.dataset_config['num_ar_buckets'])
            self.ars = np.geomspace(min_ar, max_ar, num=num_ar_buckets)
        self.log_ars = np.log(self.ars)
        frame_buckets = self.directory_config.get('frame_buckets', self.dataset_config.get('frame_buckets', [1]))
        if 1 not in frame_buckets:
            # always have an image bucket for convenience
            frame_buckets.append(1)
        frame_buckets.sort()
        self.frame_buckets = np.array(frame_buckets)

    def validate(self):
        resolutions = self.directory_config.get('resolutions', self.dataset_config.get('resolutions', []))
        if len(resolutions) > 3:
            if is_main_process():
                print('WARNING: You have set a lot of resolutions in the dataset config. Please read the comments in the example dataset.toml file,'
                      ' and make sure you understand what this setting does. If you still want to proceed with the current configuration,'
                      ' run the script with the --i_know_what_i_am_doing flag.')
            quit()

    def cache_metadata(self, regenerate_cache=False, trust_cache=False):
        def check_grouped_metadata():
            all_grouped_metadata_exists = False
            unique_grouping_keys = None
            if self.grouping_keys_json_file.exists():
                with open(self.grouping_keys_json_file) as f:
                    unique_grouping_keys = json.load(f)
                if self.use_size_buckets and not all(len(key) == 3 for key in unique_grouping_keys):
                    # Using size buckets but have AR keys.
                    return False, unique_grouping_keys
                elif not all(len(key) == 2 for key in unique_grouping_keys):
                    # Using AR buckets but have size bucket keys
                    return False, unique_grouping_keys
                all_grouped_metadata_exists = all(
                    (self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(key)}').exists()
                    for key in unique_grouping_keys
                )
            return all_grouped_metadata_exists, unique_grouping_keys

        # Check if all the grouped metadata datasets exist. If so, we can directly load them.
        all_grouped_metadata_exists, unique_grouping_keys = check_grouped_metadata()
        if regenerate_cache or not all_grouped_metadata_exists or not trust_cache:
            # Otherwise, need to compute the ungrouped metadata and then group.
            print('Grouped metadata is not cached. Computing ungrouped metadata and then grouping.')
            unique_grouping_keys = self._group_metadata_and_save_to_disk(regenerate_cache=regenerate_cache, trust_cache=trust_cache)
        else:
            print('Found grouped metadata cache. Directly loading it.')

        for grouping_key in unique_grouping_keys:
            grouped_cache_dir = self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(grouping_key)}'
            print(f'Loading grouped metadata with grouping key {grouping_key}')
            metadata = datasets.load_from_disk(str(grouped_cache_dir))
            if self.use_size_buckets:
                assert len(grouping_key) == 3
                self.size_bucket_datasets.append(
                    SizeBucketDataset(
                        metadata,
                        self.directory_config,
                        grouping_key,
                        self.model_name,
                    )
                )
            else:
                self.ar_bucket_datasets.append(
                    ARBucketDataset(
                        grouping_key,
                        self.resolutions,
                        metadata,
                        self.directory_config,
                        self.model_name,
                    )
                )

    def _group_metadata_and_save_to_disk(self, regenerate_cache=False, trust_cache=False):
        metadata_dataset = self._get_ungrouped_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)
        grouped_metadata = defaultdict(lambda: defaultdict(list))
        unique_grouping_keys = set()
        for example in tqdm(metadata_dataset, desc='Grouping examples'):
            if self.use_size_buckets:
                grouping_key = tuple(example['size_bucket'])
            else:
                grouping_key = example['ar_bucket']
                grouping_key = (grouping_key[0], int(grouping_key[1]))
            unique_grouping_keys.add(grouping_key)
            d = grouped_metadata[grouping_key]
            for k, v in example.items():
                d[k].append(v)
        unique_grouping_keys = list(unique_grouping_keys)

        if self.use_size_buckets:
            for size_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                grouped_cache_dir = self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(size_bucket)}'
                metadata.save_to_disk(str(grouped_cache_dir))
        else:
            for ar_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                grouped_cache_dir = self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(ar_bucket)}'
                metadata.save_to_disk(str(grouped_cache_dir))

        with open(self.grouping_keys_json_file, 'w') as f:
            json.dump(unique_grouping_keys, f)

        return unique_grouping_keys

    def _get_ungrouped_metadata(self, regenerate_cache=False, trust_cache=False):
        # This method caches some intermediate datasets so we don't have to enumerate all the files each time.
        metadata_cache_file_1 = self.cache_dir / f'metadata/metadata_intermediate'
        metadata_cache_file_2 = self.cache_dir / f'metadata/metadata.arrow'

        if regenerate_cache or not metadata_cache_file_1.exists() or not trust_cache:
            print('Intermediate metadata is not cached. Enumerating all files.')
            files = list(self.path.glob('*'))
            # deterministic order
            files.sort()

            # Mask can have any extension, it just needs to have the same stem as the image.
            mask_file_stems = {path.stem: path for path in self.mask_path.glob('*') if path.is_file()} if self.mask_path is not None else {}
            control_file_stems = {path.stem: path for path in self.control_path.glob('*') if path.is_file()} if self.control_path is not None else {}

            def process_file(file):
                if file.suffix != '.tar':
                    return [(None, str(file))]
                with tarfile.TarFile(file) as tar_f:
                    return [(str(file), name) for name in tar_f.getnames()]

            captions_json = self.path / CAPTIONS_JSON_FILE
            has_captions_json = captions_json.exists()

            image_specs = []
            caption_files = []
            mask_files = []
            control_files = []
            for file in tqdm(files):
                if not file.is_file() or file.suffix == '.txt' or file.suffix == '.npz' or file.suffix == '.json' or file.suffix == '.parquet':
                    continue
                for image_spec in process_file(file):
                    image_file = Path(image_spec[1])
                    caption_file = image_file.with_suffix('.txt')
                    if has_captions_json or not os.path.exists(caption_file):
                        caption_file = ''
                    image_specs.append(image_spec)
                    caption_files.append(str(caption_file))
                    # mask
                    if image_file.stem in mask_file_stems:
                        mask_files.append(str(mask_file_stems[image_file.stem]))
                    elif self.default_mask_file is not None:
                        mask_files.append(str(self.default_mask_file))
                    else:
                        if self.mask_path is not None:
                            logger.warning(f'No mask file was found for image {image_file}, not using mask.')
                        mask_files.append(None)
                    # control (e.g. Flux Kontext)
                    if self.control_path:
                        if image_file.stem not in control_file_stems:
                            raise RuntimeError(f'No control file exists for image {image_file}')
                        control_files.append(str(control_file_stems[image_file.stem]))
            assert len(image_specs) > 0, f'Directory {self.path} had no images/videos!'

            d = {'image_spec': image_specs, 'caption_file': caption_files, 'mask_file': mask_files}
            if self.control_path:
                d['control_file'] = control_files
            metadata_dataset = datasets.Dataset.from_dict(d)

            if captions_json.exists():
                print('Loading captions JSON')
                with open(captions_json) as f:
                    caption_data = json.load(f)

                def add_captions(example):
                    captions = caption_data.get(example['image_spec'][1], None)
                    if captions is None:
                        logger.warning(f'Image file {image_file} does not have an entry in captions.json')
                    else:
                        assert isinstance(captions, list), 'captions.json must contain lists of captions'
                    return {'caption': captions}

                metadata_dataset = metadata_dataset.map(add_captions, keep_in_memory=True, desc='Adding captions')
                del caption_data

            # Shuffle the data. Use a deterministic seed, so the dataset is identical on all processes.
            # Seed is based on the hash of the directory path, so that if directories have the same set of images, they are shuffled differently.
            seed = int(hashlib.md5(str.encode(str(self.path))).hexdigest(), 16) % int(1e9)
            metadata_dataset = metadata_dataset.shuffle(seed=seed)
            print('Saving intermediate metadata dataset.')
            metadata_dataset.save_to_disk(str(metadata_cache_file_1))
            # Need to delete and load from disk, or else the map() call below is extremely slow to launch worker processes
            # and they use huge amounts of memory. Probably because this dataset is in memory?
            del metadata_dataset

        print('Loading intermediate metadata dataset.')
        metadata_dataset = datasets.load_from_disk(str(metadata_cache_file_1))

        metadata_map_fn = self._metadata_map_fn()
        print('Caching ungrouped metadata.')
        metadata_dataset = metadata_dataset.map(
            metadata_map_fn,
            cache_file_name=str(metadata_cache_file_2),
            load_from_cache_file=(not regenerate_cache and trust_cache),
            batched=True,
            batch_size=1,
            num_proc=NUM_PROC,
            remove_columns=metadata_dataset.column_names,
        )
        return metadata_dataset

    def _set_defaults(self, directory_config, dataset_config):
        directory_config.setdefault('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        directory_config.setdefault('shuffle_tags', dataset_config.get('shuffle_tags', False))
        directory_config.setdefault('caption_prefix', dataset_config.get('caption_prefix', ''))
        directory_config.setdefault('num_repeats', dataset_config.get('num_repeats', 1))

    def _metadata_map_fn(self):
        tarfile_map = {}

        def fn(example):
            # batch size always 1
            caption_file = example['caption_file'][0]
            image_spec = example['image_spec'][0]
            image_file = Path(image_spec[1])
            captions = None
            if 'caption' in example:
                # Already put in dataset from captions.json file.
                captions = example['caption'][0]
            if captions is None and caption_file:
                with open(caption_file) as f:
                    captions = [f.read().strip()]
            if captions is None:
                captions = ['']
                logger.warning(f'Cound not find caption for {image_file}. Using empty caption.')
            if self.directory_config['shuffle_tags'] and self.shuffle == 0: # backwards compatibility
                self.shuffle = 1
            captions = shuffle_captions(captions, self.shuffle, self.shuffle_delimiter, self.directory_config['caption_prefix'])
            empty_return = {'image_spec': [], 'mask_file': [], 'caption': [], 'ar_bucket': [], 'size_bucket': [], 'is_video': []}
            if self.control_path:
                empty_return['control_file'] = []

            if image_spec[0] is None:
                tar_f = None
                file_obj = open(image_file, 'rb')
            else:
                tar_filename = image_spec[0]
                tar_f = tarfile_map.setdefault(tar_filename, tarfile.TarFile(tar_filename))
                file_obj = tar_f.extractfile(str(image_file))

            if image_file.suffix == '.webp':
                # Make sure this this object stays alive so it doesn't close file_obj on us.
                reader = imageio.get_reader(file_obj)
                frames = reader.get_length()
                if frames > 1:
                    raise NotImplementedError('WebP videos are not supported.')
            try:
                if image_file.suffix in VIDEO_EXTENSIONS:
                    # 100% accurate frame count, but much slower.
                    # frames = 0
                    # for frame in imageio.v3.imiter(image_file):
                    #     frames += 1
                    #     height, width = frame.shape[:2]
                    # TODO: this is an estimate of frame count. What happens if variable frame rate? Is
                    # it still close enough?
                    meta = imageio.v3.immeta(file_obj)
                    first_frame = next(imageio.v3.imiter(file_obj))
                    height, width = first_frame.shape[:2]
                    assert self.framerate is not None, "Need model framerate but don't have it. This shouldn't happen. Is the framerate attribute on the model set?"
                    frames = int(self.framerate * meta['duration'])
                else:
                    pil_img = Image.open(file_obj)
                    width, height = pil_img.size
                    frames = 1
            except Exception as e:
                logger.warning(f'Media file {image_file} could not be opened. Skipping.')
                return empty_return
            finally:
                file_obj.close()

            is_video = (frames > 1)
            log_ar = np.log(width / height)

            if self.use_size_buckets:
                size_bucket = self._find_closest_size_bucket(log_ar, frames, is_video)
                if size_bucket is None:
                    print(f'video with frames={frames} is being skipped because it is too short')
                    return empty_return
                ar_bucket = None
            else:
                ar_bucket = self._find_closest_ar_bucket(log_ar, frames, is_video)
                if ar_bucket is None:
                    print(f'video with frames={frames} is being skipped because it is too short')
                    return empty_return
                size_bucket = None

            ret = {
                'image_spec': [image_spec],
                'mask_file': [example['mask_file'][0]],
                'caption': [captions],
                'ar_bucket': [ar_bucket],
                'size_bucket': [size_bucket],
                'is_video': [is_video],
            }
            if self.control_path:
                ret['control_file'] = [example['control_file'][0]]
            return ret

        return fn

    def _find_closest_ar_bucket(self, log_ar, frames, is_video):
        # Best AR bucket is the one with the smallest AR difference in log space.
        i = np.argmin(np.abs(log_ar - self.log_ars))
        # find closest frame bucket where the number of frames is greater than or equal to the bucket
        diffs = frames - self.frame_buckets
        positive_diffs = diffs[diffs >= 0]
        if len(positive_diffs) == 0:
            # video not long enough to find any valid frame bucket
            return None
        j = np.argmin(positive_diffs)
        if is_video and self.frame_buckets[j] == 1:
            # don't let video be mapped to the image frame bucket
            return None
        ar_bucket = (self.ars[i], self.frame_buckets[j])
        return ar_bucket

    def _find_closest_size_bucket(self, log_ar, frames, is_video):
        # Best AR bucket is the one with the smallest AR difference in log space.
        ar_diffs = np.abs(log_ar - self.log_ars)
        candidate_size_buckets = self.size_buckets[np.argsort(ar_diffs, kind='stable')]
        # Find closest size bucket where the number of frames is greater than or equal to the bucket.
        # self.size_buckets was already sorted longest -> shortest frame length
        found = False
        for size_bucket in candidate_size_buckets:
            if is_video and size_bucket[-1] == 1:
                # don't let video be mapped to the image frame bucket
                continue
            if frames >= size_bucket[-1]:
                found = True
                break
        if not found:
            # video not long enough to find any valid frame bucket
            return None
        return size_bucket

    def _process_user_provided_ars(self, ars):
        ar_buckets = set()
        for ar in ars:
            if isinstance(ar, (tuple, list)):
                assert len(ar) == 2
                ar = round(ar[0] / ar[1], 6)
            ar_buckets.add(ar)
        ar_buckets = list(ar_buckets)
        ar_buckets.sort()
        return np.array(ar_buckets)

    def _process_user_provided_resolutions(self, resolutions):
        result = set()
        for res in resolutions:
            if isinstance(res, (tuple, list)):
                assert len(res) == 2
                res = round(math.sqrt(res[0] * res[1]), 6)
            result.add(res)
        result = list(result)
        result.sort()
        return result

    def get_size_bucket_datasets(self):
        if self.use_size_buckets:
            return self.size_bucket_datasets
        result = []
        for ar_bucket_dataset in self.ar_bucket_datasets:
            result.extend(ar_bucket_dataset.get_size_bucket_datasets())
        return result

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.path}')
        datasets = self.size_bucket_datasets if self.use_size_buckets else self.ar_bucket_datasets
        for ds in datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.path}')
        datasets_list = self.size_bucket_datasets if self.use_size_buckets else self.ar_bucket_datasets
        for ds in datasets_list:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)
        # TODO: do this separately for is_video True and False for models that support it?
        empty_caption_ds = datasets.Dataset.from_dict({'caption': [''], 'is_video': [False], 'image_spec': [(None, None)]})
        uncond_text_embeddings_ds = _map_and_cache(
            empty_caption_ds,
            map_fn,
            cache_dir=self.cache_dir,
            cache_file_prefix='uncond_text_embeddings_',
            regenerate_cache=regenerate_cache,
        )
        for size_bucket_ds in self.get_size_bucket_datasets():
            size_bucket_ds.uncond_text_embeddings.append(uncond_text_embeddings_ds)


# Outermost dataset object that the caller uses. Contains multiple ConcatenatedBatchedDataset. Responsible
# for returning the correct batch for the process's data parallel rank. Calls model.prepare_inputs so the
# returned tuple of tensors is whatever the model needs.
class Dataset:
    def __init__(self, dataset_config, model, skip_dataset_validation=False):
        super().__init__()
        self.dataset_config = dataset_config
        self.model = model
        self.model_name = self.model.name
        self.post_init_called = False
        self.eval_quantile = None
        if not skip_dataset_validation:
            self.model.model_specific_dataset_config_validation(self.dataset_config)

        self.directory_datasets = []
        for directory_config in dataset_config['directory']:
            directory_dataset = DirectoryDataset(
                directory_config,
                dataset_config,
                self.model_name,
                framerate=model.framerate,
                skip_dataset_validation=skip_dataset_validation,
            )
            self.directory_datasets.append(directory_dataset)

    def post_init(self, data_parallel_rank, data_parallel_world_size, per_device_batch_size, gradient_accumulation_steps, per_device_batch_size_image):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.batch_size = per_device_batch_size * gradient_accumulation_steps
        self.batch_size_image = per_device_batch_size_image * gradient_accumulation_steps
        self.global_batch_size = self.data_parallel_world_size * self.batch_size
        self.global_batch_size_image = self.data_parallel_world_size * self.batch_size_image

        # group same size_bucket together
        datasets_by_size_bucket = defaultdict(list)
        for directory_dataset in self.directory_datasets:
            for size_bucket_dataset in directory_dataset.get_size_bucket_datasets():
                datasets_by_size_bucket[size_bucket_dataset.size_bucket].append(size_bucket_dataset)
        self.buckets = []
        for datasets in datasets_by_size_bucket.values():
            self.buckets.append(ConcatenatedBatchedDataset(datasets))

        for bucket in self.buckets:
            bucket.post_init(self.global_batch_size, self.global_batch_size_image)

        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            iteration_order.extend([i]*(len(bucket)))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.buckets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        if DEBUG:
            print(f'Dataset iteration_order: {self.iteration_order}')

        self.post_init_called = True

        if subsample_ratio := self.dataset_config.get('subsample_ratio', None):
            new_len = int(len(self) * subsample_ratio)
            self.iteration_order = self.iteration_order[:new_len]

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order)

    def __getitem__(self, idx):
        assert self.post_init_called
        i, j = self.iteration_order[idx]
        examples = self.buckets[i][j]
        start_idx = self.data_parallel_rank*self.batch_size
        examples_for_this_dp_rank = examples[start_idx:start_idx+self.batch_size]
        if DEBUG:
            print((start_idx, start_idx+self.batch_size))
        batch = self._collate(examples_for_this_dp_rank)
        return batch

    # Collates a list of feature dictionaries into a single dictionary of batched features.
    # Each feature can be a tensor, list, or single item.
    def _collate(self, examples):
        ret = {}
        for key, value in examples[0].items():
            if key == 'mask':
                continue  # mask is handled specially below
            if torch.is_tensor(value):
                ret[key] = torch.stack([example[key] for example in examples])
            else:
                ret[key] = [example[key] for example in examples]
        # Only some items in the batch might have valid mask.
        masks = [example['mask'] for example in examples]
        # See if we have any valid masks. If we do, they should all have the same shape.
        shape = None
        for mask in masks:
            if mask is not None:
                assert shape is None or mask.shape == shape
                shape = mask.shape
        if shape is not None:
            # At least one item has a mask. Need to make the None masks all 1s.
            for i, mask in enumerate(masks):
                if mask is None:
                    masks[i] = torch.ones(shape, dtype=torch.float16)
            ret['mask'] = torch.stack(masks)
        else:
            # We can leave the batch mask as None and the loss_fn will skip masking entirely.
            ret['mask'] = None
        return ret

    def cache_metadata(self, regenerate_cache=False, trust_cache=False):
        for ds in self.directory_datasets:
            ds.cache_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        for ds in self.directory_datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        for ds in self.directory_datasets:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)


def _cache_fn(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, trust_cache, caching_batch_size):
    # Dataset map() starts a bunch of processes. Make sure torch uses a limited number of threads
    # to avoid CPU contention.
    # TODO: if we ever change Datasets map to use spawn instead of fork, this might not work.
    #torch.set_num_threads(os.cpu_count() // NUM_PROC)
    # HF Datasets map can randomly hang if this is greater than one (???)
    # See https://github.com/pytorch/pytorch/issues/10996
    # Alternatively, we could try fixing this by using spawn instead of fork.
    torch.set_num_threads(1)

    for ds in datasets:
        ds.cache_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)

    pipes = {}

    def latents_map_fn(example, rank):
        is_edit_dataset = ('control_file' in example)
        first_size_bucket = example['size_bucket'][0]
        tensors_and_masks = []
        image_specs = []
        captions = []
        control_tensors_and_masks = []
        for i, (image_spec, mask_path, size_bucket, caption) in enumerate(
            zip(example['image_spec'], example['mask_file'], example['size_bucket'], example['caption'])
        ):
            assert size_bucket == first_size_bucket
            items = preprocess_media_file_fn(image_spec, mask_path, size_bucket)
            tensors_and_masks.extend(items)
            image_specs.extend([image_spec] * len(items))
            captions.extend([caption] * len(items))
            if is_edit_dataset:
                control_file = example['control_file'][i]
                control_items = preprocess_media_file_fn((None, control_file), None, size_bucket)
                assert len(control_items) == 1
                assert len(items) == 1
                control_tensors_and_masks.append(control_items[0])
            else:
                control_tensors_and_masks.append(None)

        if len(tensors_and_masks) == 0:
            assert not is_edit_dataset
            return {'latents': [], 'mask': [], 'image_spec': [], 'caption': []}

        caching_batch_size = len(example['image_spec'])
        results = defaultdict(list)
        for i in range(0, len(tensors_and_masks), caching_batch_size):
            tensor = torch.stack([t[0] for t in tensors_and_masks[i:i+caching_batch_size]])
            c_tensor = torch.stack([t[0] for t in control_tensors_and_masks[i:i+caching_batch_size]]) if is_edit_dataset else None
            parent_conn, child_conn = pipes.setdefault(rank, mp.Pipe(duplex=False))
            queue.put((0, tensor, c_tensor, child_conn))
            result = parent_conn.recv()  # dict
            for k, v in result.items():
                results[k].append(v)
        # concatenate the list of tensors at each key into one batched tensor
        for k, v in results.items():
            results[k] = torch.cat(v)
        results['image_spec'] = image_specs
        results['mask'] = [t[1] for t in tensors_and_masks]
        results['caption'] = captions
        return results

    for ds in datasets:
        ds.cache_latents(latents_map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    for text_encoder_idx in range(num_text_encoders):
        def text_embedding_map_fn(example, rank):
            parent_conn, child_conn = pipes.setdefault(rank, mp.Pipe(duplex=False))
            queue.put((text_encoder_idx+1, example['caption'], example['is_video'], child_conn))
            result = parent_conn.recv()  # dict
            result['image_spec'] = example['image_spec']
            return result
        for ds in datasets:
            ds.cache_text_embeddings(text_embedding_map_fn, text_encoder_idx+1, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    # signal that we're done
    queue.put(None)


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model, regenerate_cache=False, trust_cache=False, caching_batch_size=1):
        self.model = model
        self.vae = self.model.get_vae()
        self.text_encoders = self.model.get_text_encoders()
        self.submodels = [self.vae] + list(self.text_encoders)
        self.call_vae_fn = self.model.get_call_vae_fn(self.vae)
        self.call_text_encoder_fns = [self.model.get_call_text_encoder_fn(text_encoder) for text_encoder in self.text_encoders]
        self.regenerate_cache = regenerate_cache
        self.trust_cache = trust_cache
        self.caching_batch_size = caching_batch_size
        self.datasets = []

    def register(self, dataset):
        self.datasets.append(dataset)

    # Some notes for myself:
    # Use the third-party multiprocess library because HF Datasets uses it for the map() calls.
    # Mix and match native multiprocessing / torch.multiprocessing and multiprocess at your peril! Things can break.
    # In patches.py we register reductions so Tensors sent over Queues and Pipes are efficient just like in torch.multiprocessing.
    def cache(self, unload_models=True):
        if is_main_process():
            manager = mp.Manager()
            queue = [manager.Queue()]
        else:
            queue = [None]
        torch.distributed.broadcast_object_list(queue, src=0, group=dist.get_world_group())
        queue = queue[0]

        # start up a process to run through the dataset caching flow
        if is_main_process():
            process = mp.Process(
                target=_cache_fn,
                args=(
                    self.datasets,
                    queue,
                    self.model.get_preprocess_media_file_fn(),
                    len(self.text_encoders),
                    self.regenerate_cache,
                    self.trust_cache,
                    self.caching_batch_size,
                )
            )
            process.start()

        # loop on the original processes (one per GPU) to handle tasks requiring GPU models (VAE, text encoders)
        while True:
            task = queue.get()
            if task is None:
                # Propagate None so all worker processes break out of this loop.
                # This is safe because it's a FIFO queue. The first None always comes after all work items.
                queue.put(None)
                break
            self._handle_task(task)

        if unload_models:
            # Free memory in all unneeded submodels. This is easier than trying to delete every reference.
            # TODO: check if this is actually freeing memory.
            for model in self.submodels:
                if self.model.name == 'sdxl' and model is self.vae:
                    # If full fine tuning SDXL, we need to keep the VAE weights around for saving the model.
                    model.to('cpu')
                else:
                    model.to('meta')

        dist.barrier()
        if is_main_process():
            process.join()

        # Now load all datasets from cache.
        for ds in self.datasets:
            ds.cache_metadata(trust_cache=True)
            ds.cache_latents(None)
            for i in range(1, len(self.text_encoders)+1):
                ds.cache_text_embeddings(None, i)

    @torch.no_grad()
    def _handle_task(self, task):
        id = task[0]
        # moved needed submodel to cuda, and everything else to cpu
        if next(self.submodels[id].parameters()).device.type != 'cuda':
            for i, submodel in enumerate(self.submodels):
                if i != id:
                    submodel.to('cpu')
            self.submodels[id].to('cuda')
        if id == 0:
            tensor, control_tensor, pipe = task[1:]
            if control_tensor is not None:
                # edit dataset
                results = self.call_vae_fn(tensor, control_tensor)
            else:
                results = self.call_vae_fn(tensor)
        elif id > 0:
            caption, is_video, pipe = task[1:]
            results = self.call_text_encoder_fns[id-1](caption, is_video=is_video)
        else:
            raise RuntimeError()
        # Need to move to CPU here. If we don't, we get this error:
        # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        # I think this is because HF Datasets uses the multiprocess library (different from Python multiprocessing!) so it will always use fork.
        results = {k: v.to('cpu') for k, v in results.items()}
        pipe.send(results)


def split_batch(batch, pieces):
    # Each of features, label is a tuple of tensors.
    features, label = batch
    split_size = features[0].size(0) // pieces
    # The tuples passed to Deepspeed need to only contain tensors. For None (e.g. mask, or optional conditioning), convert to empty tensor.
    split_features = zip(*(torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])]*pieces for tensor in features))
    split_label = zip(*(torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])]*pieces for tensor in label))
    # Deepspeed works with a tuple of (features, labels).
    return list(zip(split_features, split_label))


# Splits an example (feature dict) along the batch dimension into a list of examples.
# Keeping this code because we might want to switch to this way of doing things eventually.
# def split_batch(example, pieces):
#     key, value = example.popitem()
#     input_batch_size = len(value)
#     example[key] = value
#     split_size = input_batch_size // pieces
#     examples = [{} for _ in range(pieces)]
#     for key, value in example.items():
#         assert len(value) == input_batch_size
#         for i, j in enumerate(range(0, input_batch_size, split_size)):
#             examples[i][key] = value[j:j+split_size]
#     return examples


# DataLoader that divides batches into microbatches for gradient accumulation steps when doing
# pipeline parallel training. Iterates indefinitely (deepspeed requirement). Keeps track of epoch.
# Updates epoch as soon as the final batch is returned (notably different from qlora-pipe).
class PipelineDataLoader:
    def __init__(self, dataset, model_engine, gradient_accumulation_steps, model, num_dataloader_workers=2):
        if len(dataset) == 0:
            raise RuntimeError(
                'Processed dataset was empty. Probably caused by rounding down for each size bucket.\n'
                'Try decreasing the global batch size, or increasing num_repeats.\n'
                f'The dataset config that triggered this error was:\n{dataset.dataset_config}'
            )
        self.model = model
        self.dataset = dataset
        self.model_engine = model_engine
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_dataloader_workers = num_dataloader_workers
        self.iter_called = False
        self.eval_quantile = None
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        # Be careful to only create the DataLoader some bounded number of times: https://github.com/pytorch/pytorch/issues/91252
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.data = self._pull_batches_from_dataloader()

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __iter__(self):
        self.iter_called = True
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        if self.next_micro_batch == None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            if self.recreate_dataloader:
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = None
            self.epoch += 1
        return ret

    def _create_dataloader(self, skip_first_n_batches=None):
        if skip_first_n_batches is not None:
            sampler = SkipFirstNSampler(skip_first_n_batches, len(self.dataset))
        else:
            sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=False,
            batch_size=None,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            persistent_workers=(self.num_dataloader_workers > 0),
        )

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            features, label = self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)
            target, mask = label
            # The target depends on the noise, so we must broadcast it from the first stage to the last.
            # NOTE: I had to patch the pipeline parallel TrainSchedule so that the LoadMicroBatch commands
            # would line up on the first and last stage so that this doesn't deadlock.
            target = self._broadcast_target(target)
            label = (target, mask)
            self.num_batches_pulled += 1
            for micro_batch in split_batch((features, label), self.gradient_accumulation_steps):
                yield micro_batch

    def _broadcast_target(self, target):
        model_engine = self.model_engine
        if not model_engine.is_pipe_parallel:
            return target

        assert model_engine.is_first_stage() or model_engine.is_last_stage()
        grid = model_engine.grid

        src_rank = grid.stage_to_global(0)
        dest_rank = grid.stage_to_global(model_engine.num_stages - 1)
        assert src_rank in grid.pp_group
        assert dest_rank in grid.pp_group
        target = target.to('cuda')  # must be on GPU to broadcast

        if model_engine.is_first_stage():
            dist.send(target, dest_rank)
        else:
            dist.recv(target, src_rank)
        return target

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        assert not self.iter_called
        self.epoch = state_dict['epoch']
        # -1 because by preloading the next micro_batch, it's always going to have one more batch
        # pulled than the actual number of batches iterated by the caller.
        self.num_batches_pulled = state_dict['num_batches_pulled'] - 1
        self._create_dataloader(skip_first_n_batches=self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
        # Recreate the dataloader after the first pass so that it won't skip
        # batches again (we only want it to skip batches the first time).
        self.recreate_dataloader = True


class SkipFirstNSampler(torch.utils.data.Sampler):
    def __init__(self, n, dataset_length):
        super().__init__()
        self.n = n
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __iter__(self):
        for i in range(self.n, self.dataset_length):
            yield i


if __name__ == '__main__':
    from utils import common
    common.is_main_process = lambda: True
    from contextlib import contextmanager
    @contextmanager
    def _zero_first():
        yield
    common.zero_first = _zero_first

    from utils import dataset as dataset_util
    dataset_util.DEBUG = True

    from models import flux
    model = flux.CustomFluxPipeline.from_pretrained('/data2/imagegen_models/FLUX.1-dev', torch_dtype=torch.bfloat16)
    model.model_config = {'guidance': 1.0, 'dtype': torch.bfloat16}

    import toml
    dataset_manager = dataset_util.DatasetManager(model)
    with open('/home/anon/code/diffusion-pipe-configs/datasets/tiny1.toml') as f:
        dataset_config = toml.load(f)
    train_data = dataset_util.Dataset(dataset_config, model)
    dataset_manager.register(train_data)
    dataset_manager.cache()

    train_data.post_init(data_parallel_rank=0, data_parallel_world_size=1, per_device_batch_size=1, gradient_accumulation_steps=2)
    print(f'Dataset length: {len(train_data)}')

    for item in train_data:
        pass