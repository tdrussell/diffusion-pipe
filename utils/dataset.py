from pathlib import Path
import os.path
import random
from collections import defaultdict
import math
import os
import hashlib
import json

import numpy as np
import torch
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from datasets.fingerprint import Hasher
from PIL import Image
import imageio
import multiprocess as mp

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple


DEBUG = False
IMAGE_SIZE_ROUND_TO_MULTIPLE = 32
NUM_PROC = min(8, os.cpu_count())


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


def shuffle_captions(captions: list[str], count: int = 0, delimiter: str = ', ', caption_prefix: str = '') -> list[str]:
    if count == 0: return captions

    def shuffle_caption(caption: str, delimiter: str = ", ") -> str:
        split = caption.split(delimiter)
        random.shuffle(split)
        return delimiter.join(split)

    return [caption_prefix + shuffle_caption(caption, delimiter) for caption in captions for _ in range(count)]


def process_caption_fn(shuffle_tags=False, caption_prefix=''):
    def fn(example):
        with open(example['caption_file']) as f:
            caption = f.read().strip()
        if shuffle_tags:
            tags = [tag.strip() for tag in caption.split(',')]
            random.shuffle(tags)
            caption = ', '.join(tags)
        caption = caption_prefix + caption

        example['caption'] = caption
        return example
    return fn


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
    )
    dataset.set_format('torch')
    return dataset


class TextEmbeddingDataset:
    def __init__(self, te_dataset):
        self.te_dataset = te_dataset
        self.image_file_to_te_idx = defaultdict(list)
        for i, image_file in enumerate(te_dataset['image_file']):
            self.image_file_to_te_idx[image_file].append(i)

    def get_text_embeddings(self, image_file, caption_number):
        return self.te_dataset[self.image_file_to_te_idx[image_file][caption_number]]


def _cache_text_embeddings(metadata_dataset, map_fn, i, cache_dir, regenerate_cache, caching_batch_size):

    def flatten_captions(example):
        image_file_out, caption_out, is_video_out = [], [], []
        for image_file, captions, is_video in zip(example['image_file'], example['caption'], example['is_video']):
            for caption in captions:
                image_file_out.append(image_file)
                caption_out.append(caption)
                is_video_out.append(is_video)
        return {'image_file': image_file_out, 'caption': caption_out, 'is_video': is_video_out}

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
        self.cache_dir = self.path / 'cache' / self.model_name / f'cache_{size_bucket[0]}x{size_bucket[1]}x{size_bucket[2]}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_embedding_datasets = []
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
        for example in self.latent_dataset.select_columns(['image_file', 'caption']):
            image_file = example['image_file']
            captions = example['caption']
            for i, caption in enumerate([captions[i:i + self.shuffle_skip] for i in range(0, len(captions), self.shuffle_skip)]):
                iteration_order.append((image_file, caption, i))
        # Shuffle again, since one media file can produce multiple training examples. E.g. video, or maybe
        # in the future data augmentation. Don't need to shuffle text embeddings since those are looked
        # up by image file name.
        shuffle_with_seed(iteration_order, 42)
        self.iteration_order = iteration_order
        self.image_file_to_latents_idx = {
            image_file: i
            for i, image_file in enumerate(self.latent_dataset['image_file'])
        }


    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.size_bucket}')
        te_dataset = _cache_text_embeddings(self.metadata_dataset, map_fn, i, self.cache_dir, regenerate_cache, caching_batch_size)
        self.text_embedding_datasets.append(te_dataset)

    def add_text_embedding_dataset(self, te_dataset):
        self.text_embedding_datasets.append(te_dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.iteration_order)
        image_file, caption, caption_number = self.iteration_order[idx]
        ret = self.latent_dataset[self.image_file_to_latents_idx[image_file]]
        if DEBUG:
            print(Path(image_file).stem)
        offset = random.randrange(self.shuffle_skip)
        caption_idx = (caption_number*self.shuffle_skip) + offset
        for ds in self.text_embedding_datasets:
            ret.update(ds.get_text_embeddings(image_file, caption_idx))
        ret['caption'] = caption[caption_idx]
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
        self.cache_dir = self.path / 'cache' / self.model_name / f'ar_frames_{self.ar_frames[0]:.3f}_{self.ar_frames[1]}'
        os.makedirs(self.cache_dir, exist_ok=True)

        for res in resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_nearest_multiple(w, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            h = round_to_nearest_multiple(h, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            size_bucket = (w, h, self.ar_frames[1])
            metadata_with_size_bucket = self.metadata_dataset.map(lambda example: {'size_bucket': size_bucket}, keep_in_memory=True)
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
        else:
            self.resolutions = self._process_user_provided_resolutions(
                directory_config.get('resolutions', dataset_config['resolutions'])
            )
        self.shuffle = directory_config.get('cache_shuffle_num', dataset_config.get('cache_shuffle_num', 0))
        self.directory_config['cache_shuffle_num'] = self.shuffle # Make accessible if it wasn't yet, for picking one out
        self.shuffle_delimiter = directory_config.get('cache_shuffle_delimiter', dataset_config.get('cache_shuffle_delimiter', ", "))
        self.path = Path(self.directory_config['path'])
        self.mask_path = Path(self.directory_config['mask_path']) if 'mask_path' in self.directory_config else None
        # For testing. Default if a mask is missing.
        self.default_mask_file = Path(self.directory_config['default_mask_file']) if 'default_mask_file' in self.directory_config else None
        self.cache_dir = self.path / 'cache' / self.model_name

        if not self.path.exists() or not self.path.is_dir():
            raise RuntimeError(f'Invalid path: {self.path}')
        if self.mask_path is not None and (not self.mask_path.exists() or not self.mask_path.is_dir()):
            raise RuntimeError(f'Invalid mask_path: {self.mask_path}')
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

    def cache_metadata(self, regenerate_cache=False):
        files = list(self.path.glob('*'))
        # deterministic order
        files.sort()

        # Mask can have any extension, it just needs to have the same stem as the image.
        mask_file_stems = {path.stem: path for path in self.mask_path.glob('*') if path.is_file()} if self.mask_path is not None else {}

        image_files = []
        caption_files = []
        mask_files = []
        for file in files:
            if not file.is_file() or file.suffix == '.txt' or file.suffix == '.npz' or file.suffix == '.json':
                continue
            image_file = file
            caption_file = image_file.with_suffix('.txt')
            if not os.path.exists(caption_file):
                caption_file = ''
            image_files.append(str(image_file))
            caption_files.append(str(caption_file))
            if image_file.stem in mask_file_stems:
                mask_files.append(str(mask_file_stems[image_file.stem]))
            elif self.default_mask_file is not None:
                mask_files.append(str(self.default_mask_file))
            else:
                if self.mask_path is not None:
                    logger.warning(f'No mask file was found for image {image_file}, not using mask.')
                mask_files.append(None)
        assert len(image_files) > 0, f'Directory {self.path} had no images/videos!'

        metadata_dataset = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files, 'mask_file': mask_files})
        # Shuffle the data. Use a deterministic seed, so the dataset is identical on all processes.
        # Seed is based on the hash of the directory path, so that if directories have the same set of images, they are shuffled differently.
        seed = int(hashlib.md5(str.encode(str(self.path))).hexdigest(), 16) % int(1e9)
        metadata_dataset = metadata_dataset.shuffle(seed=seed)
        metadata_map_fn = self._metadata_map_fn()
        fingerprint = Hasher.hash([metadata_dataset._fingerprint, metadata_map_fn])
        print('caching metadata')
        metadata_dataset = metadata_dataset.map(
            metadata_map_fn,
            cache_file_name=str(self.cache_dir / f'metadata/metadata_{fingerprint}.arrow'),
            load_from_cache_file=(not regenerate_cache),
            batched=True,
            batch_size=1,
            num_proc=NUM_PROC,
            remove_columns=metadata_dataset.column_names,
        )

        grouped_metadata = defaultdict(lambda: defaultdict(list))
        for example in metadata_dataset:
            if self.use_size_buckets:
                grouping_key = tuple(example['size_bucket'])
            else:
                grouping_key = example['ar_bucket']
                grouping_key = (grouping_key[0], int(grouping_key[1]))
            d = grouped_metadata[grouping_key]
            for k, v in example.items():
                d[k].append(v)

        if self.use_size_buckets:
            self.size_bucket_datasets = []
            for size_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                self.size_bucket_datasets.append(
                    SizeBucketDataset(
                        metadata,
                        self.directory_config,
                        size_bucket,
                        self.model_name,
                    )
                )
        else:
            self.ar_bucket_datasets = []
            for ar_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                self.ar_bucket_datasets.append(
                    ARBucketDataset(
                        ar_bucket,
                        self.resolutions,
                        metadata,
                        self.directory_config,
                        self.model_name,
                    )
                )

    def _set_defaults(self, directory_config, dataset_config):
        directory_config.setdefault('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        directory_config.setdefault('shuffle_tags', dataset_config.get('shuffle_tags', False))
        directory_config.setdefault('caption_prefix', dataset_config.get('caption_prefix', ''))
        directory_config.setdefault('num_repeats', dataset_config.get('num_repeats', 1))

    def _metadata_map_fn(self):
        captions_file = self.path / 'captions.json'
        if captions_file.exists():
            with open(captions_file) as f:
                caption_data = json.load(f)
        else:
            caption_data = None

        def fn(example):
            # batch size always 1
            caption_file = example['caption_file'][0]
            image_file = example['image_file'][0]
            captions = None
            if caption_data is not None:
                captions = caption_data.get(Path(image_file).name, None)
                if captions is None:
                    logger.warning(f'Image file {image_file} does not have an entry in captions.json')
                else:
                    assert isinstance(captions, list), 'captions.json must contain lists of captions'
            if captions is None and caption_file:
                with open(caption_file) as f:
                    captions = [f.read().strip()]
            if captions is None:
                captions = ['']
                logger.warning(f'Cound not find caption for {image_file}. Using empty caption.')
            if self.directory_config['shuffle_tags'] and self.shuffle == 0: # backwards compatibility
                self.shuffle = 1
            captions = shuffle_captions(captions, self.shuffle, self.shuffle_delimiter, self.directory_config['caption_prefix'])
            empty_return = {'image_file': [], 'mask_file': [], 'caption': [], 'ar_bucket': [], 'size_bucket': [], 'is_video': []}

            image_file = Path(image_file)
            if image_file.suffix == '.webp':
                frames = imageio.get_reader(image_file).get_length()
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
                    meta = imageio.v3.immeta(image_file)
                    first_frame = next(imageio.v3.imiter(image_file))
                    height, width = first_frame.shape[:2]
                    assert self.framerate is not None, "Need model framerate but don't have it. This shouldn't happen. Is the framerate attribute on the model set?"
                    frames = int(self.framerate * meta['duration'])
                else:
                    pil_img = Image.open(image_file)
                    width, height = pil_img.size
                    frames = 1
            except Exception:
                logger.warning(f'Media file {image_file} could not be opened. Skipping.')
                return empty_return
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

            return {
                'image_file': [str(image_file)],
                'mask_file': [example['mask_file'][0]],
                'caption': [captions],
                'ar_bucket': [ar_bucket],
                'size_bucket': [size_bucket],
                'is_video': [is_video]
            }

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
        datasets = self.size_bucket_datasets if self.use_size_buckets else self.ar_bucket_datasets
        for ds in datasets:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)


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

    def cache_metadata(self, regenerate_cache=False):
        for ds in self.directory_datasets:
            ds.cache_metadata(regenerate_cache=regenerate_cache)

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        for ds in self.directory_datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        for ds in self.directory_datasets:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)


def _cache_fn(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, caching_batch_size):
    # Dataset map() starts a bunch of processes. Make sure torch uses a limited number of threads
    # to avoid CPU contention.
    # TODO: if we ever change Datasets map to use spawn instead of fork, this might not work.
    #torch.set_num_threads(os.cpu_count() // NUM_PROC)
    # HF Datasets map can randomly hang if this is greater than one (???)
    # See https://github.com/pytorch/pytorch/issues/10996
    # Alternatively, we could try fixing this by using spawn instead of fork.
    torch.set_num_threads(1)

    for ds in datasets:
        ds.cache_metadata(regenerate_cache=regenerate_cache)

    def latents_map_fn(example):
        first_size_bucket = example['size_bucket'][0]
        tensors_and_masks = []
        image_files = []
        captions = []
        for path, mask_path, size_bucket, caption in zip(example['image_file'], example['mask_file'], example['size_bucket'], example['caption']):
            assert size_bucket == first_size_bucket
            items = preprocess_media_file_fn(path, mask_path, size_bucket)
            tensors_and_masks.extend(items)
            image_files.extend([path] * len(items))
            captions.extend([caption] * len(items))

        if len(tensors_and_masks) == 0:
            return {'latents': [], 'mask': [], 'image_file': [], 'caption': []}

        caching_batch_size = len(example['image_file'])
        results = defaultdict(list)
        for i in range(0, len(tensors_and_masks), caching_batch_size):
            tensors = [t[0] for t in tensors_and_masks[i:i+caching_batch_size]]
            batched = torch.stack(tensors)
            parent_conn, child_conn = mp.Pipe(duplex=False)
            queue.put((0, batched, child_conn))
            result = parent_conn.recv()  # dict
            for k, v in result.items():
                results[k].append(v)
        # concatenate the list of tensors at each key into one batched tensor
        for k, v in results.items():
            results[k] = torch.cat(v)
        results['image_file'] = image_files
        results['mask'] = [t[1] for t in tensors_and_masks]
        results['caption'] = captions
        return results

    for ds in datasets:
        ds.cache_latents(latents_map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    for text_encoder_idx in range(num_text_encoders):
        def text_embedding_map_fn(example):
            parent_conn, child_conn = mp.Pipe(duplex=False)
            queue.put((text_encoder_idx+1, example['caption'], example['is_video'], child_conn))
            result = parent_conn.recv()  # dict
            result['image_file'] = example['image_file']
            return result
        for ds in datasets:
            ds.cache_text_embeddings(text_embedding_map_fn, text_encoder_idx+1, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    # signal that we're done
    queue.put(None)


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model, regenerate_cache=False, caching_batch_size=1):
        self.model = model
        self.vae = self.model.get_vae()
        self.text_encoders = self.model.get_text_encoders()
        self.submodels = [self.vae] + list(self.text_encoders)
        self.call_vae_fn = self.model.get_call_vae_fn(self.vae)
        self.call_text_encoder_fns = [self.model.get_call_text_encoder_fn(text_encoder) for text_encoder in self.text_encoders]
        self.regenerate_cache = regenerate_cache
        self.caching_batch_size = caching_batch_size
        self.datasets = []

    def register(self, dataset):
        self.datasets.append(dataset)

    # Some notes for myself:
    # Use a manager queue, since that can be pickled and unpickled, and sent to other processes.
    # IMPORTANT: we use multiprocess library (not Python multiprocessing!) just like HF Datasets does.
    # After hours of debugging and looking up related issues, I have concluded multiprocessing is outright bugged
    # for this use case. Something about making a manager queue and sending it to the caching process, and then
    # further sending it to map() workers via the pickled map function, is broken. It gets through a lot of the caching,
    # but eventually, inevitably, queue.put() will fail with BrokenPipeError. Switching from multiprocessing to multiprocess,
    # which has basically the same API, and everything works perfectly. ¯\_(ツ)_/¯
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
            ds.cache_metadata()
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
            tensor, pipe = task[1:]
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
            pin_memory=True,
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